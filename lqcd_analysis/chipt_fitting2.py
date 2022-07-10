import pandas as pd
import numpy as np
import gvar as gv
import functools
from collections import namedtuple
import inspect
from tqdm import tqdm
import lsqfit
from . import analysis
from . import serialize
from . import chipt
from . import su2
from . import staggered
from . import dataset
from . import chipt_fitting as fitting
from allhisq_analysis import data_tables

def main():


    pass

def bundle_mask(dataframe, **kwargs):
    return functools.reduce(lambda a, b: a&b,
                  [dataframe[key] == value for key, value in kwargs.items()])


def apply_implicit(df, fcn):
    """
    Apply a function to a dataframe, implicitly finding the columns matching
    the argument names of the function.
    """
    arg_names = list(inspect.signature(fcn).parameters.keys())
    for col in arg_names:
        if col not in df.columns:
            raise ValueError("Missing column", col, "?")
    return df[arg_names].apply(lambda args: fcn(*args), axis=1)


def get_masses(engine, ens_id, process, alias_heavy):
    """
    Reads masses from the database using 2pt fit results.
    """
    hadron_masses = pd.read_sql("SELECT * FROM hadron_masses;", engine)
    for key in ['pion', 'kaon', 'd', 'ds']:
        hadron_masses[key] = hadron_masses[key].apply(gv.gvar)
    mask =\
        (hadron_masses['alias_heavy'] == alias_heavy) &\
        (hadron_masses['ens_id'] == ens_id)
    hadron_masses = hadron_masses[mask]
    if process in ['Ds to K', 'Ds2K']:
        M_daughter = hadron_masses['kaon'].item()
        M_mother = hadron_masses['ds'].item()
    elif process in ['D to K', 'D2K']:
        M_daughter = hadron_masses['kaon'].item()
        M_mother = hadron_masses['d'].item()
    elif process in ['D to pi', 'D2pi']:
        M_daughter = hadron_masses['pion'].item()
        M_mother = hadron_masses['d'].item()
    else:
        raise ValueError("Unrecognized process", process)
    return gv.mean(M_daughter), gv.mean(M_mother)


def fparallel_pcvc(MH, ML, EL, p2, f0, fperp):
    """
    Computes fparallel in terms of f0 and fperp using PCVC.
    """
    return (MH**2 - ML**2)*f0/np.sqrt(2*MH)/(MH - EL) - p2*fperp/(MH-EL)


def fperp_pcvc(MH, ML, EL, p2, f0, fparallel, df0dp2, dfparalleldp2):
    """
    Computes fperp in terms of f0 and fparallel using PCVC.
    Uses the derivatives of f0 and fparallel to compute the value at p2=0,
    where the usual formula is defined only via a limit as p2 approaches 0.
    """
    if p2 != 0:
        return -(MH - EL)*fparallel/p2 + (MH**2 - ML**2)*f0/np.sqrt(2*MH)/p2
    # return np.nan
    return -(MH - ML)*dfparalleldp2 + (MH**2 - ML**2)*df0dp2/np.sqrt(2*MH) + fparallel/(2*ML)


def f0_pcvc(MH, ML, EL, fperp, fparallel):
    """
    Computes f0 in terms of fperp and fparallel using PCVC.
    """
    return np.sqrt(2*MH)/(MH**2 - ML**2) * ((MH - EL)*fparallel + (EL**2 - ML**2)*fperp)


def estimate_derivatives(dataframe):
    """
    TODO: doc here
    """
    # Align form factor and momentum values on each draw
    mask = dataframe['momentum'].isin(['p000','p100'])
    cols = ['draw_number','momentum', 'f0','fparallel','p2']
    pivot = dataframe[mask][cols].pivot(
        index=['draw_number'],
        columns='momentum',
        values=['f0','fparallel','p2'])
    # Compute the derivatives
    dp2 = pivot['p2']['p100'] - pivot['p2']['p000']
    df0 = pivot['f0']['p100'] - pivot['f0']['p000']
    dfparallel = pivot['fparallel']['p100'] - pivot['fparallel']['p000']
    pivot['df0dp2'] = df0/dp2
    pivot['dfparalleldp2'] = dfparallel/dp2
    pivot.reset_index(inplace=True)
    # Restrict to output columns
    pivot = pivot[['draw_number','df0dp2','dfparalleldp2']]
    # replace MultiIndex columns
    pivot.columns = ['draw_number','df0dp2','dfparalleldp2']
    pivot['momentum'] = 'p000'
    return pivot


def correct_covariance(adict, bstrap=True, ndraw=500):
    """ Corrects the covariance matrix by applying nonlinear shrinkage. """
    # Combine without shrinkage
    ds = gv.dataset.avg_data(adict, bstrap=bstrap)
    # Isolate mean and standard error
    mean = gv.mean(ds)
    err = gv.sdev(ds)
    # Apply shrinkage
    # Critical: For consistent reassembly, the order of the rows must match
    # the order of keys indexing the errors.
    keys = list(err.keys())
    arr = np.array([adict[key] for key in keys])
    # Each row is different observable
    # Each column is a different bootstap sample
    # Filter out any bootstrap samples that involve nans
    bad = np.any(np.isnan(arr), axis=0)
    nbad = np.sum(bad)
    if nbad > 0:
        print("Dropping", nbad)
        arr = arr[:, ~bad]
    _, corr = dataset.nonlinear_shrink(arr.T, ndraw)
    cov = {}
    for ii in range(len(ds)):
        for jj in range(len(ds)):
            key_ii = keys[ii]
            key_jj = keys[jj]
            cov[key_ii, key_jj] = err[key_ii] * corr[ii, jj] * err[key_jj]
    # Combine means with corrected covariance matrix
    return gv.gvar(mean, cov)


def align_form_factors(engine, process, dataframe):
    """
    Align form factors, masses, and energies on each draw.
    Estimate the derivatives "df/dp2" at p000.
    """
    keys = ['ens_id', 'description', 'ns', 'm_heavy', 'm_light', 'm_spectator',
            'alias_heavy', 'alias_light', 'alias_spectator', 'dm_heavy']
    Tags = namedtuple('Tags', keys)
    dfs = []
    for values, df in dataframe.groupby(keys):
        tags = Tags(*values)
        # Line up form factors
        pivot = df.pivot(index=['draw_number','momentum'], columns=['spin_taste_current'], values='form_factor')
        pivot.reset_index(inplace=True)
        pivot.rename_axis(None, axis=1, inplace=True)
        pivot.rename(columns={'S-S':'f0', 'V4-V4':'fparallel', 'Vi-S': 'fperp'}, inplace=True)
        # Carry along other values
        for key, value in zip(keys, values):
            pivot[key] = value
        pivot['p2'] = pivot[['momentum','ns']].apply(lambda args: analysis.p2(*args), axis=1)
        # Grab masses / energies
        ML, MH = get_masses(engine, tags.ens_id, process, tags.alias_heavy)
        pivot['MH'] = MH
        pivot['ML'] = ML
        pivot['EL'] = np.sqrt(pivot['ML']**2 + pivot['p2'])
        # Estimate derivatives
        dfdp2 = estimate_derivatives(pivot)
        pivot = pd.merge(pivot, dfdp2, how='left', on=['momentum', 'draw_number'])
        dfs.append(pivot)
    df = pd.concat(dfs)
    return df


def combine_bootstrap(engine, process, dataframe, shrink_choice='nonlinear', svdcut=1e-2):
    """ doc here """
    if shrink_choice != 'nonlinear':
        raise NotImplementedError

    df = align_form_factors(engine, process, dataframe)
    # Use PCVC to compute alternative estimates of the form factors
    df['fparallel2'] = apply_implicit(df, fparallel_pcvc)
    df['fperp2'] = apply_implicit(df, fperp_pcvc)
    df['f02'] = apply_implicit(df, f0_pcvc)

    # Compute correlated bootstrap averages
    avg = []
    keys1 = ['ens_id', 'description', 'ns']
    Tags1 = namedtuple('Tags1', keys1)
    keys2 = ['form_factor_type', 'momentum', 'alias_heavy', 'alias_light', 'alias_spectator']
    Tags2 = namedtuple('Tags2', keys2)
    for tags1, subdf in df.groupby(keys1):
        tags1 = Tags1(*tags1)
        # Correlates pairs of form factors
        # TODO: Generalize to different variations
        pairs = [['f0', 'f02'], ['fperp', 'fperp2'], ['fparallel', 'fparallel2']]
        for pair in pairs:
            pivot = subdf.pivot(
                index='draw_number',
                columns=['momentum', 'alias_heavy', 'alias_light', 'alias_spectator'],
                values=pair)
            pivot.dropna(axis=1, how='all', inplace=True)
            adict = {key: pivot[key].values for key in pivot.columns}
            adict = correct_covariance(adict, bstrap=True, ndraw=len(pivot))
            if svdcut is not None:
                print(f"--> Apply svdcut={svdcut}")
                adict = gv.svd(adict, svdcut=svdcut)
            # Unpackage results
            for tags2, form_factor in adict.items():
                # Level 2
                tags2 = Tags2(*tags2)
                payload = tags2._asdict()
                payload['form_factor'] = form_factor
                # Level 1
                for tag, value in tags1._asdict().items():
                    payload[tag] = value
                # Extras
                mask = bundle_mask(subdf,
                    alias_heavy=tags2.alias_heavy,
                    alias_light=tags2.alias_light,
                    alias_spectator=tags2.alias_spectator)
                extras = ['m_heavy', 'm_spectator', 'dm_heavy']
                for extra in extras:
                    payload[extra] = subdf[mask][extra].unique().item()
                avg.append(payload)
    avg = pd.DataFrame(avg)
    avg['p2'] = avg[['momentum','ns']].apply(lambda args: analysis.p2(*args), axis=1)
    avg['phat2'] = avg['momentum'].apply(analysis.phat2)

    return avg


def read_all(process, engine, shrink_choice='nonlinear', svdcut=None, inflate=1.0):
    """
    Reads all form factor data for decays of D-mesons.
    """
    if inflate != 1.0:
        raise NotImplementedError

    dfs = []
    # for ens_id in [15]:
    # for ens_id in [25]: #, 28, 13, 12, 36, 35]:
    # for ens_id in [25, 15, 28, 13, 12, 35]:
    # for ens_id in [25, 28, 13, 12, 36, 35]:
    for ens_id in [25, 15, 28, 13, 12, 36, 35]:
        if process in ('Ds2K', 'Ds to K', 'D2pi', 'D to pi'):
            alias_spectator = '1.0 m_strange'
        elif process in ('D2K', 'D to K'):
            if ens_id in [25, 15, 28]:
                alias_spectator = '1.0 m_light'
            elif ens_id in [13, 12]:
                alias_spectator = '0.1 m_strange'
            elif ens_id in [35, 36]:
                alias_spectator = '0.2 m_strange'
        else:
            raise NotImplemented

        df = read_data(engine, ens_id, process, alias_spectator, shrink_choice, svdcut)
        # Read masses from 2pt fits
        hadron_masses = pd.read_sql("SELECT * FROM hadron_masses;", engine)
        for key in ['pion', 'kaon', 'd', 'ds']:
            hadron_masses[key] = hadron_masses[key].apply(gv.gvar)
        df = pd.merge(df, hadron_masses, on=['ens_id', 'alias_heavy', 'm_heavy'])
        if process in ['Ds to K', 'Ds2K']:
            df['M_daughter'] = df['kaon']
            df['M_mother'] = df['ds']
        elif process in ['D to K', 'D2K']:
            df['M_daughter'] = df['kaon']
            df['M_mother'] = df['d']
        elif process in ['D to pi', 'D2pi']:
            df['M_daughter'] = df['pion']
            df['M_mother'] = df['d']
        else:
            raise ValueError("Unrecognized process", process)
        dfs.append(df)

    data = pd.concat(dfs)
    # Isolate energies for later convenience
    data['E_daughter'] = np.sqrt(data['M_daughter']**2 + data['p2'])
    data['E_mother'] = data['M_mother']

    # Read sea masses
    print("Reading quark sea masses.")
    sea_masses = fitting.read_sea_masses(engine)
    data = pd.merge(data, sea_masses[['ens_id', 'm_light', 'm_strange']], on='ens_id')

    # Read lattice spacing
    print("Reading lattice spacing.")
    lattice_spacing = pd.read_sql("select ens_id, a_fm from lattice_spacing;", engine)
    data = pd.merge(data, lattice_spacing, on='ens_id')

    # Read scale-setting data
    print("Reading Wilson-flow scale.")
    scale = data_tables.ScaleSetting().data
    scale = scale[['a[fm]','description','w0_orig/a']]
    scale.rename(columns={'a[fm]': 'a_fm'}, inplace=True)
    data = pd.merge(data, scale, on=['a_fm', 'description'])

    # Convert form factors to w0 units
    print("Expressing fperp and fparallel as dimensionless ratios with w0.")
    for ff_type in ['fperp', 'fperp2']:
        mask = (data['form_factor_type'] == ff_type)
        data.loc[mask, 'form_factor'] = data[mask]['form_factor'] / np.sqrt(data['w0_orig/a'])
    for ff_type in ['fparallel', 'fparallel2']:
        mask = (data['form_factor_type'] == ff_type)
        data.loc[mask, 'form_factor'] = data[mask]['form_factor'] * np.sqrt(data['w0_orig/a'])

    # Apply HQET correction factor to restore normalization
    print("Applying HQET correction factor to restore normalization")
    data['form_factor'] = data['form_factor'] * data['m_heavy'].apply(staggered.chfac)
    return data


def read_data(engine, ens_id, process, alias_spectator='1.0 m_strange', shrink_choice='nonlinear', svdcut=None):
    """
    Reads data from the database for all currents (i.e., all matrix elements)
    from a single ensemble for a given decay process.
    """
    if (process in ('D to K', 'D2K')) and (alias_spectator == '1.0 m_strange'):
        raise ValueError("Incorrect spectator for D2K?", alias_spectator)

    print(f" Reading ens_id={ens_id} ".center(60, "#"))
    dfs = []
    for current in ['S-S', 'Vi-S', 'V4-V4']:
        print("Reading", current)
        df = fitting.read_boot_data(engine, ens_id, current, process)
        if current in ['Vi-S', 'V4-V4']:
            print("--> Renormalizing", current)
            z = fitting.read_boot_renormalization(engine, ens_id, current, alias_spectator)
            df = pd.merge(df, z, on=['ens_id','m_heavy','m_light', 'draw_number'])
            df['form_factor'] = df['form_factor']*df['Z']
        else:
            print("--> Absolutedly normalized", current)
        print("--> Found", len(df), "total lines for", current)
        dfs.append(df)
    df = pd.concat(dfs)
    df['form_factor'] = df['form_factor'] * np.sign(df['form_factor'])
    for col in ['energy_src','energy_snk']:
        df[col] = df[col].apply(float)
    df['phat2'] = df['momentum'].apply(analysis.phat2)
    df['p2']= df[['momentum','ns']].apply(lambda args: analysis.p2(*args), axis=1)
    print("Total size before bootstrap averaging", len(df))
    df = combine_bootstrap(engine, process, df, shrink_choice, svdcut)
    print("Total size after bootstrap averaging", len(df))
    return df

def run_fits(process, data, noise_cut=0.2, **kwargs):
    """
    Runs all the fits for fperp, fparallel, and f0.
    """
    svdcuts = kwargs.pop('svdcuts', None)

    if process in ['D to pi', 'D to K']:
        mother_name = 'D'
    elif process in ['Ds to K']:
        mother_name = 'Ds'
    else:
        raise ValueError("Unexpected process", process)

    if noise_cut is not None:
        print("Size before noise cut", len(data))
        data = data[np.abs(data['form_factor'].apply(analysis.n2s)) < noise_cut]
        print("Size after noise cut", len(data))
    scale = data_tables.ScaleSetting()
    ctm = data_tables.ContinuumConstants()
    lam = gv.mean(700 * scale.w0_fm / ctm.hbarc)

    models = {
        'f_0': {
            'HardSU2': su2.HardSU2Model,
            'HardSU2:continuum': su2.HardSU2Model,
            'LogLess': chipt.LogLessModel,
        },
        'f_perp': {
            'HardSU2': su2.HardSU2Model,
            'HardSU2:continuum': su2.HardSU2Model,
            'LogLess': chipt.LogLessModel,
        },
        'f_parallel': {
            'HardSU2': su2.HardSU2Model,
            'HardSU2:continuum': su2.HardSU2Model,
            'LogLess': chipt.LogLessModel,
        },
    }

    def is_base_fit(channel, model_name, mask_label, label):
        if (channel == 'f_perp') & (model_name == 'HardSU2') & (mask_label == 'full') & (label == 'NNLO'):
            return True
        if (channel == 'f_parallel') & (model_name == 'HardSU2') & (mask_label == 'full') & (label == 'NNLO'):
            return True
        if (channel == 'f_0') & (model_name == 'HardSU2') & (mask_label == 'full') & (label == 'NNLO'):
            return True
        return False


    results = []
    for channel in ['f_perp', 'f_parallel', 'f_0']:
        for suffix in ['', '2']:
            form_factor_type = f"{channel.replace('_', '')}{suffix}"
            print("Running fits for", form_factor_type)
            for model_name, model_fcn in models[channel].items():
                print("Starting fits for", model_name)

                # Define models
                if model_name == 'LogLess':
                    model = model_fcn(channel, process, lam=lam)
                else:
                    if ('continuum' in model_name):
                        continuum_logs = True
                    else:
                        continuum_logs = False
                    if channel == 'f_0':
                        _channel = 'f_parallel'
                    else:
                        _channel = channel
                    model = model_fcn(_channel, process, lam=lam, continuum_logs=continuum_logs)

                wrapped = fitting.WrappedModel(model)
                model_continuum = model_fcn(channel, process, lam=lam, continuum=True)
                continuum = fitting.ContinuumLimit(model.process)

                # Masks for dropping parts of the dataset
                masks = {
                    'full': data['a_fm'] > 0,  # trivially true by definition. The full dataset.
                    'omit 0.12 fm': data['a_fm'] != 0.12,  # drop the coarsest lattice spacing
                    'omit 0.042 fm': data['a_fm'] != 0.042,  # drop the finest lattice spacing
                    'mh/mc <= 1.5': ~data['alias_heavy'].isin(['2.0 m_charm', '2.2 m_charm']),
                    'mh/mc <= 1.1': ~data['alias_heavy'].isin(['1.4 m_charm', '1.5 m_charm', '2.0 m_charm', '2.2 m_charm']),
                    'mh/mc = 1.0 only': data['alias_heavy'] == '1.0 m_charm',
                    'physical pions only': data['description'] == '1/27',
                }
                for mask_label, mask in masks.items():
                    # Build data
                    mask = mask & (data['form_factor_type'] == form_factor_type)
                    x, y_data = fitting.build_fit_data(data[mask], mother_name=mother_name)

                    # Run variations on the model
                    priors = fitting.ModelVariations(model.process, model_name).priors
                    for label, prior in tqdm(priors.items()):
                        if (label not in ('NNLO',)) & (mask_label not in ('full', 'mh/mc <= 1.1')):
                        # if (label != 'NNLO') & (mask_label not in ('full', )):
                            # Keep: full data and NNLO
                            # Keep: full data and model variation
                            # Keep: drop data and NNLO
                            # Skip: drop data and model variation simultaneously
                            continue

                        if svdcuts is not None:
                            if form_factor_type in svdcuts:
                                svdcut = svdcuts[form_factor_type]
                            else:
                                svdcut = None
                            kwargs['svdcut'] = svdcut

                        # If svdcut is nonzero, vary to check stability
                        svdcut = kwargs.pop('svdcut', None)
                        if is_base_fit(channel, model_name, mask_label, label):
                            if svdcut is None:
                                run_svdcuts = [None]
                            else:
                                run_svdcuts = [svdcut * val for val in [0.1, 1, 10]]
                        else:
                            run_svdcuts = [svdcut]
                        for svdcut in run_svdcuts:
                            fit = lsqfit.nonlinear_fit(data=(x, y_data), fcn=wrapped, prior=prior, debug=True, svdcut=svdcut, **kwargs)
                            fit = serialize.SerializableNonlinearFit(fit)
                            y_ctm = model_continuum(continuum.x, fit.p)
                            result = fit.serialize()
                            result['form_factor_type'] = form_factor_type
                            result['model_name'] = model_name
                            result['model'] = model
                            result['model_ctm'] = model_continuum
                            result['continuum'] = continuum
                            result['label'] = label
                            result['dataset'] = mask_label
                            result['fit'] = fit
                            result['process'] = process
                            result['channel'] = channel
                            result['f(q2max)'] = y_ctm[0]
                            result['f(q2=0)'] = y_ctm[-1]
                            result['f(q2=middle)'] = y_ctm[len(y_ctm)//2]
                            result['f'] = y_ctm
                            result['svdcut'] = svdcut
                            results.append(result)

    return data, pd.DataFrame(results)




if __name__ == '__main__':
    main()