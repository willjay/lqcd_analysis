import re
import itertools
from tqdm import tqdm
import numpy as np
import pandas as pd
import gvar as gv
import lsqfit
import seaborn as sns
from . import analysis
from . import dataset
from . import visualize as plt
from . import serialize
from . import chipt
from . import su2
from . import su3

from allhisq_analysis import data_tables


def read_masses(engine):

    query = """
    SELECT ens_id, m_light, m_heavy, energy as mass
    FROM campaign_results_two_point
    WHERE (momentum = 'p000') AND (basename like 'P5-P5%%');"""
    df = pd.read_sql(query, engine)
    df['mass'] = df['mass'].apply(gv.gvar)
    return df


def read_sea_masses(engine):
    """
    Infers the quark sea masses from the ensemble name.
    """
    df = pd.read_sql_query("SELECT ens_id, name FROM ensemble;", engine)
    regex = re.compile(r'^l\d+f211b\d{3}m(\d+)m(\d+)m(\d+)-allHISQ')
    sea_masses = []
    for _, row in df.iterrows():
        name = row['name']
        ens_id = row['ens_id']
        match = regex.match(name)
        if match:
            tokens = match.groups()
            sea_masses.append({
                'ens_id': ens_id,
                'm_light': float(f"0.{tokens[0]}"),
                'm_strange': float(f"0.{tokens[1]}"),
                'm_charm': float(f"0.{tokens[2]}"),
            })
    return pd.DataFrame(sea_masses)


def read_boot_data(engine, ens_id, spin_taste_current, process):


    def get_mass_combinations(process):
        if process in ('Ds to K', 'Ds2K'):
            return {
                'alias_heavy': "('0.9 m_charm', '1.0 m_charm', '1.1 m_charm')",
                'alias_light': "('1.0 m_light', '0.1 m_strange', '0.2 m_strange')",
                'alias_spectator': "('1.0 m_strange')",
            }
        if process in ('D to K', 'D2K'):
            return {
                'alias_heavy': "('0.9 m_charm', '1.0 m_charm', '1.1 m_charm')",
                'alias_light': "('1.0 m_strange')",
                'alias_spectator': "('1.0 m_light', '0.1 m_strange', '0.2 m_strange')",
            }
        if process in ('D to pi', 'D2pi'):
            return {
                'alias_heavy': "('0.9 m_charm', '1.0 m_charm', '1.1 m_charm')",
                'alias_light': "('1.0 m_light', '0.1 m_strange', '0.2 m_strange')",
                'alias_spectator': "('1.0 m_light', '0.1 m_strange', '0.2 m_strange')",
            }
        raise ValueError("Unrecognized process", process)

    masses = get_mass_combinations(process)
    if process in ('D to pi', 'D2pi'):
        match = " AND (alias_light = alias_spectator)"
    else:
        match = ""
    query = (
        "SELECT * FROM form_factor"
        " JOIN ensemble USING(ens_id)"
        " JOIN alias_form_factor USING(form_factor_id)"
        " JOIN result_form_factor_bootstrap USING(form_factor_id) WHERE"
        f" (ens_id, spin_taste_current) = ({ens_id}, '{spin_taste_current}')"
        f" AND (alias_heavy in {masses['alias_heavy']})"
        f" AND (alias_light in {masses['alias_light']})"
        f" AND (alias_spectator in {masses['alias_spectator']})"
        f"{match};"
    )
    df = pd.read_sql(query, engine)
    df['form_factor'] = df['form_factor'].apply(float)

    # Include the heavy quark mass "mistuning" difference dm = m - m0.
    # Use the bare value we associate with "1.0 m_charm" as the fiducial value m0.
    if len(df) > 0:
        df_tmp = []
        mask = (df['alias_heavy'] == '1.0 m_charm')
        for (ens_id, m_charm), _ in df[mask].groupby(['ens_id','m_heavy']):
            df_tmp.append({'ens_id': ens_id, 'm_charm': m_charm})
        df_tmp = pd.DataFrame(df_tmp)
        df = pd.merge(df, df_tmp)
        df['dm_heavy'] = df['m_heavy'] - df['m_charm']
        df.drop(columns='m_charm', inplace=True)
    return df


def correlate_boot_data(dataframe):
    """
    Correlates bootstrap data.
    """
    level_1 = ['ens_id', 'description']
    level_2 = ['form_factor_id','alias_light','alias_heavy','momentum']
    output = []
    for (ens_id, description), df in dataframe.groupby(level_1):
        data = []
        boot = {k: [] for k in ['form_factor','energy_src','energy_snk','amp_src','amp_snk']}
        for tags, subdf in df.groupby(level_2):
            form_factor_id, alias_light, alias_heavy, momentum = tags
            ns = subdf['ns'].unique().item()
            for key in boot:
                boot[key].append(subdf[key].values)
            data.append({
                'ens_id': ens_id,
                'description': description,
                'form_factor_id': form_factor_id,
                'alias_light': alias_light,
                'alias_heavy': alias_heavy,
                'm_light': subdf['m_light'].unique().item(),
                'm_heavy': subdf['m_heavy'].unique().item(),
                'm_spectator': subdf['m_spectator'].unique().item(),
                'dm_heavy': subdf['dm_heavy'].unique().item(),
                'momentum': momentum,
                'p2': analysis.p2(momentum, ns),
            })

        # Correlate all data belonging to a single ensemble
        for key, value in boot.items():
            boot[key] = np.vstack(value).transpose()

        mean = gv.dataset.avg_data(boot, bstrap=True, noerror=True)
        cov = dataset.correct_covariance(boot, shrink_choice='nonlinear', bstrap=True)
        boot = gv.gvar(mean, cov)

        # Repackage result
        data = pd.DataFrame(data)
        for key, value in boot.items():
            data[key] = value
        output.append(data)

    output = pd.concat(output)

    # Rename for generality
    output.rename(inplace=True, columns={
        'energy_src': 'E_daughter',
        'energy_snk': 'E_mother',
        'amplitude_src': 'amp_daughter',
        'amplitude_snk': 'amp_mother',
    })
    return output


def read_all(current, process, engine):

    dfs = []
    for ens_id in [25, 15, 28, 13, 12]:
    # for ens_id in [25, 15, 28, 13, 12, 36]:
    # for ens_id in [11, 25, 15, 28, 13, 12]:
        df = read_boot_data(engine, ens_id, current, process)
        print(f"Read {len(df)} lines for ens_id={ens_id}")
        dfs.append(df)
    df = pd.concat(dfs)
    data = correlate_boot_data(df)
    data['phat2'] = data['momentum'].apply(analysis.phat2)
    lattice_spacing = pd.read_sql("select ens_id, a_fm from lattice_spacing;", engine)
    data = pd.merge(data, lattice_spacing, on='ens_id')

    # Read masses from 2pt fits
    masses = read_masses(engine)
    if process in ['Ds to K', 'Ds2K', 'D to pi', 'D2pi']:
        # For Ds to K, the "heavy" quark in the 2pt function corresponds to the
        # strange spectator. The "light" quark corresponds to the daughter quark
        # of the decay, which we've unfortunately also named "light" (thinking
        # "heavy-to-light" decays).
        # For D to pi, the "heavy" and "light" quark in the daughter meson are
        # degenerate. Therefore, we simply identify the "heavy" one with the spectator.
        masses.rename(columns={'mass': 'M_daughter', 'm_heavy':'m_spectator'}, inplace=True)
    elif process in ['D to K', 'D2K']:
        # For D to K, the "heavy" quark in the 2pt function corresponds to the
        # daughter strange quark. The "light" quark in the 2pt function corresponds
        # to the spectator.
        masses.rename(columns={'mass': 'M_daughter', 'm_heavy':'m_light', 'm_light': 'm_spectator'}, inplace=True)
    else:
        raise ValueError("Unrecognized process", process)
    masses.rename(columns={'mass': 'M_daughter', 'm_heavy':'m_spectator'}, inplace=True)
    data = pd.merge(data, masses, on=['ens_id', 'm_light', 'm_spectator'])

    # Read sea masses
    sea_masses = read_sea_masses(engine)
    data = pd.merge(data, sea_masses[['ens_id', 'm_strange']], on='ens_id')
    return data


class FormFactorData:
    def __init__(self, process, engine):
        self._valid_names = ['Ds to K', 'D to K', 'D to pi','Ds2K', 'D2K', 'D2pi']
        self._valid_channels = ['f_parallel', 'f_perp', 'f_scalar', 'f_plus', 'f_0']
        if process not in self._valid_names:
            raise ValueError("Unrecognized process", process)
        self.process = process
        self.engine = engine
        self._scalar = None
        self._parallel = None
        self._perp = None
        self._plus = None
        self.renormalization = data_tables.RenormalizationFactors().data
        self.renormalization.drop(columns='a_fm', inplace=True)
        self._w0 = None

    def __getitem__(self, channel):
        if channel not in self._valid_channels:
            raise ValueError("Unrecognized channel", channel)
        return self.__getattribute__(channel)

    @property
    def w0(self):
        if self._w0 is None:
            df = data_tables.ScaleSetting().data
            df = df[['a[fm]','description','w0_orig/a']]
            df.rename(columns={'a[fm]': 'a_fm'}, inplace=True)
            self._w0 = df
        return self._w0

    def get_z_factors(self, choice):
        """
        Gets the necessary Z-factors for constructing renormalized form factors.
        Args:
            choice: str, 'V4-V4' or 'Vi-S'
            alias_spectator: str, which specator mass to use. Default '1.0 m_strange'.
        Returns:
            pd.DataFrame
        Note:
        Z-factors are properties of currents and so are independent of the spectator.
        We are thus free to choose the spectator to suit our purposes.
        A heavier spectator (e.g., '1.0 m_strange') typically yields cleaner results.
        """
        if choice not in ['ZV4', 'ZVi']:
            raise ValueError("Please choose ZV4 or ZVi")
#         spectators = ['1.0 m_strange']
        if self.process in ('Ds to K', 'Ds2K', 'D to pi', 'D2pi'):
            spectators = ['1.0 m_strange']
        elif self.process in ('D to K', 'D2K'):
            spectators = ['1.0 m_light', '0.1 m_strange', '0.2 m_strange']
        else:
            raise ValueError("Unrecognized process", self.process)
        mask = (self.renormalization['alias_spectator'].isin(spectators))
        renorm = pd.DataFrame(self.renormalization[mask])
        renorm.drop(columns='alias_spectator', inplace=True)
        return renorm

    @property
    def f_scalar(self):
        if self._scalar is None:
            self._scalar = read_all('S-S', self.process, self.engine)
        return self._scalar

    @property
    def f_0(self):
        return self.f_scalar

    @property
    def f_parallel(self):
        if self._parallel is None:
            # Get bare form factors
            df = read_all('V4-V4', self.process, self.engine)
            print("Size before merging Z factors", len(df))
            # Renormalize
            renorm = self.get_z_factors('ZV4')
            df = pd.merge(df, renorm, on=['ens_id', 'alias_light', 'alias_heavy'])
            print("Size after merging Z factors", len(df))
            df['form_factor'] = df['form_factor'] * df['ZV4']
            df['form_factor'] = df['form_factor'] * np.sign(df['form_factor'])
            # Convert from lattice units to dimensionless units of w0
            # Note: [f_parallel] = +1/2
            df = pd.merge(df, self.w0, on=['a_fm', 'description'])
            print("Size after merging w0 scale", len(df))
            df['form_factor'] = df['form_factor'] * np.sqrt(df['w0_orig/a'])
            self._parallel = df
        return self._parallel

    @property
    def f_perp(self):
        if self._perp is None:
            # Get bare form factors
            df = read_all('Vi-S', self.process, self.engine)
            print("Size before merging Z factors", len(df))
            # Renormalize
            renorm = self.get_z_factors('ZVi')
            df = pd.merge(df, renorm, on=['ens_id', 'alias_light', 'alias_heavy'])
            print("Size after merging Z factors", len(df))
            df['form_factor'] = df['form_factor'] * df['ZVi']
            # Convert from lattice units to dimensionless units of w0
            # Note: [f_perp] = -1/2
            df = pd.merge(df, self.w0, on=['a_fm', 'description'])
            print("Size after merging w0 scale", len(df))
            df['form_factor'] = df['form_factor'] / np.sqrt(df['w0_orig/a'])
            self._perp = df
        return self._perp

    @property
    def f_plus(self):
        raise NotImplementedError("f_plus not yet supported")

    def visualize_correlations(self, channel):
        """
        Visualize the correlations by making heatmaps of the correlation
        and covariance matrices and by plotting their eigenvalue spectra.
        Args:
            channel: str, the name of the channel (e.g., 'f_parallel')
        Returns:
            (fig, axarr)
        """
        if channel not in self._valid_channels:
            raise ValueError("Unsupported channel", channel)
        dataframe = self.__getattribute__(channel)
        groups = dataframe.groupby('ens_id')
        ncols = len(groups)
        fig, axarr = plt.subplots(nrows=3, ncols=ncols, figsize=(5*ncols, 15))

        for idx, (ens_id, df) in enumerate(groups):
            ax_col = axarr[:, idx]
            ax1, ax2, ax3 = ax_col

            df = df.sort_values(by=['alias_light', 'alias_heavy', 'phat2'])
            corr = gv.evalcorr(df['form_factor'].values)
            sns.heatmap(corr, ax=ax1)
            cov = gv.evalcov(df['form_factor'].values)
            sns.heatmap(cov, ax=ax2)

            matrices = {'corr': corr, 'cov:full': cov, 'cov:diag': np.diag(cov)}
            markers = ['o', 's', '^']
            for (label, mat), marker in zip(matrices.items(), markers):
                if label == 'cov:diag':
                    w = mat
                else:
                    w = np.linalg.eigvals(mat)
                w = np.sort(w)[::-1]
                w /= max(w)
                plt.plot(w, ax=ax3, label=label, marker=marker)

            ax1.set_title(f"Correlation matrix: {ens_id}")
            ax2.set_title(f"Covariance matirx: {ens_id}")
            ax3.set_title(f"Eigenvalue spectra: {ens_id}")

            ax3.legend()
            ax3.set_yscale("log")

        return fig, axarr

class InputData:
    def __init__(self, a_fm, m_light, m_heavy, m_strange, p2, M_daughter, description):
        """
        TODO: doc here
        """
        scale = data_tables.ScaleSetting().data
        mask = (scale['a[fm]'] == a_fm) & (scale['description'] == description)
        w0 = gv.mean(scale[mask]['w0_orig/a'].item())
        # Convert to dimensionless units of w0
        self.m_light = m_light * w0
        self.m_heavy = m_heavy * w0
        self.m_strange = m_strange * w0
        self.mpi5 = gv.mean(M_daughter) * w0
        self.E = np.sqrt(M_daughter**2 + p2) * w0

    def asdict(self):
        return dict(self.__dict__)

    def __str__(self):
        return (f"InputData({self.a_fm}, {self.m_light}, {self.m_heavy}, {self.m_strange}, "
                f"{self.p2}, {self.E}, {self.mpi5}")

    def __getitem__(self, key):
        return self.__dict__.get(key)

    def get(self, key, default=None):
        """ Return the value for key if key is in the dictionary, else default. """
        return self.__dict__.get(key, default)


def build_fit_data(dataframe, model_type=None):
    """
    Builds dictionaries suitable for interpretation as input data
    for chiral-continuum fits with lsqfit, "data=(x,y)".
    Args:
        dataframe: pd.DataFrame containing correlated data
    Returns:
        xdict, ydict: the data dictionaries for the fit
    """
    keys = ['description', 'a_fm', 'm_light', 'm_heavy', 'dm_heavy', 'm_spectator', 'm_strange']
    groups = dataframe.groupby(keys)
    xdict, ydict = {}, {}
    for key, subdf in groups:
        (description, a_fm, m_light, m_heavy, dm_heavy, m_spectator, m_strange) = key
        subdf = subdf.sort_values(by='E_daughter')
        y = subdf['form_factor'].values
        M_daughter = subdf['M_daughter'].apply(gv.mean).unique().item()
        p2 = subdf['p2'].values
        x = InputData(a_fm, m_light, dm_heavy, m_strange, p2, M_daughter, description).asdict()

        # Include continuum constants like fpi as independent "x-parameters"
        scale = data_tables.ScaleSetting()
        ctm = data_tables.ContinuumConstants()
        fpi = ctm.pdg['fpi'] * scale.w0_fm / ctm.hbarc
        x['fpi'] = gv.mean(fpi)

        # Include staggered low-energy constants as independent "x-parameters"
        const = data_tables.StaggeredConstants().get_row(a_fm=a_fm)
        w0 = gv.mean(scale.get_row(a_fm=a_fm, description=description)['w0_orig/a'])

        # Quantities with mass dimension +1
        x['mu'] = const['mu'] * w0

        # Quantities with mass dimension +2
        for k in ['Delta_P', 'Delta_A', 'Delta_T', 'Delta_V', 'Delta_I',
                    'DeltaBar', 'Hairpin_V', 'Hairpin_A']:
            x[k] = gv.mean(const[k]) * w0**2
        x['mu'] = const['mu'] * w0

        if model_type is not None:
            if 'SU3' in model_type:
                x['mpi5'] = const['mu'] * (2 * m_light) * w0**2
                x['mK5'] = const['mu'] * (m_light + m_strange) * w0**2
                x['mS5'] = const['mu'] * (2 * m_strange) * w0**2

        # Collect results
        ydict[key] = y
        xdict[key] = x

    return xdict, ydict

class ContinuumLimit:
    def __init__(self, process):
        valid_names = ['Ds to K', 'D to K', 'D to pi', 'Ds2K', 'D2K', 'D2pi']
        if process not in valid_names:
            raise ValueError("Invalid process", process)
        self.process = process
        self.mother, self.daughter = self.get_hadron_masses()

    def get_hadron_masses(self):
        """
        Gets the hadron masses from the PDG associated with the process.
        Returns:
            mother, daughter: the hadron masses in MeV
        """
        ctm = data_tables.ContinuumConstants()
        if self.process in ('Ds to K', 'Ds2K'):
            mother = ctm.pdg['D_s']
            daughter = ctm.pdg['K']
            return mother, daughter

        if self.process in ('D to K', 'D2K'):
            mother = ctm.pdg['D']
            daughter = ctm.pdg['K']
            return mother, daughter

        if self.process in ('D to pi', 'D2pi'):
            mother = ctm.pdg['D']
            daughter = ctm.pdg['pi']
            return mother, daughter

    def get_energy_bounds(self, MeV=False):
        """
        Get the mininum and maximum energies for the daughter
        hadron for a physical process. The minimum energy corresponds
        to the daughter hadron at rest (q2=q2_max). The maximum
        energy corresponds to the daughter hadron moving with
        zero energy transfered to the leptonic system (q2=0).
        """
        scale = data_tables.ScaleSetting()
        ctm = data_tables.ContinuumConstants()
        energy_min = self.daughter
        energy_max = (self.mother**2 + self.daughter**2) / (2*self.mother)
        if not MeV:
            energy_min = energy_min * scale.w0_fm / ctm.hbarc
            energy_max = energy_max * scale.w0_fm / ctm.hbarc
        return (energy_min, energy_max)

    @property
    def x(self):
        """
        Gets input arguments "x_ctm" for use in the continuum limit
        at the physical point.
        """
        scale = data_tables.ScaleSetting()
        ctm = data_tables.ContinuumConstants()

        # Quark masses
        ml_ctm_MeV = 3.402  # Eq (5.3) of 1802.04248
        ms_ctm_MeV = 92.47  # Eq (5.2) of 1802.04248
        mc_ctm_MeV = 1090   # Eq (5.9) of 1802.04248
        ml_ctm = ml_ctm_MeV * scale.w0_fm / ctm.hbarc
        ms_ctm = ms_ctm_MeV * scale.w0_fm / ctm.hbarc
        mc_ctm = mc_ctm_MeV * scale.w0_fm / ctm.hbarc

        # The physical low-energy constant "mu"
        # Infer LEC mu from Gell-Mann Oakes Renner: Mpi**2 = 2*mu*ml
        # Note: no special treatment is required for daughter kaons here.
        # The physics reason is that the LEC "mu" is basically related
        # (up to conventional numerical constants) to the value of the
        # quark condensate.
        # Numerically, one finds
        # 2.326(21) -- from Mpi / (2*ml)
        # 2.243(21) -- from MK / (ml + ms)
        mu_ctm = ctm.pdg['pi']**2 / (2*ml_ctm_MeV) * scale.w0_fm / ctm.hbarc
        energy_min, energy_max = self.get_energy_bounds()
        f = ctm.pdg['fpi'] * scale.w0_fm / ctm.hbarc
        # f = ctm.pdg['fK'] * scale.w0_fm / ctm.hbarc
        # Note:
        # m_heavy really represents the "mistuing" dm = (m-m0) of the heavy quark.
        # By definition, this difference vanishes at the physical point.
        return {
            'fpi': gv.mean(f),
            'm_light': gv.mean(ml_ctm),
            'm_strange': gv.mean(ms_ctm),
            'm_heavy': 0,
            'E': np.linspace(gv.mean(energy_min), gv.mean(energy_max)),
            'mpi5': gv.mean(self.daughter * scale.w0_fm / ctm.hbarc),
            'mK5': mu_ctm * (ml_ctm + ms_ctm),
            'mS5': mu_ctm * (2.0 * ms_ctm),
            'mu': mu_ctm,
            'DeltaBar': 0,
        }


class ModelVariations:
    def __init__(self, process, model_name=None):
        self.process = process
        self.model_name = model_name
        self.priors = self.build_priors()

    def build_base_prior(self):
        scale = data_tables.ScaleSetting()
        ctm = data_tables.ContinuumConstants()
        delta_pole = ctm.get_delta_pole(self.process, fractional_width=1.0) * scale.w0_fm / ctm.hbarc
        prior = {
            'leading': gv.gvar(10, 10),
            'log(delta_pole)': np.log(delta_pole),
            'c_a2': gv.gvar(0.0, 1.0),
        }
        if self.model_name != 'LogLess':
            prior['g'] = gv.gvar(10, 10)
        return prior

    def build_priors(self):
        priors = {}

        nlo_terms = ['c_l', 'c_h', 'c_E', 'c_E2']
        nnlo_terms = ['c_l2', 'c_lh', 'c_lE', 'c_h2', 'c_hE']
        if 'SU3' in self.model_name:
            nlo_terms.append('c_s')
            nnlo_terms.extend(['c_sl', 'c_sh', 'c_sE', 'c_s2'])
        num_nnlo = len(nnlo_terms)

        # Minimal fit
        prior = self.build_base_prior()
        for key in ['c_h', 'c_E', 'c_E2', 'c_lE', 'c_Eh']:
            prior[key] = gv.gvar('0.0(1.0)')
        priors['minimal'] = prior

        # NLO
        prior = self.build_base_prior()
        for key in nlo_terms:
            if key.startswith('log'):
                prior[key] = np.log(gv.gvar(0.1, 1.0))
            else:
                prior[key] = gv.gvar(0.0, 1.0)
        priors['NLO'] = prior

        # Adding terms to NLO
        for key in nnlo_terms:
            prior = self.build_base_prior()
            keys = nlo_terms+ [key]
            for key in keys:
                prior[key] = gv.gvar(0.0, 1.0)
            label = 'NLO+{' + key +'}'
            priors[label] = prior
        priors['NLO'] = prior

        # Full NNLO
        keys = nlo_terms + nnlo_terms
        prior = self.build_base_prior()
        for key in keys:
            if key.startswith('log'):
                prior[key] = np.log(gv.gvar(0.1, 1.0))
            else:
                prior[key] = gv.gvar(0.0, 1.0)
        priors['NNLO'] = prior

        # Dropping terms from NNLO
        for keys in itertools.combinations(nnlo_terms, num_nnlo-1):
            keys = nlo_terms + list(keys)
            prior = self.build_base_prior()
            for key in keys:
                if key.startswith('log'):
                    prior[key] = np.log(gv.gvar(0.1, 1.0))
                else:
                    prior[key] = gv.gvar(0.0, 1.0)

            dropped = "{" + ",".join(np.setdiff1d(nlo_terms + nnlo_terms, keys)) + "}"
            label = "NNLO-" + dropped
            priors[label] = prior

        return priors


class WrappedModel:
    def __init__(self, model):
        self.model = model
    def __call__(self, x, p):
        return {key: self.model(xvalue, p) for key, xvalue in x.items()}


def run_fits(process, channel, engine):

    data = FormFactorData(process, engine)[channel]
    scale = data_tables.ScaleSetting()
    ctm = data_tables.ContinuumConstants()
    lam = gv.mean(700 * scale.w0_fm / ctm.hbarc)

    models = {
        'SU2': su2.SU2Model,
        'HardSU2': su2.HardSU2Model,
        # 'HardSU3': su3.HardSU3Model,
        'SU2:continuum': su2.SU2Model,
        'HardSU2:continuum': su2.HardSU2Model,
        'LogLess': chipt.LogLessModel,
    }

    results = []
    for model_name, model_fcn in models.items():
        print("Starting fits for", model_name)
        # Build data
        # E_daughter = data['E_daughter'] / data['a_fm'] * ctm.hbarc
#         cut = (E_daughter < 1050)  # i.e., < 1050 MeV
#         x, y_data = build_fit_data(data[cut], model_name)
        x, y_data = build_fit_data(data, model_name)

        # Define models
        if model_name == 'LogLess':
            model = model_fcn(channel, process, lam=lam)
        else:
            if ('continuum' in model_name):
                continuum_logs = True
            else:
                continuum_logs = False
            model = model_fcn(channel, process, lam=lam, continuum_logs=continuum_logs)

        wrapped = WrappedModel(model)
        model_continuum = model_fcn(channel, process, lam=lam, continuum=True)
        wrapped_continuum = WrappedModel(model_continuum)
        continuum = ContinuumLimit(model.process)

        # run the fit
        priors = ModelVariations(model.process, model_name).priors
        for label, prior in tqdm(priors.items()):
            fit = lsqfit.nonlinear_fit(data=(x, y_data), fcn=wrapped, prior=prior, debug=True)#, svdcut=1e-2)
            fit = serialize.SerializableNonlinearFit(fit)
            y_ctm = model_continuum(continuum.x, fit.p)
            result = fit.serialize()
            result['model_name'] = model_name
            result['model'] = model
            result['model_ctm'] = model_continuum
            result['continuum'] = continuum
            result['label'] = label
            result['fit'] = fit
            result['process'] = process
            result['channel'] = channel
            result['f(q2max)'] = y_ctm[0]
            result['f(q2=0)'] = y_ctm[-1]
            result['f'] = y_ctm
            results.append(result)

    return data, pd.DataFrame(results)