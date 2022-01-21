import re
from collections import namedtuple
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
from . import staggered

from allhisq_analysis import data_tables


FitKey = namedtuple('FitKey', ['a_fm', 'description', 'm_light', 'm_strange', 'm_heavy'])


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
                'alias_heavy': "('0.9 m_charm', '1.0 m_charm', '1.1 m_charm', '1.4 m_charm', '1.5 m_charm', '2.0 m_charm')",
                'alias_light': "('1.0 m_light', '0.1 m_strange', '0.2 m_strange')",
                'alias_spectator': "('1.0 m_strange')",
            }
        if process in ('D to K', 'D2K'):
            return {
                'alias_heavy': "('0.9 m_charm', '1.0 m_charm', '1.1 m_charm', '1.4 m_charm', '1.5 m_charm', '2.0 m_charm')",
                'alias_light': "('1.0 m_strange')",
                'alias_spectator': "('1.0 m_light', '0.1 m_strange', '0.2 m_strange')",
            }
        if process in ('D to pi', 'D2pi'):
            return {
                'alias_heavy': "('0.9 m_charm', '1.0 m_charm', '1.1 m_charm', '1.4 m_charm', '1.5 m_charm', '2.0 m_charm')",
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
                # 'm_light': subdf['m_light'].unique().item(),
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
    # for ens_id in [25, 15, 28, 13, 12]:
    for ens_id in [25, 15, 28, 13, 12, 36, 35]:
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
    hadron_masses = pd.read_sql("SELECT * FROM hadron_masses;", engine)
    for key in ['pion', 'kaon', 'd', 'ds']:
        hadron_masses[key] = hadron_masses[key].apply(gv.gvar)
    data = pd.merge(data, hadron_masses, on=['ens_id', 'alias_heavy', 'm_heavy'])
    if process in ['Ds to K', 'Ds2K']:
        data['M_daughter'] = data['kaon']
        data['M_mother'] = data['ds']
    elif process in ['D to K', 'D2K']:
        data['M_daughter'] = data['kaon']
        data['M_mother'] = data['d']
    elif process in ['D to pi', 'D2pi']:
        data['M_daughter'] = data['pion']
        data['M_mother'] = data['d']
    else:
        raise ValueError("Unrecognized process", process)

    # Read sea masses
    sea_masses = read_sea_masses(engine)
    data = pd.merge(data, sea_masses[['ens_id', 'm_light', 'm_strange']], on='ens_id')
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
            df = read_all('S-S', self.process, self.engine)
            # Renormalization is not required, since the scalar density is absolutely normalized.
            # Changing units is not required, since the scalar form factor is dimensionless.
            # Apply normalization to remove leading-order discretization effect from HQET
            df['form_factor'] = df['form_factor'] * df['m_heavy'].apply(staggered.chfac)
            self._scalar = df
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
            # Apply normalization to remove leading-order discretization effect from HQET
            df['form_factor'] = df['form_factor'] * df['m_heavy'].apply(staggered.chfac)
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
            # Apply normalization to remove leading-order discretization effect from HQET
            df['form_factor'] = df['form_factor'] * df['m_heavy'].apply(staggered.chfac)            
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
    def __init__(self, a_fm, description, m_light, m_strange, m_heavy, dm_heavy, M_daughter, p2):
        """
        Args:
            a_fm: float, approximate lattice spacing in fm (e.g., 0.15). Used to look up exact scale
            description: str, the ratio ml/ms (e.g., '1/27'). Used to look up the exact scale
            m_light: float, bare mass of the light (u/d) quarks
            m_strange: float, bare mass of the strange quark
            m_heavy: float, bare mass of the heavy quark
            dm_heavy: float, "mistuning" of the bare heavy quark mass in the problem (m-m_physical)
            M_daughter: float or gvar, the mass of the daughter hadron
            p2: np.array, the squared lattice momenta of the daughter hadron
        """
        scale = data_tables.ScaleSetting().data
        mask = (scale['a[fm]'] == a_fm) & (scale['description'] == description)
        w0 = gv.mean(scale[mask]['w0_orig/a'].item())
        # Convert to dimensionless units of w0
        self.m_light = m_light * w0
        self.m_strange = m_strange * w0
        self.m_heavy = m_heavy * w0
        self.dm_heavy = dm_heavy * w0
        self.E = np.sqrt(M_daughter**2 + p2) * w0

    def asdict(self):
        return dict(self.__dict__)

    def __getitem__(self, key):
        return self.__dict__.get(key)

    def get(self, key, default=None):
        """ Return the value for key if key is in the dictionary, else default. """
        return self.__dict__.get(key, default)


def build_fit_data(dataframe):
    """
    Builds dictionaries suitable for interpretation as input data
    for chiral-continuum fits with lsqfit, "data=(x,y)".
    Args:
        dataframe: pd.DataFrame containing correlated data
    Returns:
        xdict, ydict: the data dictionaries for the fit
    """
    keys = ['a_fm', 'description', 'm_light', 'm_strange', 'm_heavy', 'dm_heavy']
    groups = dataframe.groupby(keys)
    xdict, ydict = {}, {}
    for (a_fm, description, m_light, m_strange, m_heavy, dm_heavy), subdf in groups:
        subdf = subdf.sort_values(by='E_daughter')
        y = subdf['form_factor'].values
        M_daughter = subdf['M_daughter'].apply(gv.mean).unique().item()
        p2 = subdf['p2'].values
        x = InputData(a_fm, description, m_light, m_strange, m_heavy, dm_heavy, M_daughter, p2).asdict()

        # Include continuum constants like fpi as independent "x-parameters"
        scale = data_tables.ScaleSetting()
        ctm = data_tables.ContinuumConstants()
        fpi = ctm.pdg['fpi'] * scale.w0_fm / ctm.hbarc
        x['fpi'] = gv.mean(fpi)

        # Include staggered low-energy constants as independent "x-parameters"
        const = data_tables.StaggeredConstants().get_row(a_fm=a_fm)
        w0 = gv.mean(scale.get_row(a_fm=a_fm, description=description)['w0_orig/a'])

        # Quantities with mass dimension zero
        x['alpha_s'] = const['alpha_s']

        # Quantities with mass dimension +1
        x['mu'] = const['mu'] * w0

        # Quantities with mass dimension +2
        for k in ['Delta_P', 'Delta_A', 'Delta_T', 'Delta_V', 'Delta_I',
                    'DeltaBar', 'Hairpin_V', 'Hairpin_A']:
            x[k] = gv.mean(const[k]) * w0**2

        # Hadron masses
        x['M_daughter'] = subdf['M_daughter'].apply(gv.mean).unique().item() * w0
        x['M_mother'] = subdf['M_mother'].apply(gv.mean).unique().item() * w0
        x['mpi5'] = subdf['pion'].apply(gv.mean).unique().item() * w0
        x['mK5'] = subdf['kaon'].apply(gv.mean).unique().item() * w0
        x['mS5'] = np.sqrt(const['mu'] * (2 * m_strange) * w0)

        # Collect results
        key = FitKey(a_fm, description, m_light, m_strange, m_heavy)
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

        # Hadron masses
        mpi5 = ctm.pdg['pi'] * scale.w0_fm / ctm.hbarc
        mK5 = ctm.pdg['K'] * scale.w0_fm / ctm.hbarc
        mS5 = np.nan
        mother = self.mother * scale.w0_fm / ctm.hbarc
        daughter = self.daughter * scale.w0_fm / ctm.hbarc

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
        # dm_heavy represents the "mistuing" dm = (m-m0) of the heavy quark.
        # By definition, this difference vanishes at the physical point.
        return {
            'fpi': gv.mean(f),
            'm_light': gv.mean(ml_ctm),
            'm_strange': gv.mean(ms_ctm),
            'm_heavy': gv.mean(mc_ctm),
            'dm_heavy': 0,
            'alpha_s': 0,
            'E': np.linspace(gv.mean(energy_min), gv.mean(energy_max)),
            'mpi5': gv.mean(mpi5),
            'mK5': gv.mean(mK5),
            'mS5': gv.mean(mS5),
            # 'mpi5': gv.mean(self.daughter * scale.w0_fm / ctm.hbarc),
            # 'mK5': mu_ctm * (ml_ctm + ms_ctm),
            # 'mS5': mu_ctm * (2.0 * ms_ctm),
            'mu': gv.mean(mu_ctm),
            'DeltaBar': 0,
            'M_mother': gv.mean(mother),
            'M_daughter': gv.mean(daughter),
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
        }
        if self.model_name != 'LogLess':
            prior['g'] = gv.gvar(10, 10)
        return prior

    def build_priors(self):
        def _load_prior(adict, keys, width=1.0):
            for key in keys:
                if key.startswith('log'):
                    adict[key] = np.log(gv.gvar(0.1, width))
                else:
                    adict[key] = gv.gvar(0.0, width)
            return adict

        priors = {}
        nlo_terms = ['c_l', 'c_H', 'c_E', 'c_E2']
        nnlo_terms = ['c_l2', 'c_lH', 'c_lE', 'c_H2', 'c_HE']
        minimal_terms = ['c_H', 'c_E', 'c_E2', 'c_lE', 'c_EH']
        if 'SU3' in self.model_name:
            nlo_terms.append('c_s')
            nnlo_terms.extend(['c_sl', 'c_sH', 'c_sE', 'c_s2'])

        # NLO fit
        # priors['NLO, a2'] = _load_prior(self.build_base_prior(), ['c_a2'] + nlo_terms)

        # Minimal fit
        # priors['minimal'] = _load_prior(self.build_base_prior(), ['c_a2'] + minimal_terms)
        # priors['minimal, wide'] = _load_prior(self.build_base_prior(), ['c_a2'] + minimal_terms, 100)

        # Full NNLO
        priors['NNLO'] = _load_prior(self.build_base_prior(), ['c_a2'] + nlo_terms + nnlo_terms)
        priors['NNLO, priors 10x'] = _load_prior(self.build_base_prior(), ['c_a2'] + nlo_terms + nnlo_terms, 10)
        # priors['NNLO, priors 100x'] = _load_prior(self.build_base_prior(), ['c_a2'] + nlo_terms + nnlo_terms, 100)

        # Alternative treatments of discretization errors
        # - a = generic discetization effects
        # - h = HQET discretization effects
        priors['NNLO, a'] =  _load_prior(self.build_base_prior(), ['c_a'] + nlo_terms + nnlo_terms)
        priors['NNLO, a4'] = _load_prior(self.build_base_prior(), ['c_a4'] + nlo_terms + nnlo_terms)
        priors['NNLO, h2'] = _load_prior(self.build_base_prior(), ['c_h2'] + nlo_terms + nnlo_terms)
        priors['NNLO, h4'] = _load_prior(self.build_base_prior(), ['c_h4'] + nlo_terms + nnlo_terms)
        priors['NNLO, a2+a4'] = _load_prior(self.build_base_prior(), ['c_a2', 'c_a4'] + nlo_terms + nnlo_terms)
        # priors['NNLO, a2+h2'] = _load_prior(self.build_base_prior(), ['c_a2', 'c_h2'] + nlo_terms + nnlo_terms)
        priors['NNLO, a2+h4'] = _load_prior(self.build_base_prior(), ['c_a2', 'c_h4'] + nlo_terms + nnlo_terms)
        priors['NNLO, h2+a4'] = _load_prior(self.build_base_prior(), ['c_h2', 'c_a4'] + nlo_terms + nnlo_terms)
        # priors['NNLO, h2+h4'] = _load_prior(self.build_base_prior(), ['c_h2', 'c_h4'] + nlo_terms + nnlo_terms)
        # priors['NNLO, a2+a4+h2+h4'] = _load_prior(self.build_base_prior(), ['c_a2', 'c_a4', 'c_h2', 'c_h4'] + nlo_terms + nnlo_terms)

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

    if process == 'Ds to K':
        models = {
            # 'SU2': su2.SU2Model,
            'HardSU2': su2.HardSU2Model,
            # 'SU2:continuum': su2.SU2Model,
            'HardSU2:continuum': su2.HardSU2Model,
            'LogLess': chipt.LogLessModel,
        }
    else:
        models = {
            # 'SU2': su2.SU2Model,
            'HardSU2': su2.HardSU2Model,
            # 'SU2:continuum': su2.SU2Model,
            'HardSU2:continuum': su2.HardSU2Model,
            'LogLess': chipt.LogLessModel,
        }

    results = []
    for model_name, model_fcn in models.items():
        print("Starting fits for", model_name)
        
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
        continuum = ContinuumLimit(model.process)
        
        # Masks for dropping parts of the dataset
        masks = {
            'full': data['a_fm'] > 0,  # trivially true by definition. The full dataset.
            'omit 0.12 fm': data['a_fm'] != 0.12,  # drop the coarsest lattice spacing
            'omit 0.042 fm': data['a_fm'] != 0.42,  # drop the finest lattice spacing
            'mh/mc <= 1.1': ~data['alias_heavy'].isin(['1.4 m_charm', '1.5 m_charm', '2.0 m_charm', '2.2 m_charm']),
        }
        for mask_label, mask in masks.items():
            # Build data
            x, y_data = build_fit_data(data[mask])

            # Run variations on the model
            priors = ModelVariations(model.process, model_name).priors
            for label, prior in tqdm(priors.items()):
                if (label != 'NNLO') & (mask_label not in ('full', 'mh/mc <= 1.1')):
                    # Keep: full data and NNLO
                    # Keep: full data and model variation
                    # Keep: drop data and NNLO
                    # Skip: drop data and model variation simultaneously
                    continue

                fit = lsqfit.nonlinear_fit(data=(x, y_data), fcn=wrapped, prior=prior, debug=True)
                fit = serialize.SerializableNonlinearFit(fit)
                y_ctm = model_continuum(continuum.x, fit.p)
                result = fit.serialize()
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
                results.append(result)


        


    return data, pd.DataFrame(results)

