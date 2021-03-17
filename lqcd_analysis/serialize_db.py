"""
After conducting an analysis, one is faced with the task saving the results.
This module provides classes and functions for saving the results of fits in a standardized way to
databases.
"""
import os
from contextlib import contextmanager
from collections import namedtuple
import sqlite3
import pathlib
import numpy as np
import aiosql
import sqlalchemy

sqlite3.register_adapter(np.int64, int)  # Kludge for obscure Python3 issue with integers in dbs


@contextmanager
def connection_scope(engine):
    """
    Context manager for working with raw low-level DBAPI connections.
    Args:
        engine: connection engine
    """
    connection = engine.raw_connection()
    try:
        yield connection
    except:
        print("Issue encountered in connection_scope")
        raise
    finally:
        connection.commit()
        connection.close()


def abspath(dirname):
    """
    Builds the absolute path of relative directories with respect to this file's location, i.e.,
    ./dir/  --> full/path/to/this/directory/dir/
    Args:
        dirname: str, directory name
    Returns:
        str: a full path
    """
    return os.path.join(pathlib.Path(__file__).parent.absolute(), dirname)


def connect(database, **kwargs):
    """
    Creates a connection engine for the specified database and creates the necessary schema.
    Args:
        database: str, the full path to the database
        kwargs: arguments passed to create_engine
    Returns:
        engine
    """
    if not database.endswith('.sqlite'):
        raise ValueError("Only SQLite databases are supports")
    engine = sqlalchemy.create_engine('sqlite:///' + database, **kwargs)
    queries = aiosql.from_path(abspath("sql"), "sqlite3")
    with connection_scope(engine) as conn:
        queries.create_schema(conn)
    return engine


def build_upsert_query(engine, table_name):
    """
    Builds a generic parameterized "upsert query" for updating non-unique columns.
    Args:
        engine: database connection
        table_name: str, the name of the database table
    Returns:
        sqlalchemy.sql.elements.TextClause, the query
    """
    def get_unique(table):
        """
        Gets the names of the columns which appear as a part of the table's
        unique constraint.
        """
        for constraint in table.constraints:
            if isinstance(constraint, sqlalchemy.UniqueConstraint):
                return [col.name for col in constraint.columns]
        raise ValueError("No unique constaint located.")

    def get_complement(columns, uniques):
        """
        Gets the names of the columns which are not a part of the table's unique constraint.
        """
        complement = []
        for col in columns:
            if col not in uniques:
                complement.append(col)
        return complement

    def join_columns(cols):
        """
        Converts a list into a comma-separated string: ['a', 'b', 'c'] -> 'a, b, c'.
        """
        return ", ".join(cols)

    def join_params(cols):
        """
        Converts a list into a comma-separated string with each element preceded by a semicolon:
        ['a', 'b', 'c'] -> ':a, :b, :c'.
        """
        return ", ".join(f":{col}" for col in cols)

    # Reflect table structure from database
    metadata = sqlalchemy.MetaData(bind=engine)
    table = sqlalchemy.Table(table_name, metadata, autoload=True, autoload_with=engine)
    # Extract primary key column
    pk_column, = table.primary_key.columns.keys()
    # Extract unique columns
    uniques = get_unique(table)
    # Remaining columns are the "complement"
    columns = [col.name for col in table.columns if col.name != pk_column]
    complement = get_complement(columns, uniques)
    # Finally build the query itself
    payload = {
        'columns': join_columns(columns),
        'values': join_params(columns),
        'uniques': join_columns(uniques),
        'set_columns': join_columns(complement),
        'set_values': join_params(complement),
    }
    query_template = (
        "INSERT INTO result_form_factor ({columns}) "
        "VALUES ({values}) ON CONFLICT ({uniques}) "
        "DO UPDATE SET ({set_columns})=({set_values})")
    return sqlalchemy.text(query_template.format(**payload))


class DictLike:
    """ Simple base class for dict-like objects. """
    def keys(self):
        """ Returns a set-like object providing a view on the keys. """
        return self.__dict__.keys()

    def values(self):
        """ Returns a set-like object providing a view on the values. """
        return self.__dict__.values

    def items(self):
        """ Returns a set-like object providing a view on the dictionary items. """
        adict = {key: val for key, val in self.__dict__.items() if not key.startswith("_")}
        return adict.items()

    def __iter__(self):
        return self.__dict__.__iter__()

    def asdict(self):
        """ Returns a dictionary representation. """
        return dict(self.items())

    def get(self, key, default=None):
        """ Return the value for key if key is in the dictionary, else default. """
        return self.__dict__.get(key, default)


class Ensemble(DictLike):
    """
    Wrapper class for managing a "physical" description of an ensemble and writing to a database.
    """
    def __init__(self, name, ns, nt, description, a_fm):
        self.name = name
        self.ns = ns
        self.nt = nt
        self.description = description
        self.a_fm = a_fm

    def fetch_id(self, engine):
        """
        Fetches the associated form_factor_id from the database.
        Args:
            engine: connection engine
        Returns:
            int, form_factor_id
        """
        queries = aiosql.from_path(abspath("sql/"), "sqlite3")
        with connection_scope(engine) as conn:
            queries.write_ensemble(conn, **self.asdict())
            ens_id = queries.fetch_ens_id(conn, **self.asdict())
            if ens_id is None:
                raise ValueError("bad ens_id")
        return ens_id

class FormFactor(DictLike):
    """
    Wrapper class for managing a "physical" description of a form factor and writing to a database.
    """
    def __init__(self, current, momentum, mother, daughter, spectator, source=None, sink=None):
        self.ens_id = None
        self.current = current
        self.momentum = momentum
        self.mother = mother
        self.daughter = daughter
        self.spectator = spectator
        self.source = source
        self.sink = sink

        self.mother_hadron = None
        self.daughter_hadron = None
        self.transition_name = None

        self.m_mother = None
        self.m_daughter = None
        self.m_spectator = None

    def identify(self, table_meson_names):
        """
        Identify the mother and daughter hadrons and the transition name using the table, setting
        attributes 'mother_hadron', 'daughter_hadron', and 'transition_name'
        """
        self.mother_hadron = table_meson_names.identify(self.mother, self.spectator)
        self.daughter_hadron = table_meson_names.identify(self.daughter, self.spectator)
        self.transition_name = f"{self.mother_hadron} to {self.daughter_hadron}"

    def determine_bare_masses(self, get_mq, ensemble):
        """
        Determine the bare masses, setting attributes 'm_mother', 'm_daughter', and 'm_spectator'
        Args:
            get_mq: function accepting keyword argument 'quark_alias'
        """
        kwargs = {'a_fm': ensemble.a_fm, 'description': ensemble.description}
        self.m_mother = get_mq(quark_alias=self.mother, **kwargs)
        self.m_daughter = get_mq(quark_alias=self.daughter, **kwargs)
        self.m_spectator = get_mq(quark_alias=self.spectator, **kwargs)

    def identify_daughter_quarks(self):
        """
        Identifies which quarks to take as the "heavy" and "light" quarks inside a daughter hadron.

        Note:
        Daughter hadrons include pions, kaons, D-, and Ds-mesons. Depending on the transition,
        either the "daughter quark" or the "spectator" quark can be the lighter / heavier of the
        two. Correctly estimating the mass of the daughter hadron requires correctly handling
        either possibility.
        """
        if self.transition_name in ['D to pi', 'B to pi', 'Ds to K', 'Bs to Ds', 'Bs to K']:
            # For pions, masses are degenerate, so light/heavy is arbitrary
            # For K and Ds, the spectator is strange, and the daughter quark is a
            # light quark.
            alias_light = self.daughter
            alias_heavy = self.spectator
        elif self.transition_name in ['B to K', 'D to K', 'B to D']:
            # Spectator is a light quark. Daughter quark is a strange quark
            alias_light = self.spectator
            alias_heavy = self.daughter
        else:
            raise ValueError(f"Unrecognized transition {self.transition_name}")
        return namedtuple('Aliases', ['light', 'heavy'])(alias_light, alias_heavy)

    def fetch_id(self, engine):
        """
        Fetches the associated form_factor_id from the database.
        Args:
            engine: connection engine
        Returns:
            int, form_factor_id
        """
        queries = aiosql.from_path(abspath("sql/"), "sqlite3")
        with connection_scope(engine) as conn:
            queries.write_form_factor(conn, **self.asdict())
            form_factor_id = queries.fetch_form_factor_id(conn, **self.asdict())
            if form_factor_id is None:
                raise ValueError("bad form_factor_id")
        return form_factor_id


class ResultFormFactor(DictLike):
    """
    Wrapper class for managing results of fits to extract form factors, mapping them into a
    standardized form, and writing the results to a database.
    Args:
        form_factor_fitter: analysis.FormFactorAnalysis
    """
    def __init__(self, form_factor_fitter):
        # map keys onto standardized names
        attr_map = {
            'tmin_ll': 'tmin_src',
            'tmin_hl': 'tmin_snk',
            'tmax_ll': 'tmax_src',
            'tmax_hl': 'tmax_snk',
            'n_decay_ll': 'n_decay_src',
            'n_decay_hl': 'n_decay_snk',
            'n_oscillating_ll': 'n_oscillating_src',
            'n_oscillating_hl': 'n_oscillating_snk',
            'q': None,
            'pedestal': None,
            'is_sane': None,
        }
        sanitize = {'r': str,}
        identity = lambda x: x
        # Grab attributes from serialized version of the fitter
        for key, value in form_factor_fitter.serialize().items():
            name = attr_map.get(key, key)
            # Skip keys mapped to None
            if name:
                # Use the mapped name if a standardized version has been specified.
                # Otherwise, simply use the existing name.
                setattr(self, name, sanitize.get(name, identity)(value))

    def write(self, database, ensemble, form_factor):
        """
        Writes the result to a database.
        Args:
            database: str, full path to a database
            ensemble: Ensemble
            form_factor: FormFactor
        """
        # Ensure that the necessary keys are prsent
        for key in ['name', 'ns', 'nt', 'description', 'a_fm']:
            if key not in ensemble:
                raise ValueError(f"ensemble missing key '{key}'")
        for key in ['current', 'momentum', 'm_mother', 'm_daughter', 'm_spectator']:
            if key not in form_factor:
                raise ValueError(f"form_factor missing key '{key}'")

        if form_factor.source is None:
            form_factor.source = 'default'
        if form_factor.sink is None:
            form_factor.sink = 'default'

        engine = connect(database)
        queries = aiosql.from_path(abspath("sql/"), "sqlite3")

        form_factor.ens_id = ensemble.fetch_id(engine)
        result = self.asdict()
        result['form_factor_id'] = form_factor.fetch_id(engine)
        with connection_scope(engine) as conn:
            queries.write_result_form_factor(conn, **result)
