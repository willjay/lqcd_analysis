"""
After conducting an analysis, one is faced with the task saving the results.
This module provides classes and functions for saving the results of fits in a standardized way to
databases.
"""
import os
from collections import namedtuple
import sqlite3
import numpy as np
import sqlalchemy
from sqlalchemy import Column, Sequence, ForeignKey, UniqueConstraint
from sqlalchemy import Integer, Float, DateTime, TEXT
from sqlalchemy.ext.declarative import declarative_base
from . import serialize_db

sqlite3.register_adapter(np.int64, int)  # Kludge for obscure Python3 issue with integers in dbs

Base = declarative_base()
MassTriplet = namedtuple("MassTriplet", ['mother', 'daughter', 'spectator'])
OperatorTriplet = namedtuple("OperatorTriplet", ['current', 'source', 'sink'])

class Ensemble(Base):
    """
    Database table "ensemble"
    """
    __tablename__ = "ensemble"
    ens_id = Column(Integer, Sequence('ens_id_seq'), primary_key=True)
    name = Column(TEXT, nullable=False)
    ns = Column(Integer, nullable=False)
    nt = Column(Integer, nullable=False)
    description = Column(TEXT, nullable=False)
    a_fm = Column(Float, nullable=False)

    # An ensemble is taken to be uniquely defined by its name.
    # Other columns are simply present for additional convenience at the analysis level.
    __table_args__ = (
        UniqueConstraint('name'),
    )

    def __init__(self, name, ns, nt, description, a_fm):
        self.name = name
        self.ns = ns
        self.nt = nt
        self.description = description
        self.a_fm = a_fm

    def items(self):
        """
        Returns a set-like object providing a view on the dictionary items obtained from __dict__
        """
        adict = {key: val for key, val in self.__dict__.items() if not key.startswith("_")}
        return adict.items()

    def __iter__(self):
        return self.__dict__.__iter__()

    def get_id(self, connection):
        """
        Gets the ens_id associated with the ensemble.
        Also sets the (local) value in the process.
        Args:
            connection: database connection
        Returns:
            ens_id: int
        """
        query = sqlalchemy.text("SELECT ens_id FROM ensemble WHERE name=:name LIMIT 1;")
        ens_id, = connection.execute(query, {"name": self.name}).fetchone()
        self.ens_id = ens_id
        return ens_id

    def write(self, connection):
        """
        Writes the object, ignoring upon conflict
        Args:
            connection: database connection
        """
        stmt = sqlalchemy.insert(Ensemble).values(self).prefix_with("OR IGNORE")
        connection.execute(stmt)


class FormFactor(Base):
    """
    Database table "form_factor" for registering a form factor.
    """
    __tablename__ = "form_factor"
    form_factor_id = Column(Integer, Sequence('form_factor_id_seq'), primary_key=True)
    ens_id = Column(Integer, ForeignKey('ensemble.ens_id'), nullable=False)
    m_mother = Column(Float, nullable=False)
    m_daughter = Column(Float, nullable=False)
    m_spectator = Column(Float, nullable=False)
    current = Column(TEXT, nullable=False)
    sink = Column(TEXT, nullable=False)
    source = Column(TEXT, nullable=False)
    momentum = Column(TEXT, nullable=False)

    __table_args__ = (
        UniqueConstraint('ens_id', 'momentum', 'm_mother', 'm_daughter', 'm_spectator',
                         'current', 'sink', 'source'),
    )

    def __init__(self, masses, operators, momentum):
        self.ens_id = None
        self.m_mother = masses.mother
        self.m_daughter = masses.daughter
        self.m_spectator = masses.spectator
        self.current = operators.current
        self.source = operators.source
        self.sink = operators.sink
        self.momentum = momentum

    def set_ens_id(self, ens_id):
        """ Sets the value of self.ens_id """
        self.ens_id = ens_id

    def items(self):
        """
        Returns a set-like object providing a view on the dictionary items obtained from __dict__
        """
        adict = {key: val for key, val in self.__dict__.items() if not key.startswith("_")}
        return adict.items()

    def __iter__(self):
        return self.__dict__.__iter__()

    def asdict(self):
        """ Returns a dictionary representation. """
        return dict(self.items())

    def get_id(self, connection):
        """
        Gets the form_factor_id associated with the form factor.
        Also sets the (local) value in the process.
        Args:
            connection: database connection
        Return
        """
        query = sqlalchemy.text(
            "SELECT form_factor_id FROM form_factor WHERE "
            "(ens_id, m_mother, m_daughter, m_spectator, current, source, sink)="
            "(:ens_id, :m_mother, :m_daughter, :m_spectator, :current, :source, :sink) LIMIT 1;")
        form_factor_id, = connection.execute(query, self.asdict()).fetchone()
        self.form_factor_id = form_factor_id
        return form_factor_id

    def write(self, connection):
        """
        Writes the form factor to a database
        Args:
            connection: database connection
        """
        stmt = sqlalchemy.insert(FormFactor).values(self).prefix_with("OR IGNORE")
        connection.execute(stmt)


class ResultFormFactor(Base):
    """
    Database table "result_form_factor" for storing results for joint fits to 2pt and 3pt functions,
    which yield matrix elements / form factors.
    """
    __tablename__ = "result_form_factor"
    fit_id = Column(Integer, Sequence('fit_id_seq'), primary_key=True)
    form_factor_id = Column(Integer, ForeignKey('form_factor.form_factor_id'), nullable=False)
    calcdate = Column(DateTime, nullable=False)

    # analysis choices
    n_decay_src = Column(Integer, nullable=False)
    n_oscillating_src = Column(Integer, nullable=False)
    n_decay_snk = Column(Integer, nullable=False)
    n_oscillating_snk = Column(Integer, nullable=False)
    tmin_src = Column(Integer, nullable=False)
    tmax_src = Column(Integer, nullable=False)
    tmin_snk = Column(Integer, nullable=False)
    tmax_snk = Column(Integer, nullable=False)

    # str representations of dicts containing the parameter's prior and posterior values
    params = Column(TEXT, nullable=False)
    prior = Column(TEXT, nullable=False)
    prior_alias = Column(TEXT, nullable=False)  # descriptive tag like 'standard prior'

    # posterior values for key parameters in the fit
    r = Column(TEXT, nullable=False)
    r_guess = Column(TEXT, nullable=False)  # well, sort of a prior
    energy_src = Column(TEXT, nullable=False)
    energy_snk = Column(TEXT, nullable=False)
    amp_src = Column(TEXT, nullable=False)
    amp_snk = Column(TEXT, nullable=False)

    # statistical metrics
    aic = Column(Float, nullable=False)
    chi2_aug = Column(Float, nullable=False)
    chi2 = Column(Float, nullable=False)
    chi2_per_dof = Column(Float, nullable=False)
    model_probability = Column(Float, nullable=False)
    p_value = Column(Float, nullable=False)
    q_value = Column(Float, nullable=False)
    dof = Column(Integer, nullable=False)
    nparams = Column(Integer, nullable=False)
    npoints = Column(Integer, nullable=False)

    __table_args__ = (
        UniqueConstraint('form_factor_id', 'tmin_src', 'tmax_src', 'tmin_snk', 'tmax_snk',
                         'n_decay_src', 'n_oscillating_src', 'n_decay_snk', 'n_oscillating_snk'),
    )

    def __init__(self, form_factor, form_factor_fitter):
        self.form_factor_id = form_factor.form_factor_id
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

    def asdict(self):
        """ Returns a dictionary representation. """
        return dict(self.items())

    def items(self):
        """
        Returns a set-like object providing a view on the dictionary items obtained from __dict__
        """
        adict = {key: val for key, val in self.__dict__.items() if not key.startswith("_")}
        return adict.items()

    def __iter__(self):
        return self.__dict__.__iter__()

    def upsert(self, connection):
        """
        Upserts the result into the database, i.e., inserting the result if new or updating an old
        result if it already exists
        """
        stmt = serialize_db.build_upsert_query(connection, self.__tablename__)
        connection.execute(stmt, self.asdict())


def connect(database, echo=False):
    """
    Gets a connection engine for a SQLite database. If the database does not already exist, one is
    created with the tables defined in this module.
    Args:
        database: str, full path to the database
        echo: bool, whether or not to echo the queries issued to the database
    Returns:
        engine, the database connection engine
    """
    if not database.endswith('.sqlite'):
        raise ValueError("Only SQLite databases are supports")
    engine = sqlalchemy.create_engine('sqlite:///' + database, echo=echo)
    if not os.path.isfile(database):
        Base.metadata.create_all(engine)
    return engine
