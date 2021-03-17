-- name: create_schema#
-- Creates the schema for the database
CREATE TABLE IF NOT EXISTS ensemble
(
    ens_id integer PRIMARY KEY,
    name text not NULL,
    ns integer not NULL,
    nt integer not NULL,
    description text not NULL,
    a_fm float not NULL,
    UNIQUE(name)
);

CREATE TABLE IF NOT EXISTS form_factor
(
    form_factor_id integer PRIMARY KEY,
    ens_id integer REFERENCES ensemble(ens_id),
    m_mother float not NULL,
    m_daughter float not NULL,
    m_spectator float not NULL,
    current text not NULL,
    sink text not NULL,
    source text not NULL,
    momentum text not NULL,
    UNIQUE(ens_id, momentum, m_mother, m_daughter, m_spectator, current, sink, source)
);

CREATE TABLE IF NOT EXISTS result_form_factor
(
    fit_id integer PRIMARY KEY,
    form_factor_id integer REFERENCES form_factor(form_factor_id),
    calcdate timestamp with time zone,
    -- analysis choices
    n_decay_src integer not NULL,
    n_oscillating_src integer not NULL,
    n_decay_snk integer not NULL,
    n_oscillating_snk integer not NULL,
    tmin_src integer not NULL,
    tmax_src integer not NULL,
    tmin_snk integer not NULL,
    tmax_snk integer not NULL,
    -- str representations of dicts containing the parameter's prior and posterior values
    params text not NULL,
    prior text not NULL,
    prior_alias text not NULL,  -- descriptive tag like 'standard prior'
    -- posterior values for key parameters in the fit
    r text not NULL,
    r_guess text not NULL,  -- well, sort of a prior
    energy_src text not NULL,
    energy_snk text not NULL,
    amp_src text not NULL,
    amp_snk text not NULL,
    -- statistical metrics
    aic float not NULL,
    chi2_aug float not NULL,
    chi2 float not NULL,
    chi2_per_dof float not NULL,
    model_probability float not NULL,
    p_value float not NULL,
    q_value float not NULL,
    dof integer not NULL,
    nparams integer not NULL,
    npoints integer not NULL,
    UNIQUE(
        form_factor_id, tmin_src, tmax_src, tmin_snk, tmax_snk,
        n_decay_src, n_oscillating_src, n_decay_snk, n_oscillating_snk)
);