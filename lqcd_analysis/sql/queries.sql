-- name: write_ensemble!
INSERT OR IGNORE INTO ensemble(name, ns, nt, description, a_fm)
VALUES (:name, :ns, :nt, :description, :a_fm);

--name: fetch_ens_id$
SELECT ens_id from ensemble where name=:name;

--name: write_form_factor!
INSERT OR IGNORE INTO form_factor
(ens_id, m_mother, m_daughter, m_spectator, current, sink, source, momentum)
VALUES (:ens_id, :m_mother, :m_daughter, :m_spectator, :current, :sink, :source, :momentum);

--name: fetch_form_factor_id$
SELECT form_factor_id FROM form_factor
WHERE (ens_id, m_mother, m_daughter, m_spectator, current, sink, source, momentum)=
(:ens_id, :m_mother, :m_daughter, :m_spectator, :current, :sink, :source, :momentum);

-- name: write_result_form_factor!
INSERT INTO result_form_factor(
    form_factor_id, calcdate,
    n_decay_src, n_oscillating_src, n_decay_snk, n_oscillating_snk,
    tmin_src, tmax_src, tmin_snk, tmax_snk,
    params, prior, prior_alias, r, r_guess,
    energy_src, energy_snk, amp_src, amp_snk,
    aic, chi2_aug, chi2, chi2_per_dof, model_probability,
    p_value, q_value, dof, nparams, npoints, matrix_element
)
VALUES (
    :form_factor_id, :calcdate,
    :n_decay_src, :n_oscillating_src, :n_decay_snk, :n_oscillating_snk,
    :tmin_src, :tmax_src, :tmin_snk, :tmax_snk,
    :params, :prior, :prior_alias, :r, :r_guess,
    :energy_src, :energy_snk, :amp_src, :amp_snk,
    :aic, :chi2_aug, :chi2, :chi2_per_dof, :model_probability,
    :p_value, :q_value, :dof, :nparams, :npoints, :matrix_element)
ON CONFLICT (
    form_factor_id,
    tmin_src, tmax_src, tmin_snk, tmax_snk,
    n_decay_src, n_oscillating_src, n_decay_snk, n_oscillating_snk)
DO UPDATE SET (
    calcdate,
    params, prior, prior_alias, r, r_guess,
    energy_src, energy_snk, amp_src, amp_snk,
    aic, chi2_aug, chi2, chi2_per_dof, model_probability,
    p_value, q_value, dof, nparams, npoints, matrix_element)
    =(
    :calcdate,
    :params, :prior, :prior_alias, :r, :r_guess,
    :energy_src, :energy_snk, :amp_src, :amp_snk,
    :aic, :chi2_aug, :chi2, :chi2_per_dof, :model_probability,
    :p_value, :q_value, :dof, :nparams, :npoints, :matrix_element);
