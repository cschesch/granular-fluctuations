"""Moving-block directional-bootstrap UCLs for the matched FSW comparison.

Keep weights and propagation fixed. Account jointly for grouped covariance and
the estimated two-factor variance denominator. The latter is smooth locally;
its derivative is evaluated by a central covariance perturbation.
"""
import numpy as np
import pandas as pd

from mgp_core import (RUN_OUTPUT_CACHE, analysis_block_length, build_lpt_grouped_data,
                      choose_solver, moving_block_indices, sample_cov,
                      solve_grouped_mgp_dual)
from near_optimal import delta_rate, near_optimal_derivative
from applications.foerster.run_foerster_matched_comparison import WINDOWS, export_tables, load_inputs, two_factor_decomposition


def fitted_variance(cov, v):
    sd = np.sqrt(np.diag(cov))
    eigenvalues, eigenvectors = np.linalg.eigh(cov / np.outer(sd, sd))
    load = sd[:, None] * eigenvectors[:, -2:]
    common = (load * eigenvalues[-2:]) @ load.T
    fitted = common + np.diag(np.diag(cov - common))
    return float(np.sum(v * fitted))


def main():
    eps, v = load_inputs()
    solver = choose_solver('auto')
    influence = np.sqrt(np.diag(v))
    scale = influence.sum() ** 2
    rows, draws = [], []
    for j, (start, end, _) in enumerate(WINDOWS):
        x = eps.loc[start:end].to_numpy(float)
        t = len(x)
        grouped, _, _, _ = build_lpt_grouped_data(x.T, influence / influence.sum(), 10)
        cg, ce = sample_cov(grouped), sample_cov(x)
        denominator = fitted_variance(ce, v)
        np.testing.assert_allclose(denominator, two_factor_decomposition(x, v)['model_variance'], rtol=1e-10)
        mgp = solve_grouped_mgp_dual(cg, solver)
        estimate = scale * float(mgp['value']) / denominator
        rng = np.random.default_rng(20261020 + j)
        block = analysis_block_length(t)
        directions, dd, dd_check = [], [], []
        for _ in range(1000):
            idx = moving_block_indices(t, block, rng)
            directions.append(np.sqrt(t) * (sample_cov(grouped[idx]) - cg))
            h = np.sqrt(t) * (sample_cov(x[idx]) - ce)
            for step, target in [(1e-5, dd), (5e-6, dd_check)]:
                target.append((fitted_variance(ce + step*h, v) - fitted_variance(ce - step*h, v)) / (2*step))
        np.testing.assert_allclose(dd, dd_check, rtol=2e-5, atol=1e-5)
        level, ok = near_optimal_derivative(np.array(directions), cg[None], np.array([float(mgp['value']) * (1+delta_rate(t))]))
        if not ok.all():
            raise RuntimeError('Directional SDP failed')
        derivative = (scale * level - estimate * np.array(dd)) / denominator
        ucl = min(1., estimate - np.quantile(derivative, .05) / np.sqrt(t))
        row = dict(period=f'{start}--{end}', T=t, block_length=block, bootstrap_b=1000,
                   matched_mgp_ceiling=estimate, moving_block_gradient_ucl95=ucl,
                   moving_block_gradient_lcl05=max(0., estimate-np.quantile(derivative,.95)/np.sqrt(t)),
                   denominator=denominator, delta_T=delta_rate(t), seed=20261020+j)
        rows.append(row)
        draws.append(pd.DataFrame(dict(period=row['period'], draw=np.arange(1000), derivative=derivative)))
        print(row, flush=True)
    result = pd.DataFrame(rows)
    result.to_pickle(RUN_OUTPUT_CACHE / 'foerster_matched_inference.pkl')
    result.to_csv(RUN_OUTPUT_CACHE / 'foerster_matched_inference.csv', index=False)
    pd.concat(draws).to_pickle(RUN_OUTPUT_CACHE / 'foerster_matched_inference_draws.pkl')
    comparison = pd.read_pickle(RUN_OUTPUT_CACHE / 'foerster_matched_comparison.pkl')
    table = comparison.merge(result, on=['period', 'T'], validate='one_to_one', suffixes=('', '_bootstrap'))
    assert len(table) == len(WINDOWS)
    np.testing.assert_allclose(table.matched_mgp_ceiling, table.matched_mgp_ceiling_bootstrap)
    export_tables(table.to_dict('records'))


if __name__ == '__main__':
    main()
