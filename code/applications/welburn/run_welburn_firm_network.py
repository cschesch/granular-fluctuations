"""One fixed-cutoff Welburn application, using the shared grouped MGP estimator."""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from mgp_core import (
    ANALYSIS_BOOTSTRAP_DRAWS, ROOT, RUN_OUTPUT_CACHE, analysis_block_length,
    choose_solver, make_panel_rng, mgp_share, sample_cov,
)
from gradient_bootstrap import bootstrap_family
from applications.welburn.welburn_network import SUMMARY_NAME, build_welburn_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--solver', default='auto')
    parser.add_argument('--bootstrap-b', type=int, default=ANALYSIS_BOOTSTRAP_DRAWS)
    parser.add_argument('--seed', type=int, default=20260908)
    parser.add_argument('--refresh-inputs', action='store_true')
    args = parser.parse_args()
    if args.bootstrap_b < 0 or args.bootstrap_b == 1:
        parser.error('--bootstrap-b must be zero or at least two')
    inputs = build_welburn_inputs(refresh_inputs=args.refresh_inputs)
    solver = choose_solver(args.solver)
    point = mgp_share(inputs.grouped, solver)
    block = analysis_block_length(len(inputs.dates))
    family = bootstrap_family(
        inputs.grouped, b=args.bootstrap_b, block_length=block,
        rng=make_panel_rng(args.seed, 'welburn_output_network_gradient'), solver=solver,
    ) if args.bootstrap_b else {}
    sigma = sample_cov(inputs.outcomes)
    variance = float(point['aggregate_variance'])
    direct = float(np.sum(inputs.shares**2 * np.diag(sigma)) / variance)
    homoskedastic = float(np.mean(np.diag(sigma)) * np.sum(inputs.shares**2) / variance)
    record = dict(inputs.diagnostics, dataset='welburn_firm_network_M10',
        table_outcome='network-adjusted firm output innovations', frequency='quarterly',
        mgp_share=float(point['mgp_share']), mgp_level=float(point['mgp_level']),
        aggregate_variance=variance, solver=point['solver'], status=point['status'],
        dual_min_eig=float(point['dual_min_eig']), dual_min_diag=float(point['dual_min_diag']),
        bootstrap_b=args.bootstrap_b, block_bootstrap_b=args.bootstrap_b,
        inference='moving-block gradient bootstrap' if args.bootstrap_b else 'point estimate only',
        seed=args.seed, block_length=block,
        se_share=float(point['se_share']), normal_ucl95=float(point['normal_ucl95']),
        bootstrap_mean=family.get('bootstrap_mean', np.nan),
        bootstrap_p05=family.get('bootstrap_low', np.nan),
        bootstrap_p95=family.get('bootstrap_high', np.nan),
        moving_block_bootstrap_mean=family.get('block_mean', np.nan),
        moving_block_bootstrap_p05=family.get('block_low', np.nan),
        moving_block_bootstrap_p95=family.get('block_high', np.nan),
        gradient_bootstrap_p05=family.get('gradient_low', np.nan),
        gradient_bootstrap_p95=family.get('gradient_high', np.nan),
        moving_block_gradient_bootstrap_p05=family.get('block_gradient_low', np.nan),
        moving_block_gradient_bootstrap_p95=family.get('block_gradient_high', np.nan),
        gradient_bootstrap_common_rank=family.get('directional_common_rank', np.nan),
        gradient_bootstrap_regular=bool(family.get('directional_regular', False)),
        gradient_bootstrap_delta_T=family.get('directional_delta_T', np.nan),
        gradient_bootstrap_fallbacks=family.get('directional_fallbacks', 0),
        HHI=float(inputs.shares @ inputs.shares), top1_share=float(inputs.shares.max()),
        top10_share=float(np.sort(inputs.shares)[-10:].sum()),
        direct_diagonal=direct, naive=direct, naive_homoscedastic=homoskedastic,
        notes='Improved disclosed links; density-matching population-logit cutoff; original sales/COGS LP; requirements inversion; M=10.')
    RUN_OUTPUT_CACHE.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([record]).to_pickle(RUN_OUTPUT_CACHE / SUMMARY_NAME)
    (ROOT / 'output/welburn_firm_network_diagnostics.json').write_text(json.dumps(record, indent=2) + '\n')
    print(f'Welburn: N={record["N"]}, T={record["T"]}, M=10; granularity={100*record["mgp_share"]:.8f}%', flush=True)


if __name__ == '__main__':
    main()
