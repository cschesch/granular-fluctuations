"""Collect the core-application estimates and naive benchmarks into one CSV.

Reads the cached run summaries and primitive-shock exports; runs no estimator,
filter, or resampling. Writes applications_coefficients_data.csv in the result cache, which
build_applications_vertical.py (main table) and plot_expanded_applications.py
(Figure 1) both consume.
"""
import numpy as np
import pandas as pd
from applications.foerster.run_foerster_epsilon import DEFAULT_EPSILON_CSV
from applications.foerster.run_foerster_matched_comparison import load_inputs as load_foerster_inputs

from mgp_core import RUN_OUTPUT_CACHE as CACHE, rule_summary_stem
from project_paths import DERIVED_CACHE, ROOT

ROWS = [
    ('Sales · quarterly', 'compustat_qtr', 'compustat_quarterly_sales'),
    ('Sales · annual', 'compustat_ann_sales', 'compustat_annual_sale'),
    ('Productivity · quarterly', 'compustat_qtr_tfp', 'compustat_quarterly_tfp'),
    ('Productivity · annual', 'compustat_ann_tfp', 'compustat_annual_tfp'),
    ('Employment · annual', 'compustat_ann_emp', 'compustat_annual_emp'),
    ('Employment · CES', 'bls_ces', 'bls_ces_employment'),
    ('Inflation · 1968–2025', 'cpi1967', 'bls_cpi_1967_inflation'),
    ('Stocks', 'crsp', 'crsp_daily_returns'),
    ('Wealth', 'forbes_full', 'forbes_billionaire_wealth'),
    ('Foerster et al. · industrial production', 'foerster_net', None),
    ('Acemoglu et al. · manufacturing TFP', 'acemoglu2012_bea1992', None),
    ('Carvalho–Gabaix · GDP proxy', 'carvalho_gabaix_bea_bls_historical', None),
    ('Atalay · output', 'atalay_epsilon_network', None),
]


def summary_path(stem):
    """Locate the paper summary for an application that ran."""
    path = CACHE / f'{stem}_summary.pkl'
    if path.is_file():
        return path
    if any(CACHE.glob(f'{stem}_M*_summary.pkl')):
        return CACHE / f'{rule_summary_stem(stem)}_summary.pkl'
    return None


def summary_rows(stem):
    path = summary_path(stem)
    frame = pd.read_pickle(path) if path else pd.DataFrame()
    return frame if frame.empty else frame.set_index('Entity')


def core_estimates():
    results = pd.read_csv(CACHE / 'applications_coefficients_data.csv')
    return results.set_index(results.prefix.str.replace(r'_M\d+$', '', regex=True))


def available_items():
    results = pd.read_csv(CACHE / "applications_coefficients_expanded_data.csv")
    return set(results.loc[results.estimate.notna(), "item"])


def direct_moments(y, sizes):
    sizes = sizes.reindex(y.index).astype(float)
    keep = sizes.gt(0) & np.isfinite(sizes)
    y, sizes = y.loc[keep], sizes.loc[keep]
    x = y.to_numpy(float)
    w = (sizes / sizes.sum()).to_numpy(float)
    assert np.isfinite(x).all()
    return x, w, float(np.var(w @ x, ddof=1))


def naive_moments(x, exposure_squared, denominator):
    variances = np.var(x, axis=1, ddof=1)
    exposure_squared = np.asarray(exposure_squared, dtype=float)
    mean_variance = float(variances.mean())
    hetero = float(exposure_squared @ variances / denominator)
    homo = float(exposure_squared.sum() * mean_variance / denominator)
    return dict(naive=hetero, naive_homoscedastic=homo)


def clean_panel_moments(prefix, summary):
    source = summary.get('clean_panel_path')
    if pd.isna(source):
        raise ValueError(f'Rerun {prefix} with panel caching enabled to record its exact input.')
    d = pd.read_pickle(ROOT / source)
    x, w, denominator = direct_moments(d.get('y', d.get('wide')), d['sizes'])
    dates = ('weight_date', 'start_date', 'last_date')
    if (
        any(str(d.get(date)) != str(summary[date]) for date in dates)
        or x.shape != (int(summary.N), int(summary['T']))
        or not np.isclose(w @ w, summary.HHI, rtol=1e-9)
        or not np.isclose(denominator, summary.aggregate_variance, rtol=1e-8)
    ):
        raise ValueError(f'Recorded cleaned panel does not match the saved summary for {prefix}: {source}')
    return x, w, denominator, source


def cached_naive(prefix, summary):
    x, w, denominator, source = clean_panel_moments(prefix, summary)
    return naive_moments(x, w**2, denominator), denominator, source


def primitive_naive(prefix, summary):
    if prefix.startswith('acemoglu') or prefix.startswith('carvalho'):
        if prefix.startswith('acemoglu'):
            path = DERIVED_CACHE / 'acemoglu2012' / 'bea1992_common_shocks.csv'
            unit = 'component'
        else:
            path = DERIVED_CACHE / 'carvalho_gabaix_proxy' / 'bea_bls_historical_primitive_shocks_long.csv'
            unit = 'industry_code'
        d = pd.read_csv(path)
        y = d.pivot(index=unit, columns='year', values='primitive_productivity_shock')
        sizes = d.groupby(unit).fixed_influence_weight.first()
        x, w, denominator = direct_moments(y, sizes)
        exposure_squared = w**2
        if prefix.startswith('carvalho'):
            # The saved summary expresses variance in GDP units, not normalized weights.
            assert np.isclose(sizes.sum(), summary.initial_domar_weight_sum)
            scale = float(summary.gdp_multiplier_mu * sizes.sum())**2
            exposure_squared *= scale
            denominator *= scale
    elif prefix.startswith('atalay'):
        path = DERIVED_CACHE / 'atalay2017/atalay_benchmark_productivity_shocks.csv'
        d = pd.read_csv(path)
        x = d.filter(regex='^shock_').T.to_numpy(float)
        v = pd.read_csv(path.with_name('atalay_benchmark_network_variance_matrix.csv'), header=None).to_numpy(float)
        cov = np.cov(x, ddof=1)
        denominator = float(np.sum(v * cov))
        exposure_squared = np.diag(v)
    else:
        path = DEFAULT_EPSILON_CSV
        eps, v = load_foerster_inputs()
        x = eps.T.to_numpy(float)
        cov = np.cov(x, ddof=1)
        denominator = float(np.sum(v * cov))
        exposure_squared = np.diag(v)
    assert x.shape == (int(summary.N), int(summary['T'])), (prefix, x.shape)
    assert np.isclose(denominator, summary.aggregate_variance, rtol=1e-8), (prefix, denominator, summary.aggregate_variance)
    return naive_moments(x, exposure_squared, denominator), denominator, str(path.relative_to(ROOT))


def main():
    values = []
    for label, stem, clean in ROWS:
        source = summary_path(stem)
        if source is None:
            continue
        prefix = source.stem.removesuffix('_summary')
        s = pd.read_pickle(source).iloc[0]
        benchmarks, denominator, moments_source = cached_naive(clean, s) if clean else primitive_naive(prefix, s)
        values.append(dict(label=label, prefix=prefix, estimate=s.mgp_share,
                           ci_low=s.moving_block_gradient_bootstrap_p05,
                           ci_high=s.moving_block_gradient_bootstrap_p95, **benchmarks,
                           N=int(s.N), T=int(s['T']), M=int(s.M), hhi=s.HHI,
                           start=str(s.start_date), end=str(s.last_date),
                           denominator=denominator, summary_source=str(source.relative_to(ROOT)),
                           moments_source=moments_source))
        print(f"{label}: hetero {benchmarks['naive']:.1%}, homo {benchmarks['naive_homoscedastic']:.1%}", flush=True)
    columns = ('label', 'prefix', 'estimate', 'ci_low', 'ci_high', 'naive',
               'naive_homoscedastic', 'N', 'T', 'M', 'hhi',
               'start', 'end', 'denominator', 'summary_source', 'moments_source')
    df = pd.DataFrame(values, columns=columns)
    assert (df.ci_low <= df.ci_high).all()
    df.to_csv(CACHE / 'applications_coefficients_data.csv', index=False)


if __name__ == '__main__':
    main()
