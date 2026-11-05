#!/usr/bin/env python3
"""Factor-rank tests for every application in Figure 1.

The preferred dimension is M=10.  For shorter panels, M is the largest integer
for which the ordinary covariance matrix of the M(M+1)/2 distinct second
moments can be nonsingular, i.e. M(M+1)/2 <= T-1.

Grouped panels are cached one at a time because several source reconstructions
are expensive.  Rank-test results are likewise checkpointed by application, so
an interrupted high-dimensional run can be resumed without repeating finished
applications.
"""

from __future__ import annotations

from wrds_inputs import ANALYSIS_LAST_QUARTER, wrds_path

import argparse
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from factor_rank import factor_rank_ledermann, factor_rank_tests, j_test_reportable
from mgp_core import ROOT, RUN_OUTPUT_CACHE, align_sizes_to_panel, build_lpt_grouped_data, lpt_buckets
from plot_expanded_applications import HIERARCHY, additional_loaders, available_items, innovation_loaders


PANEL_CACHE = ROOT / "code/.cache/factor_rank_figure1_panels"
RESULT_CACHE = RUN_OUTPUT_CACHE / "figure1_factor_rank"
SUMMARY_PKL = RUN_OUTPUT_CACHE / "figure1_factor_rank_summary.pkl"
SUMMARY_CSV = RUN_OUTPUT_CACHE / "figure1_factor_rank_summary.csv"
NETWORK_ITEMS = {
    "No I/O network or lags", "Foerster et al. (2011)", "Acemoglu et al. (2012)",
    "Carvalho–Gabaix (2013)", "Atalay (2017)", "Stella (2015)",
    "Welburn et al. (2020)", "Herskovic et al. (2020), Schur shocks",
}


@dataclass
class Panel:
    section: str
    item: str
    dataset: str
    x: np.ndarray
    n: int

    @property
    def t(self) -> int:
        return int(self.x.shape[0])

    @property
    def m(self) -> int:
        return int(self.x.shape[1])


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def panel_fingerprint(x: np.ndarray) -> str:
    """Identify the actual grouped observations, including their order."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    digest = hashlib.sha256(np.asarray(x.shape, dtype=np.int64).tobytes())
    digest.update(x.tobytes())
    return digest.hexdigest()


def rank_test_m(n: int, t: int) -> int:
    """Largest admissible rank-test dimension, capped at the preferred M=10."""
    moment_cap = int(np.floor((np.sqrt(8 * int(t) - 7) - 1) / 2))
    m = min(10, int(n), moment_cap)
    if m < 3:
        raise ValueError(f"Factor-rank test needs M>=3, but N={n}, T={t} allow only M={m}.")
    return m


def grouped_application(section: str, item: str, dataset: str, app, m: int | None = None) -> Panel:
    y, sizes = align_sizes_to_panel(app.y, app.sizes)
    shares = (sizes / sizes.sum()).to_numpy(float)
    m = int(m if m is not None else rank_test_m(y.shape[0], y.shape[1]))
    if getattr(app, "aggregation", "weighted") == "sum":
        y = app.relative_outcomes(y, sizes)
    x, _, _, _ = build_lpt_grouped_data(y.to_numpy(float), shares, m)
    return Panel(section, item, dataset, x, int(y.shape[0]))


def _wanted(selected: set[str], *names: str) -> bool:
    return not selected or any(
        choice == name or choice.startswith(f"{name}_M")
        for choice in selected for name in names
    )


def core_panels(selected: set[str] | None = None, available: set[str] | None = None) -> list[Panel]:
    from applications.compustat import run_compustat_quarterly as quarterly
    import run_multi_source as multi

    selected = set(selected or [])
    panels: list[Panel] = []

    def wanted(dataset, item):
        return (available is None or item in available) and _wanted(selected, dataset, item)

    if wanted("compustat_quarterly_sales", "Sales"):
        qargs = argparse.Namespace(
            compustat_csv=wrds_path("compustat_quarterly_sales"),
            last_date=ANALYSIS_LAST_QUARTER, T=100, coverage_threshold=0.8,
            use_panel_cache=True, refresh_panel_cache=False,
        )
        panel = quarterly.load_compustat_panel(qargs)
        y = panel.wide.to_numpy(dtype=float)
        sizes = panel.sizes.to_numpy(dtype=float)
        x, *_ = build_lpt_grouped_data(y - y.mean(), sizes / sizes.sum(), 10)
        panels.append(Panel("Firms", "Sales", "compustat_quarterly_sales", x, y.shape[0]))
    if wanted("compustat_annual_employment", "Employment"):
        annual_emp = multi.load_compustat_annual_panel("emp", 50, True, False)
        panels.append(grouped_application("Firms", "Employment", "compustat_annual_employment", annual_emp))
    if wanted("compustat_quarterly_tfp", "Productivity"):
        qtfp = multi.load_compustat_quarterly_tfp_panel(100, ANALYSIS_LAST_QUARTER, True, False)
        panels.append(grouped_application("Firms", "Productivity", "compustat_quarterly_tfp", qtfp))
    if wanted("crsp_daily_returns_rank", "Stocks"):
        crsp = multi.load_crsp_panel(20 * 252, True, False)
        panels.append(grouped_application("Returns", "Stocks", "crsp_daily_returns_rank_M10", crsp))

    use_returns = wanted("mutual_fund_returns", "Mutual funds")
    use_flows = wanted("Mutual fund portfolios: flows", "Mutual fund flows")
    if use_returns or use_flows:
        from applications.mutual_funds import run_mutual_fund_applications as mutual
        return_app, flow_app = mutual.load_applications()
        if use_returns:
            panels.append(grouped_application("Returns", "Mutual funds", "mutual_fund_returns", return_app))
        if use_flows:
            panels.append(grouped_application("Financial flows", "Mutual fund flows", flow_app.entity, flow_app))

    if wanted("forbes_full", "Forbes Billionaires"):
        forbes = multi.load_forbes_billionaire_panels(None, multi.FORBES_ARCHIVE_DAILY, True, False)
        full = next(panel for panel in forbes if panel.name == "forbes_billionaire_wealth_daily_full")
        forbes_panel = grouped_application("Returns", "Forbes Billionaires", "forbes_full", full)
        forbes_panel.dataset = f"forbes_full_M{forbes_panel.m}"
        panels.append(forbes_panel)
    if wanted("bls_ces", "Employment by sector"):
        bls = multi.load_bls_ces_panel(True, False)
        panels.append(grouped_application("Prices and labor", "Employment by sector", "bls_ces", bls))
    if wanted("cpi1967", "Price by item"):
        cpi = multi.load_bls_cpi_panel(True, False)
        cpi_panel = grouped_application("Prices and labor", "Price by item", "cpi1967", cpi)
        cpi_panel.dataset = f"cpi1967_M{cpi_panel.m}"
        panels.append(cpi_panel)
    return panels


def loaded_panels(loaders, prefix, available):
    sections = {item: section for section, items in HIERARCHY for item in items}
    panels = []
    for item, loader in loaders.items():
        if available is None or item in available:
            app = loader()
            panels.append(grouped_application(sections[item], item, f"{prefix}_{slug(app.entity)}", app))
    return panels


def additional_panels(available=None) -> list[Panel]:
    return loaded_panels(additional_loaders(), "additional", available)


def innovation_panels(available=None) -> list[Panel]:
    return loaded_panels(innovation_loaders(), "innovation", available)


def election_panels(available=None) -> list[Panel]:
    from applications.elections import run_election_applications as election

    jobs = (
        ("Democratic votes by state", "election_presidential_votes_by_state", election.state_application),
        ("Democratic votes by county", "election_presidential_votes_by_county", election.county_application),
    )
    jobs = [job for job in jobs if available is None or job[0] in available]
    if not jobs:
        return []
    data = election.read_returns()
    return [grouped_application("Elections", item, dataset, load(data)[0]) for item, dataset, load in jobs]


def network_panels(available=None) -> list[Panel]:
    from applications.acemoglu.run_acemoglu2012_mgp import DERIVED as ACEMOGLU_DERIVED
    from applications.atalay import run_atalay_epsilon as atalay
    from applications.carvalho_gabaix import run_carvalho_gabaix_proxy as cg
    from applications.foerster import run_foerster_epsilon as foerster
    from applications.herskovic import run_herskovic_firm_network as herskovic
    import run_sector_output_baseline as baseline
    from applications.stella import run_stella_firm_network as stella
    from applications.welburn import welburn_network as welburn

    panels: list[Panel] = []

    if available is None or "No I/O network or lags" in available:
        tfp, weights, _ = baseline.load_baseline()
        m_baseline = rank_test_m(len(weights), tfp.shape[1])
        buckets, _ = lpt_buckets(weights**2, m_baseline)
        grouped = tfp.to_numpy(float).T @ baseline.grouping_matrix(weights, buckets).T
        panels.append(Panel("Sector networks", "No I/O network or lags", f"bea_bls_no_network_rank_M{m_baseline}", grouped, len(weights)))

    if available is None or "Foerster et al. (2011)" in available:
        f = foerster.load_foerster_epsilon_panel(
            foerster.DEFAULT_EPSILON_CSV, foerster.DEFAULT_WEIGHTS_CSV,
            foerster.DEFAULT_COLUMNS_CSV, 0,
        )
        foerster_panel = grouped_application("Sector networks", "Foerster et al. (2011)", "foerster_net", f, None)
        foerster_panel.dataset = f"foerster_net_M{foerster_panel.m}"
        panels.append(foerster_panel)

    if available is None or "Acemoglu et al. (2012)" in available:
        a_panel = pd.read_csv(ACEMOGLU_DERIVED / "bea1992_common_shocks.csv")
        primitive = a_panel.pivot(index="component", columns="year", values="primitive_productivity_shock").sort_index()
        influence = a_panel.groupby("component")["fixed_influence_weight"].first().reindex(primitive.index)
        m_ace = rank_test_m(primitive.shape[0], primitive.shape[1])
        x, _, _, _ = build_lpt_grouped_data(primitive.to_numpy(float), (influence / influence.sum()).to_numpy(float), m_ace)
        panels.append(Panel("Sector networks", "Acemoglu et al. (2012)", f"acemoglu2012_M{m_ace}", x, len(primitive)))

    if available is None or "Carvalho–Gabaix (2013)" in available:
        historical, _ = cg.load_historical_panel()
        historical, _ = cg.invert_measured_tfp(historical, cg.Calibration())
        primitive = historical.pivot(index="industry_code", columns="year", values="primitive_productivity_shock").sort_index()
        influence = historical.groupby("industry_code")["fixed_influence_weight"].first().reindex(primitive.index)
        m_cg = rank_test_m(primitive.shape[0], primitive.shape[1])
        x, _, _, _ = build_lpt_grouped_data(primitive.to_numpy(float), (influence / influence.sum()).to_numpy(float), m_cg)
        panels.append(Panel("Sector networks", "Carvalho–Gabaix (2013)", f"carvalho_gabaix_M{m_cg}", x, len(primitive)))

    if available is None or "Atalay (2017)" in available:
        a, _ = atalay.load_atalay_panel(atalay.DEFAULT_SHOCKS_CSV, atalay.DEFAULT_NETWORK_V_CSV, 5)
        atalay_panel = grouped_application("Sector networks", "Atalay (2017)", "atalay", a, None)
        atalay_panel.dataset = f"atalay_M{atalay_panel.m}"
        panels.append(atalay_panel)

    if available is None or "Stella (2015)" in available:
        ids = stella.load_firm_ids(stella.PAPER_CALIBRATION)
        _, full_shocks, _, _, network_v = stella.load_dynamic_panel(stella.PAPER_CALIBRATION, ids)
        shocks = full_shocks[2:]
        exposures = np.sqrt(np.maximum(np.diag(network_v), 0))
        m_stella = rank_test_m(len(ids), len(shocks))
        x, _, _, _ = build_lpt_grouped_data(shocks.T, exposures / exposures.sum(), m_stella)
        panels.append(Panel(
            "Firm networks", "Stella (2015)",
            f"stella_revtq_dynamic_rank_M{m_stella}", x, len(ids),
        ))

    if available is None or "Welburn et al. (2020)" in available:
        w_inputs = welburn.build_welburn_inputs()
        m_welburn = rank_test_m(len(w_inputs.ids), len(w_inputs.dates))
        x, _, _, _ = build_lpt_grouped_data(
            w_inputs.outcomes.T, w_inputs.shares, m_welburn,
        )
        panels.append(Panel(
            "Firm networks", "Welburn et al. (2020)",
            f"welburn_cutoff_0135pct_rank_M{m_welburn}", x, len(w_inputs.ids),
        ))

    if available is None or "Herskovic et al. (2020)" in available:
        h_inputs = herskovic.build_herskovic_inputs()
        m_herskovic = rank_test_m(len(h_inputs.ids), len(h_inputs.dates))
        x, _, _, _ = build_lpt_grouped_data(
            h_inputs.outcomes.T, h_inputs.shares, m_herskovic,
        )
        panels.append(Panel(
            "Firm networks", "Herskovic et al. (2020), Schur shocks",
            f"herskovic_rank_M{m_herskovic}", x, len(h_inputs.ids),
        ))

    return panels


def _selected_builders(selected: set[str], builders: list) -> list:
    """Restrict a targeted refresh to the application family it can affect."""
    if selected and selected <= NETWORK_ITEMS:
        return [builder for builder in builders if builder.__name__ == "network_panels"]
    prefixes = {
        "core_panels": ("compustat_", "crsp_", "mutual_fund_", "forbes_", "bls_", "cpi"),
        "additional_panels": ("additional_",),
        "innovation_panels": ("innovation_",),
        "election_panels": ("election_",),
        "network_panels": (
            "bea_bls_", "foerster_", "acemoglu", "carvalho_", "atalay_",
            "stella_", "welburn_", "herskovic_",
        ),
    }
    matched = [
        builder for builder in builders
        if any(value.startswith(prefixes[builder.__name__]) for value in selected)
    ]
    # Item-label selections do not have stable prefixes.  Falling back to all
    # builders preserves the old behavior for those uncommon calls.
    return matched or builders


def build_panel_cache(
    refresh: bool, only: list[str] | None = None, available: set[str] | None = None,
) -> list[dict[str, object]]:
    PANEL_CACHE.mkdir(parents=True, exist_ok=True)
    selected = set(only or [])
    targeted_refresh = bool(refresh and selected)
    if refresh and not targeted_refresh:
        # Dataset names encode dimensions for several applications.  Clear the
        # generated panel cache before a refresh so a changed paper-wide M rule
        # cannot leave both the old and new versions in the manifest.
        for path in PANEL_CACHE.glob("*.pkl"):
            path.unlink()
        for path in PANEL_CACHE.glob("_*.done"):
            path.unlink()
    builders = [core_panels, additional_panels, innovation_panels, election_panels, network_panels]
    if selected:
        builders = _selected_builders(selected, builders)
    manifest: list[dict[str, object]] = []
    for builder in builders:
        group_cache = PANEL_CACHE / f"_{builder.__name__}.done"
        if group_cache.exists() and not refresh:
            continue
        built = core_panels(selected, available) if builder is core_panels else builder(available)
        for panel in built:
            # A changed reconstruction gets a new dataset key. Retain only
            # its current grouped input in the active panel manifest.
            for prior_path in PANEL_CACHE.glob("*.pkl"):
                prior_payload = pd.read_pickle(prior_path)
                if prior_payload["item"] == panel.item and prior_payload["dataset"] != panel.dataset:
                    prior_path.unlink()
            payload = {
                "section": panel.section, "item": panel.item, "dataset": panel.dataset,
                "N": panel.n, "T": panel.t, "M": panel.m, "x": panel.x,
            }
            pd.to_pickle(payload, PANEL_CACHE / f"{slug(panel.dataset)}.pkl")
            print(f"[panel] {panel.item}: N={panel.n}, T={panel.t}, M={panel.m}", flush=True)
        if not selected or group_cache.exists():
            group_cache.touch()
    for path in sorted(PANEL_CACHE.glob("*.pkl")):
        payload = pd.read_pickle(path)
        manifest.append({k: payload[k] for k in ("section", "item", "dataset", "N", "T", "M")})
    # A targeted refresh does not need to reconstruct unrelated grouped data.
    # Their dimensions remain available in the consolidated result cache.
    if (targeted_refresh or not refresh) and SUMMARY_PKL.exists():
        prior = pd.read_pickle(SUMMARY_PKL)
        previous_dataset = dict(zip(prior["item"], prior["dataset"], strict=True))
        by_item = {}
        for row in manifest:
            item = str(row["item"])
            if item not in by_item or row["dataset"] == previous_dataset.get(item):
                by_item[item] = row
        manifest = list(by_item.values())
        present = {str(row["dataset"]) for row in manifest}
        present_items = {str(row["item"]) for row in manifest}
        for _, row in prior.iterrows():
            if str(row["dataset"]) not in present and str(row["item"]) not in present_items:
                manifest.append({k: row[k] for k in ("section", "item", "dataset", "N", "T", "M")})
    if available is not None:
        manifest = [row for row in manifest if str(row["item"]).split(", Schur shocks")[0] in available]
    return manifest


def run_tests(manifest: list[dict[str, object]], args: argparse.Namespace) -> None:
    RESULT_CACHE.mkdir(parents=True, exist_ok=True)
    selected = set(args.only or [])
    not_estimable = set(args.mark_not_estimable or [])
    for meta in manifest:
        if selected and meta["dataset"] not in selected and meta["item"] not in selected:
            continue
        if int(meta["M"]) > args.max_dimension:
            continue
        result_path = RESULT_CACHE / f"{slug(str(meta['dataset']))}.pkl"
        if (meta["dataset"] in not_estimable or meta["item"] in not_estimable
                or not j_test_reportable(int(meta["M"]), int(meta["T"]))):
            pd.to_pickle(
                {
                    **meta,
                    "rank_test_status": "not_estimable",
                    "skip_reason": (
                        f"Distribution-free J test not estimable at M={meta['M']}, T={meta['T']}: "
                        "the ordinary covariance weight requires both M<=10 and "
                        "T-1 >= M(M+1)/2."
                    ),
                },
                result_path,
            )
            print(f"[rank] marked not estimable: {meta['item']} (M={meta['M']})", flush=True)
            combine_results(manifest)
            continue
        payload = pd.read_pickle(PANEL_CACHE / f"{slug(str(meta['dataset']))}.pkl")
        x = np.asarray(payload["x"], dtype=float)
        fingerprint = panel_fingerprint(x)
        if result_path.exists() and not args.force:
            cached = pd.read_pickle(result_path)
            dimensions_match = all(
                int(cached.get(key, -1)) == int(meta[key]) for key in ("N", "T", "M")
            )
            if (dimensions_match and cached.get("rank_test_status") == "ok"
                    and cached.get("input_fingerprint") == fingerprint
                    and cached.get("alpha") == args.alpha):
                print(f"[rank] cached {meta['item']}", flush=True)
                continue
            print(f"[rank] stale cache, recomputing {meta['item']}", flush=True)
        print(f"\n[rank] {meta['item']}: T={x.shape[0]}, M={x.shape[1]}", flush=True)
        # Include the diagonal (zero-factor) model so "smallest non-rejected"
        # means exactly what the text says, rather than being truncated at one.
        min_rank = 0
        try:
            detail, summary = factor_rank_tests(
                x, alpha=args.alpha, min_rank=min_rank, include_gmm=True,
                maxiter_ml=args.maxiter_ml, maxiter_gmm=args.maxiter_gmm,
                include_corrected=False,
                stop_after_first_j_accept=False,
                rank_search="safeguarded_secant" if int(meta["M"]) > 10 else "sequential",
                progress_label=str(meta["dataset"]),
            )
            detail.to_pickle(RESULT_CACHE / f"{slug(str(meta['dataset']))}_detail.pkl")
            pd.to_pickle({**meta, "rank_test_status": "ok", **summary,
                          "input_fingerprint": fingerprint}, result_path)
        except Exception as exc:
            pd.to_pickle(
                {**meta, "rank_test_status": "error", "input_fingerprint": fingerprint,
                 "skip_reason": f"{type(exc).__name__}: {exc}"},
                result_path,
            )
            print(f"[rank] ERROR {meta['item']}: {type(exc).__name__}: {exc}", flush=True)
        combine_results(manifest)


def combine_results(manifest: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for meta in manifest:
        path = RESULT_CACHE / f"{slug(str(meta['dataset']))}.pkl"
        if path.exists():
            row = pd.read_pickle(path)
            if pd.isna(row.get("rank_test_status", np.nan)) and pd.notna(row.get("factor_lb_J", np.nan)):
                row = {**row, "rank_test_status": "ok"}
        else:
            row = {
                **meta,
                "rank_test_status": "not_reportable" if not j_test_reportable(int(meta["M"]), int(meta["T"])) else "pending",
                "skip_reason": (
                    f"J weight not estimable at M={meta['M']}, T={meta['T']}"
                    if not j_test_reportable(int(meta["M"]), int(meta["T"])) else "Rank test has not been run"
                ),
            }
        rows.append(row)
    frame = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=("section", "item", "dataset", "N", "T", "M", "rank_test_status"),
    )
    frame.to_pickle(SUMMARY_PKL)
    frame.to_csv(SUMMARY_CSV, index=False)
    print(f"\n[summary] {len(frame)}/{len(manifest)} Figure 1 applications complete", flush=True)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--refresh-panels", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--available-only", action="store_true", help="Use applications in the current Figure 1 results.")
    parser.add_argument("--networks-only", action="store_true",
                        help="Refresh network inputs and retain other cached applications.")
    parser.add_argument("--mark-not-estimable", nargs="+")
    parser.add_argument("--max-dimension", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--maxiter-ml", type=int, default=2000)
    parser.add_argument("--maxiter-gmm", type=int, default=2000)
    args = parser.parse_args()
    if args.networks_only:
        if args.only:
            parser.error("Use either --networks-only or --only.")
        args.only = sorted(NETWORK_ITEMS)
        args.refresh_panels = True
    available = available_items() if args.available_only else None
    manifest = build_panel_cache(args.refresh_panels, args.only, available)
    expected = len(available) if available is not None else sum(len(items) for _, items in HIERARCHY)
    if len(manifest) != expected:
        raise RuntimeError(f"Expected {expected} Figure 1 panels, found {len(manifest)}")
    if not args.build_only:
        run_tests(manifest, args)
    combine_results(manifest)


if __name__ == "__main__":
    main()
