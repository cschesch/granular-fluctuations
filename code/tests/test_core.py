#!/usr/bin/env python3
"""Fast invariant checks for the estimation core (`make test`, runs in seconds).

Checks, on a synthetic factor DGP with known diagonal part:
  1. grouping preserves the size-weighted aggregate exactly;
  2. LPT loads partition the jobs and satisfy the list-scheduling bound;
  3. primal MTFA and dual SDP values agree (strong duality);
  4. the grouped MGP upper-bounds the true granular level S'D0S (Prop. popUB);
  5. the reported share is the level over aggregate variance and lies in [0,1].
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mgp_core import (
    analysis_block_length,
    build_lpt_grouped_data,
    choose_solver,
    fill_short_internal_return_gaps,
    mgp_share,
    remove_periodic_means,
    sample_cov,
    standardize_panel,
    solve_grouped_mgp_dual,
    solve_primal_mtfa,
)
from applications.networks import (
    network_denominator,
    network_direct_diagonal_share,
    network_share_from_covariances,
    network_share_directional_derivative,
)
from applications.common import Application, prepare_flows, estimate
from applications.epa.panel import load_campd
from applications.mutual_funds.run_mutual_fund_applications import (attach_portfolios, build_portfolio_panels,
                                           portfolio_applications, daily_portfolio_returns,
                                           eligible_share_classes)


def main() -> None:
    rng = np.random.default_rng(0)
    solver = choose_solver("auto")
    n, t, m, r = 60, 400, 8, 3

    lam = 0.3 * rng.standard_normal((n, r))
    d_true = rng.uniform(0.5, 2.0, n)
    sizes = np.sort(rng.pareto(1.5, n) + 0.1)[::-1]
    shares = sizes / sizes.sum()
    y = (rng.standard_normal((t, r)) @ lam.T + rng.standard_normal((t, n)) * np.sqrt(d_true)).T

    cleaned = standardize_panel(pd.DataFrame([[0.0, 1.0, 100.0], [2.0, 3.0, 4.0]]))
    assert np.isclose(cleaned.to_numpy().mean(), 0.0), "panel cleaning does not grand-mean center"
    assert analysis_block_length(50) == 4
    assert analysis_block_length(97, overlap_horizon=12) == 12
    assert analysis_block_length(4, overlap_horizon=12) == 4

    x_grouped, _, loads, g = build_lpt_grouped_data(y, shares, m)

    assert np.allclose(x_grouped.sum(axis=1), shares @ y), "aggregation identity failed"

    jobs = shares**2
    assert np.isclose(loads.sum(), jobs.sum()), "LPT loads do not partition the jobs"
    assert loads.max() <= loads.mean() + jobs.max() + 1e-12, "list-scheduling load bound failed"

    sigma = sample_cov(x_grouped)
    dual_value = float(solve_grouped_mgp_dual(sigma, solver)["value"])
    _, primal_value = solve_primal_mtfa(sigma, solver)
    assert abs(primal_value - dual_value) <= 1e-5 * max(abs(dual_value), 1e-12), (
        f"duality gap {primal_value - dual_value:g}"
    )

    sigma_pop = lam @ lam.T + np.diag(d_true)
    grouped_pop = g @ sigma_pop @ g.T
    bound = float(solve_grouped_mgp_dual(grouped_pop, solver)["value"])
    truth = float(jobs @ d_true)
    assert bound >= truth * (1 - 1e-8), f"population upper bound violated: {bound:g} < {truth:g}"

    obs = mgp_share(x_grouped, solver)
    one = np.ones(m)
    assert -1e-9 <= obs["mgp_share"] <= 1 + 1e-9, "share outside [0,1]"
    assert np.isclose(obs["mgp_share"], obs["mgp_level"] / float(one @ sigma @ one)), "share != level / agg var"

    # Relative flows use the same size shares in grouping and aggregation.
    raw_fluctuations = y * sizes[:, None]
    flow_grouped, raw_buckets, raw_loads, _ = build_lpt_grouped_data(
        raw_fluctuations / sizes[:, None], shares, m
    )
    raw_grouped = np.column_stack([raw_fluctuations[bucket].sum(axis=0) for bucket in raw_buckets])
    assert np.allclose(flow_grouped.sum(axis=1), raw_fluctuations.sum(axis=0) / sizes.sum())
    assert np.allclose(raw_grouped / sizes.sum(), flow_grouped)
    assert np.allclose(flow_grouped, x_grouped)
    assert np.array_equal(raw_loads, loads)
    raw_obs = mgp_share(raw_grouped, solver)
    assert np.isclose(raw_obs["mgp_share"], obs["mgp_share"], atol=1e-6)
    raw_app = Application("synthetic flows", "flow", "", "", pd.DataFrame(raw_fluctuations),
                          pd.Series(sizes), pd.Series(np.arange(n)), "", "Annual", "mean", "",
                          aggregation="sum")
    raw_row = estimate(raw_app, solver, bootstrap_b=4, seed=0, m_override=m)
    assert np.isclose(raw_row["Direct_diagonal"],
                      np.var(raw_fluctuations, axis=1, ddof=1).sum()
                      / np.var(raw_fluctuations.sum(axis=0), ddof=1))
    assert np.isclose(raw_row["Naive_homoscedastic"],
                      (shares @ shares) * np.var(y, axis=1, ddof=1).mean()
                      / np.var(shares @ y, ddof=1))

    # Quarterly flows retain levels, zeros, trends, and seasonal variation.
    flow_levels = pd.DataFrame(
        [[0, 5, 4, 9, 6, 13, 8, 17], [3, 2, 7, 4, 11, 6, 15, 8]],
        index=["a", "b"],
        columns=pd.period_range("2020Q1", periods=8, freq="Q"),
        dtype=float,
    )
    flow_levels.iloc[0, -1] = 10000
    flow_levels.loc["inactive"] = 0.0
    flow_y, flow_sizes = prepare_flows(flow_levels)
    assert "inactive" not in flow_y.index
    assert np.allclose(flow_sizes, flow_levels.loc[flow_y.index].mean(axis=1))
    pd.testing.assert_frame_equal(flow_y, flow_levels.drop(index="inactive"))
    app = Application("test", "flow", "", "", flow_y, flow_sizes,
                      pd.Series(flow_y.index, index=flow_y.index), "", "Annual", "mean", "",
                      aggregation="sum")
    assert np.array_equal(app.y.to_numpy(), flow_y.to_numpy()), "additive flows were clipped or centered"
    relative = app.relative_outcomes(app.y, app.sizes)
    weights = app.sizes / app.sizes.sum()
    assert relative.loc["a"].iloc[0] == 0.
    np.testing.assert_allclose(weights @ relative, flow_y.sum(axis=0) / flow_sizes.sum())
    # Bond quarter effects preserve aggregation; sizes remain mean raw flows.
    adjusted = remove_periodic_means(flow_y, 4)
    seasonal_total = remove_periodic_means(flow_y.sum(axis=0).to_frame().T, 4).iloc[0]
    np.testing.assert_allclose(weights @ adjusted.div(flow_sizes, axis=0),
                               seasonal_total / flow_sizes.sum(), atol=1e-14)
    signed = pd.DataFrame([[-2, -1, 0, -4, -1, 0, -2, -3], [-1, 1, -1, 1, -1, 1, -1, 1]],
                          index=["outflows", "zero_mean"], columns=flow_levels.columns)
    signed_y, _ = prepare_flows(signed)
    assert list(signed_y.index) == list(signed.index), "signed flows were incorrectly dropped"
    pd.testing.assert_frame_equal(signed_y, signed.astype(float))

    # Monthly flow accounting and outcome-window mean AUM; no seasonal terms.
    months = pd.period_range("2010-12", "2025-12", freq="M")
    fund_tna, fund_returns = [], []
    for fund, base in (("a", 100.0), ("b", 300.0)):
        for k, month in enumerate(months):
            fund_tna.append(dict(crsp_portno=fund, crsp_fundno=fund, month=month, tna=base + 2*k))
            fund_returns.append(dict(crsp_portno=fund, crsp_fundno=fund, month=month, mret=0.01))
    def monthly_fund_panels(tna, returns, *, closed=False):
        ids = tna.crsp_fundno.unique()
        headers = pd.DataFrame({"crsp_fundno": ids, "m_fund": "N", "et_flag": "",
                                "first_offer_dt": "2010-01-01", "end_dt": "2025-12-31",
                                "dead_flag": "N", "delist_cd": ""})
        mapping = tna[["crsp_fundno", "crsp_portno"]].drop_duplicates().assign(begdt="2010-01-01", enddt=None)
        if closed:
            headers.loc[headers.crsp_fundno.eq("b"), ["end_dt", "dead_flag", "delist_cd"]] = ["2011-02-28", "Y", "L"]
        return build_portfolio_panels(tna, returns, headers, mapping)
    fund_r, fund_f, mean_aum, _ = monthly_fund_panels(pd.DataFrame(fund_tna), pd.DataFrame(fund_returns))
    assert np.allclose(mean_aum, [100 + 2*np.arange(1,181).mean(), 300 + 2*np.arange(1,181).mean()])
    assert np.isclose(fund_f.loc["a"].iloc[0], 102 - 100*1.01)
    assert np.isclose(fund_f.loc["b"].iloc[0], 302 - 300*1.01)
    # Absent class assets are zero; sibling returns fill missing class returns.
    classes = pd.DataFrame(fund_tna)
    monthly = pd.DataFrame(fund_returns)
    sibling = classes.loc[classes.crsp_portno.eq("a")].copy()
    sibling["crsp_fundno"] = "a2"
    sibling["tna"] = 50.
    classes = pd.concat([classes, sibling], ignore_index=True)
    monthly.loc[monthly.crsp_portno.eq("a"), "mret"] = .02
    _, peer_flows, _, _ = monthly_fund_panels(classes, monthly)
    assert np.isclose(peer_flows.loc["a"].iloc[0], 152 - 150*1.02)
    # A liquidated fund remains, including a missing liquidation-month return;
    # a positive-AUM month without any class return excludes the entire fund.
    closed = classes.loc[classes.crsp_portno.eq("b") & classes.month.le(months[2])]
    active = classes.loc[classes.crsp_portno.eq("a")]
    missing = monthly.loc[~((monthly.crsp_portno.eq("a") & monthly.month.eq(months[3])) |
                           (monthly.crsp_portno.eq("b") & monthly.month.gt(months[2])))]
    _, closed_flows, closed_sizes, _ = monthly_fund_panels(pd.concat([active, closed]), missing, closed=True)
    assert list(closed_flows.index) == ["b"]
    assert closed_flows.loc["b", months[3]] == -304.
    assert (closed_flows.loc["b", months[4:]] == 0.).all() and closed_sizes.loc["b"] > 0
    # Missing AUM in an active class is not a zero; unknown disappearance is
    # not a documented liquidation. A legitimate pre-offer period can be zero.
    incomplete = pd.DataFrame(fund_tna)
    incomplete = incomplete.loc[~(incomplete.crsp_portno.eq("a") & incomplete.month.eq(months[10]))]
    _, strict_flows, _, diag = monthly_fund_panels(incomplete, pd.DataFrame(fund_returns))
    assert list(strict_flows.index) == ["b"] and diag["dropped_unresolved_aum_portfolios"] == 1
    _, undocumented, _, _ = monthly_fund_panels(pd.concat([active, closed]), missing)
    assert undocumented.empty
    from applications.mutual_funds.run_mutual_fund_applications import documented_asset_portfolios
    entrant = pd.DataFrame(fund_tna)
    entrant = entrant.loc[~(entrant.crsp_portno.eq("a") & entrant.month.lt(months[2]))]
    entrant_returns = pd.DataFrame(fund_returns)
    entrant_returns = entrant_returns.loc[~(entrant_returns.crsp_portno.eq("a") & entrant_returns.month.lt(months[2]))]
    headers = pd.DataFrame({"crsp_fundno": ["a", "b"], "m_fund": "N", "et_flag": "",
                            "first_offer_dt": [str(months[2].start_time.date()), "2010-01-01"],
                            "end_dt": "2025-12-31", "dead_flag": "N", "delist_cd": ""})
    mapping = pd.DataFrame({"crsp_fundno": ["a", "b"], "crsp_portno": ["a", "b"], "begdt": "2010-01-01", "enddt": None})
    _, entry_flows, _, _ = build_portfolio_panels(entrant, entrant_returns, headers, mapping)
    assert entry_flows.loc["a", months[1]] == 0. and entry_flows.loc["a", months[2]] == 104.
    # An observed return before the offer date makes that month active: its
    # missing AUM must not be zero-filled using the contradictory offer date.
    observed_early = pd.concat([entrant_returns, pd.DataFrame([dict(crsp_fundno="a",crsp_portno="a",month=months[1],mret=.01)])])
    ids, _ = documented_asset_portfolios(entrant, observed_early, headers, mapping)
    assert list(ids) == ["b"]
    # Large observed flows survive the strict lifecycle cleaning.
    large_assets = pd.DataFrame(fund_tna)
    large_assets.loc[large_assets.crsp_fundno.eq("a") & large_assets.month.eq(months[10]), "tna"] = 1000.
    _, large_flows, _, _ = monthly_fund_panels(large_assets, pd.DataFrame(fund_returns))
    assert "a" in large_flows.index and large_flows.loc["a", months[10]] > 2 * 118.
    # Fund flows retain their levels, trends, and month-of-year variation.
    raw_flows = fund_f + .1*np.arange(180) + np.sin(2*np.pi*np.arange(180)/12)
    fund_apps = portfolio_applications(fund_r, raw_flows, mean_aum, mean_aum)
    assert [app.frequency for app in fund_apps] == ["Daily", "Monthly"]
    assert all(app.sizes.equals(mean_aum) for app in fund_apps)
    assert fund_apps[1].aggregation == "sum"
    pd.testing.assert_frame_equal(fund_apps[1].y, raw_flows)

    # Daily class weights use December assets throughout January. January-end
    # assets differ sharply, so using them would fail this no-look-ahead check.
    from tempfile import TemporaryDirectory
    from pathlib import Path
    daily_tna = []
    for fund, base in (("000001", 100.), ("000002", 300.)):
        for k, month in enumerate(months):
            daily_tna.append(dict(crsp_fundno=fund, crsp_portno="p", month=month,
                                  tna=base if k == 0 else 10000.))
    daily = pd.DataFrame({"crsp_fundno": ["1", "2", "1", "2"],
                          "caldt": ["2011-01-03", "2011-01-03", "2011-01-04", "2011-01-04"],
                          "dret": [.01, .03, .02, .04]})
    headers = pd.DataFrame({"crsp_fundno": ["000001", "000002"], "m_fund": ["N", "N"], "et_flag": ["N", "N"]})
    mapping = pd.DataFrame({"crsp_fundno": ["000001", "000002"], "crsp_portno": ["p", "p"],
                            "begdt": ["2010-01-01", "2010-01-01"], "enddt": [None, None]})
    with TemporaryDirectory() as folder:
        path = Path(folder) / "daily.csv"
        daily.to_csv(path, index=False)
        daily_r, daily_sizes, diagnostics = daily_portfolio_returns(path, pd.DataFrame(daily_tna), headers, mapping)
        daily.iloc[:-1].to_csv(path, index=False)
        low_coverage_r, _, _ = daily_portfolio_returns(path, pd.DataFrame(daily_tna), headers, mapping)
        assert np.allclose(low_coverage_r.loc["p"], [.025, .02]), "Observed class weights were not renormalized"
        # Other source funds supply the trading calendar when all classes of p
        # are missing. Three internal days are zero-filled at portfolio level.
        middle = pd.DataFrame({"crsp_fundno": ["999999"]*3,
                               "caldt": ["2011-01-04", "2011-01-05", "2011-01-06"],
                               "dret": [0., 0., 0.]})
        endpoints = daily.iloc[:2].copy()
        last = endpoints.copy(); last["caldt"] = "2011-01-09"
        pd.concat([endpoints, middle, last]).to_csv(path, index=False)
        bridged_r, _, fill_diag = daily_portfolio_returns(path, pd.DataFrame(daily_tna), headers, mapping)
        assert np.allclose(bridged_r.loc["p"], [.025, 0., 0., 0., .025])
        assert fill_diag["zero_filled_return_cells"] == 3
    assert np.allclose(daily_r.loc["p"], [.025, .035])
    assert daily_sizes.loc["p"] == 400., "Fixed portfolio weights must use December 2010 AUM"
    assert diagnostics["minimum_return_aum_coverage"] == 1.0
    # Forbes uses observed capture dates as its calendar. Personal short gaps
    # hold wealth fixed; reappearance retains the full catch-up log change.
    from applications.forbes.panel import forbes_level_panel
    for n_dates in (8, 510):
        capture_dates = pd.date_range("2020-01-01", periods=n_dates + 10).delete(slice(6, 16))
        timestamps = capture_dates.strftime("%Y%m%d120000").tolist()
        observations = []
        for person, missing_dates in {"complete": set(), "short": {2, 3}, "three": {2, 3, 4},
                                      "long": {2, 3, 4, 5}, "endpoint": {0}}.items():
            for j, (ts, date) in enumerate(zip(timestamps, capture_dates)):
                if j in missing_dates:
                    continue
                observations.append(dict(archive_timestamp=ts, snapshot_date=str(date.date()),
                    archive_capture_date=str(date.date()), natural_id=person, person_name=person,
                    final_worth_billion_usd=10. + j))
        with TemporaryDirectory() as folder:
            path = Path(folder) / "forbes_daily.csv"
            pd.DataFrame(observations).to_csv(path, index=False)
            wealth, _, _ = forbes_level_panel(path, timestamps)
            assert set(wealth.index) == {"complete", "short", "three"}
            assert wealth.shape[1] == n_dates, "Unobserved archive dates were added to the calendar"
            assert wealth.loc["short"].iloc[2:4].eq(11.).all()
            growth = np.log(wealth).diff(axis=1)
            assert growth.loc["short"].iloc[2:4].eq(0.).all()
            assert np.isclose(growth.loc["short"].iloc[4], np.log(14./11.))
            path = Path(folder) / "forbes_weekly.csv"
            pd.DataFrame(observations).to_csv(path, index=False)
            weekly_wealth, _, _ = forbes_level_panel(path, timestamps)
            assert list(weekly_wealth.index) == ["complete"]
    # Fill whole internal runs, never the first three cells of a longer run.
    gap_input = pd.DataFrame([[1., np.nan, 2., np.nan, np.nan, 3., np.nan, np.nan, np.nan, 4.,
                               np.nan, np.nan, np.nan, np.nan, 5.]])
    gap_result = fill_short_internal_return_gaps(gap_input)
    assert np.allclose(gap_result.iloc[0, :10], [1., 0., 2., 0., 0., 3., 0., 0., 0., 4.])
    assert gap_result.iloc[0, 10:14].isna().all()
    assert gap_result.iloc[0, -1] == 5.
    edges = pd.DataFrame([[np.nan, np.nan, 2., np.nan, 3., np.nan, np.nan], [np.nan]*7])
    edge_result = fill_short_internal_return_gaps(edges)
    assert edge_result.iloc[0, 3] == 0.
    assert edge_result.iloc[0, [0, 1, 5, 6]].isna().all() and edge_result.iloc[1].isna().all()
    pd.testing.assert_frame_equal(fill_short_internal_return_gaps(gap_input, max_gap=0), gap_input)
    # Exercise the stock loader too: the prior-date capitalization stays the
    # weight, short internal return gaps survive, and endpoint/long gaps do not.
    from unittest.mock import patch
    from applications.crsp.panel import load_crsp_panel
    stock_dates = pd.bdate_range("2020-01-01", periods=7)
    index = pd.MultiIndex.from_product([stock_dates, [1, 2, 3, 4]], names=["date", "PERMNO"])
    stock_data = pd.DataFrame({"RET": .02, "mkt_cap": 100.}, index=index)
    stock_data.loc[(stock_dates[2:5], 1), "RET"] = np.nan
    stock_data.loc[(stock_dates[2:6], 2), "RET"] = np.nan
    stock_data.loc[(stock_dates[-1], 3), "RET"] = np.nan
    stock_data.loc[(stock_dates[0], 1), "mkt_cap"] = 321.
    with TemporaryDirectory() as folder:
        source = Path(folder) / "stock_source.parquet"
        source.write_bytes(b"synthetic source fingerprint")
        with patch("applications.crsp.prepare_crsp_daily.prepare", return_value=source), \
                patch("applications.crsp.panel.pd.read_parquet", return_value=stock_data):
            stock = load_crsp_panel(6, use_cache=False)
    assert set(stock.y.index) == {1, 4}
    assert stock.sizes.loc[1] == 321.
    expected_stock = pd.DataFrame([[.02, 0., 0., 0., .02, .02], [.02]*6], index=[1, 4], columns=stock_dates[1:])
    assert np.allclose(stock.y, standardize_panel(expected_stock))
    flags = pd.DataFrame({"crsp_portno": ["fund", "etf", "etn", "money"],
                          "m_fund": ["N", "N", "N", "Y"],
                          "et_flag": [None, "F", "N", None]})
    assert eligible_share_classes(flags).crsp_portno.tolist() == ["fund"]

    # A share class changing portfolio must use the dated map, not the header's
    # current portfolio. Overlapping valid mappings must fail rather than add AUM twice.
    dates = pd.DataFrame({"crsp_fundno": ["a", "a"], "caldt": pd.to_datetime(["2015-12-31", "2016-03-31"])})
    headers = pd.DataFrame({"crsp_fundno": ["a"], "m_fund": ["N"], "et_flag": ["N"]})
    mapping = pd.DataFrame({"crsp_fundno": ["a", "a"], "crsp_portno": ["old", "new"],
                            "begdt": ["2010-01-01", "2016-01-01"], "enddt": ["2015-12-31", None]})
    mapped = attach_portfolios(dates, headers, mapping)
    assert mapped.sort_values("caldt")["crsp_portno"].tolist() == ["old", "new"]
    overlapping = mapping.copy(); overlapping.loc[0, "enddt"] = None
    try:
        attach_portfolios(dates, headers, overlapping)
    except ValueError:
        pass
    else:
        raise AssertionError("Overlapping historical portfolio mappings were accepted")

    # Check the plant transformation against an independent quarter-dummy OLS:
    # one-quarter increments, unit effects, and adjustment before clipping.
    from unittest.mock import patch
    quarters = pd.period_range("1999Q4", "2024Q4", freq="Q")
    innovations = rng.normal(0, 0.01, (2, len(quarters) - 1))
    increments = innovations + np.array([0.04, -0.02, 0.01, -0.03])[quarters[1:].quarter - 1]
    levels = np.exp(np.column_stack([np.zeros(2), np.cumsum(increments, axis=1)])) * np.array([100., 300.])[:, None]
    plant_data = pd.DataFrame([
        {"Facility ID": str(i), "Facility Name": f"Plant {i}", "Year": quarter.year,
         "Quarter": quarter.quarter, "Gross Load (MWh)": levels[i, j]}
        for i in range(2) for j, quarter in enumerate(quarters)
    ])
    with patch("applications.epa.panel.pd.read_csv", return_value=plant_data), \
            patch("applications.epa.panel.ANALYSIS_LAST_QUARTER", "2024Q4"):
        plant = load_campd("Gross Load (MWh)", "Power plants", "Gross-load growth")
    design = np.eye(4)[quarters[1:].quarter - 1]
    residuals = increments.T - design @ np.linalg.lstsq(design, increments.T, rcond=None)[0]
    expected = standardize_panel(pd.DataFrame(residuals.T, index=plant.y.index, columns=quarters[1:]))
    assert np.allclose(plant.y, expected)
    assert plant.y.shape[1] == 100 and plant.y.columns[0] == pd.Period("2000Q1")
    assert plant.dependence_horizon == 1 and plant.weight_date == "1999Q4"
    assert np.allclose(plant.sizes, [100., 300.])

    seasonal = pd.DataFrame(
        [[1, 2, 3, 4, 2, 3, 4, 5], [4, 3, 2, 1, 5, 4, 3, 2]],
        columns=pd.period_range("2020Q1", periods=8, freq="Q"),
    )
    adjusted = remove_periodic_means(seasonal, 4)
    assert np.allclose(adjusted.T.groupby(adjusted.columns.quarter).mean(), 0.0)

    # Dynamic propagation generally cannot be normalized by the variance of
    # the sum of total-influence-weighted contemporaneous innovations.  The
    # off-diagonal entries of V retain cross-horizon response geometry.
    h0 = np.array([[1.0, 0.0], [0.0, 1.0]])
    h1 = np.array([[0.0, 1.0], [1.0, 0.0]])
    agg = np.array([0.8, 0.2])
    v = h0.T @ np.outer(agg, agg) @ h0 + h1.T @ np.outer(agg, agg) @ h1
    sigma_eps = np.array([[1.0, 0.35], [0.35, 2.0]])
    structural_var = network_denominator(sigma_eps, v)
    total_influence = np.sqrt(np.diag(v))
    rank_one_var = float(total_influence @ sigma_eps @ total_influence)
    assert not np.isclose(structural_var, rank_one_var), "dynamic denominator collapsed to rank one"

    # The direct diagonal benchmark is invariant to reciprocal rescaling of
    # each primitive shock and its exposure.  This matters for recovered firm
    # innovations, whose fixed-base normalization can vary sharply by unit.
    direct_share = network_direct_diagonal_share(sigma_eps, v)
    scale = np.diag([7.0, 0.2])
    scale_inv = np.diag(1.0 / np.diag(scale))
    sigma_rescaled = scale @ sigma_eps @ scale
    v_rescaled = scale_inv @ v @ scale_inv
    assert np.isclose(
        direct_share,
        network_direct_diagonal_share(sigma_rescaled, v_rescaled),
    ), "network direct diagonal share is not scale invariant"

    # Dynamic grouping relaxes the numerator while retaining the physical
    # model-implied denominator, so the raw ratio can exceed one.  Preserve it
    # for inference but cap the reported share at its logical upper bound.
    rho = 0.6
    sigma_eps_three = np.array(
        [[1.0, rho, 0.0], [rho, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    grouped_three = np.diag([2.0 + 2.0 * rho, 1.0])
    relaxed = network_share_from_covariances(
        grouped_three, sigma_eps_three, np.eye(3), 1.0, solver
    )
    assert np.isclose(relaxed["mgp_share_uncapped"], 1.4, atol=1e-8)
    assert relaxed["mgp_share"] == 1.0

    # The implemented joint derivative is the finite-difference derivative
    # of n(Sigma_G) / <V,Sigma_eps>, including the numerator-denominator
    # covariance term through their common direction.
    numerator, denominator = 1.7, 2.4
    dn, dd = -0.3, 0.8
    share = numerator / denominator
    analytic = float(network_share_directional_derivative(dn, dd, share, denominator))
    step = 1e-7
    finite_difference = ((numerator + step * dn) / (denominator + step * dd) - share) / step
    assert np.isclose(analytic, finite_difference, rtol=1e-7, atol=1e-9), "network quotient derivative failed"

    print("all core invariants OK")


if __name__ == "__main__":
    main()
