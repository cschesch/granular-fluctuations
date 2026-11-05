#!/usr/bin/env python3
"""Homoskedastic grouped MGP (hMGP) for every Figure 1 application.

Restricting the idiosyncratic covariance to sigma_e^2 I_N pushes forward under
grouping to sigma_e^2 Diag(bucket squared-weight loads) = sigma_e^2 Lambda, so the
grouped hMGP is the closed form sigma_e^2 = lambda_min(Lambda^{-1/2} Sigma_tilde
Lambda^{-1/2}) and the share is sigma_e^2 sum_i w_i^2 over the aggregate variance
(the model-implied <V, Sigma_eps> for the dynamic network applications).  Inference uses
the paper's moving-block gradient bootstrap with the minimum-eigenvalue directional
derivative lambda_min(Q'KQ), Q estimated by a shrinking spectral window (see HDeriv).

Every row reuses the application's own loader, the paper-wide M, and the cached
LPT partition dimensions and block length, with 1,000 moving-block draws.
Runs no headline estimator. Writes
../output/figure1_hmgp.pdf and
the result cache's hmgp_figure1_data.csv.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from mgp_core import OUTPUT_DIR as OUT, align_sizes_to_panel, build_lpt_grouped_data, sample_cov
from gradient_bootstrap import _ci, moving_block_indices
from near_optimal import delta_rate
from compute_applications_coefficients import CACHE, clean_panel_moments, summary_path, available_items
from applications.figure_layout import HIERARCHY, hierarchy_positions, tick_labels
from applications.registry import panel_loaders, innovation_loaders
from applications.stella.paths import CALIBRATION
# Imported after mgp_core, which points MPLCONFIGDIR at a writable cache.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from applications.elections import run_election_applications as elec
from applications.mutual_funds import run_mutual_fund_applications as mutual
from applications.acemoglu.panel import DERIVED as ACEMOGLU_DERIVED
from applications.atalay import run_atalay_epsilon as atalay
from applications.carvalho_gabaix import run_carvalho_gabaix_proxy as cg
from applications.foerster import run_foerster_epsilon as foerster
from applications.herskovic import run_herskovic_firm_network as herskovic
import run_sector_output_baseline as baseline
from applications.stella import run_stella_firm_network as stella
from applications.welburn import welburn_network as welburn
B = 1000

class HDeriv:
    """Directional derivative of the grouped hMGP level c * lambda_min(A), A = L^{-1/2} Sigma L^{-1/2}.

    With Q spanning the eigenspace of the smallest eigenvalue, lambda_min'(A; K) = lambda_min(Q'KQ):
    linear (v'Kv) when the eigenvalue is simple, only directionally differentiable when it is
    repeated -- the natural case under an exact homoskedastic factor model with r < M, where the
    multiplicity is M - r.  Sample noise splits repeated roots almost surely, so Q is estimated by
    the spectral window {j : lambda_j - lambda_1 <= delta_T * lambda_1} with the paper's
    delta_T = sqrt(log T / T).  This is the near-optimal dual set of Appendix C.3 restricted to
    duals supported on the near-minimal eigenspace: it contains the population optimizer set with
    probability approaching one and shrinks onto it, and the SDP collapses to an eigenvalue.
    """

    def __init__(self, sigma, loads, t):
        self.tr = float(loads.sum())
        self.w = np.diag(1 / np.sqrt(loads))
        a = self.w @ sigma @ self.w
        vals, vecs = np.linalg.eigh(0.5 * (a + a.T))
        self.lam = float(vals[0]); self.gap = float(vals[1] - vals[0]) / self.lam
        self.level = self.lam * self.tr
        self.delta = delta_rate(t)
        window = vals - vals[0] <= self.delta * vals[0]
        self.q = vecs[:, window]; self.window = int(window.sum())

    def dlevel(self, h):
        k = self.w @ h @ self.w
        return self.tr * float(np.linalg.eigvalsh(self.q.T @ (0.5 * (k + k.T)) @ self.q)[0])


def hmgp(y, shares, m, block, seed, eps=None, network_v=None, weight_scale=1.0):
    """y: N x T (normalized shares); eps/network_v only for dynamic networks (T x N, N x N)."""
    y = np.asarray(y, float); shares = np.asarray(shares, float); t = y.shape[1]
    xg, _, loads, _ = build_lpt_grouped_data(y, shares, m)
    sigma = sample_cov(xg); dd = HDeriv(sigma, loads, t)
    if network_v is None:
        denom = float(sigma.sum()); denom_draw = lambda idx: float(sample_cov(xg[idx]).sum())
        level_scale = 1.0
    else:
        eps = np.asarray(eps, float)
        denom = float(np.sum(network_v * sample_cov(eps)))
        denom_draw = lambda idx: float(np.sum(network_v * sample_cov(eps[idx])))
        level_scale = weight_scale
    g = level_scale * dd.level / denom
    rng = np.random.default_rng(seed); stats = np.empty(B)
    for b in range(B):
        idx = moving_block_indices(t, block, rng)
        hg = np.sqrt(t) * (sample_cov(xg[idx]) - sigma)
        hd = np.sqrt(t) * (denom_draw(idx) - denom)
        stats[b] = (level_scale * dd.dlevel(hg) - g * hd) / denom
    lo, hi = _ci(g, stats, t)
    return dict(hmgp=g, h_lo=lo, h_hi=hi, eiggap=dd.gap, window=dd.window, N=y.shape[0], T=t, M=m, block=block, hhi=float(shares @ shares))

def core_panel(prefix, clean):
    s = pd.read_pickle(summary_path(prefix)).iloc[0]
    x, w, _, _ = clean_panel_moments(clean, s)
    return x, w, int(s.M), int(s.block_length)

def app_panel(app, summary_file):
    s = pd.read_pickle(CACHE / summary_file).set_index("Entity").loc[app.entity]
    y, sizes = align_sizes_to_panel(app.y, app.sizes)
    shares = (sizes / sizes.sum()).to_numpy(float)
    assert y.shape == (int(s.N), int(s["T"])), (app.entity, y.shape, s.N, s["T"])
    return app.relative_outcomes(y, sizes).to_numpy(float), shares, int(s.M), int(s.Block_length)

def nonnetwork_rows(seed, available):
    # ---- core ----
    for item, prefix, clean in [("Sales", "compustat_qtr", "compustat_quarterly_sales"), ("Employment", "compustat_ann_emp", "compustat_annual_emp"),
                                ("Productivity", "compustat_qtr_tfp", "compustat_quarterly_tfp"), ("Stocks", "crsp", "crsp_daily_returns"),
                                ("Forbes Billionaires", "forbes_full", "forbes_billionaire_wealth"),
                                ("Price by item", "cpi1967", "bls_cpi_1967_inflation"), ("Employment by sector", "bls_ces", "bls_ces_employment")]:
        if item in available:
            yield item, hmgp(*core_panel(prefix, clean), seed)
    # ---- mutual funds ----
    mutual_items = ("Mutual funds", "Mutual fund flows")
    if available.intersection(mutual_items):
        for item, app in zip(mutual_items, mutual.load_applications(), strict=True):
            if item in available:
                yield item, hmgp(*app_panel(app, "mutual_fund_applications_summary.pkl"), seed)
    # ---- additional and innovation ----
    for loaders in (panel_loaders(), innovation_loaders()):
        for item, loader in loaders.items():
            if item in available:
                yield item, hmgp(*app_panel(loader(), "additional_applications_summary.pkl"), seed)
    # ---- elections ----
    elections = {"Democratic votes by state": elec.state_application,
                 "Democratic votes by county": elec.county_application}
    if available.intersection(elections):
        data = elec.read_returns()
        for item, loader in elections.items():
            if item in available:
                app, _ = loader(data)
                yield item, hmgp(*app_panel(app, "election_applications_summary.pkl"), seed)

def network_rows(seed, available):
    # ---- sector networks ----
    if "No I/O network or lags" in available:
        tfp, w, _ = baseline.load_baseline()
        sb = pd.read_pickle(summary_path("bea_bls_sector_tfp_no_network")).iloc[0]
        yield "No I/O network or lags", hmgp(tfp.to_numpy(float), w / w.sum(), int(sb.M), int(sb.block_length), seed)  # share-scale invariant

    if "Foerster et al. (2011)" in available:
        f = foerster.load_foerster_epsilon_panel(foerster.DEFAULT_EPSILON_CSV, foerster.DEFAULT_WEIGHTS_CSV, foerster.DEFAULT_COLUMNS_CSV, 0)
        n = f.y.shape[0]
        mats = [foerster.read_network_matrix(p, n, nm) for p, nm in [(foerster.DEFAULT_PII_A_CSV, "Pi_a"), (foerster.DEFAULT_XI_CSV, "Xi"), (foerster.DEFAULT_CAP_RHO_CSV, "cap_rho")]]
        initial = f.sizes.reindex(f.y.index).astype(float)
        V = foerster.network_variance_matrix(initial.to_numpy(float), *mats)
        nsizes = pd.Series(np.sqrt(np.maximum(np.diag(V), 0)), index=f.y.index)
        y, sizes = align_sizes_to_panel(f.y, nsizes)
        order = [int(str(i).split("_")[-1]) - 1 for i in y.index]; Vs = V[np.ix_(order, order)]
        sf = pd.read_pickle(summary_path("foerster_net")).iloc[0]
        yield "Foerster et al. (2011)", hmgp(y.to_numpy(float), (sizes / sizes.sum()).to_numpy(float), int(sf.M), int(sf.block_length), seed,
                                            eps=y.to_numpy(float).T, network_v=Vs, weight_scale=float(sizes.sum() ** 2))

    if "Acemoglu et al. (2012)" in available:
        a_panel = pd.read_csv(ACEMOGLU_DERIVED / "bea1992_common_shocks.csv")
        prim = a_panel.pivot(index="component", columns="year", values="primitive_productivity_shock").sort_index()
        infl = a_panel.groupby("component")["fixed_influence_weight"].first().reindex(prim.index)
        sa = pd.read_pickle(summary_path("acemoglu2012_bea1992")).iloc[0]
        yield "Acemoglu et al. (2012)", hmgp(prim.to_numpy(float), (infl / infl.sum()).to_numpy(float), int(sa.M), int(sa.block_length), seed)

    if "Carvalho–Gabaix (2013)" in available:
        hist, _ = cg.load_historical_panel(); hist, _ = cg.invert_measured_tfp(hist, cg.Calibration())
        prim = hist.pivot(index="industry_code", columns="year", values="primitive_productivity_shock").sort_index()
        infl = hist.groupby("industry_code")["fixed_influence_weight"].first().reindex(prim.index)
        sc = pd.read_pickle(summary_path("carvalho_gabaix_bea_bls_historical")).iloc[0]
        yield "Carvalho–Gabaix (2013)", hmgp(prim.to_numpy(float), (infl / infl.sum()).to_numpy(float), int(sc.M), int(sc.block_length), seed)

    if "Atalay (2017)" in available:
        st = pd.read_pickle(summary_path("atalay_epsilon_network")).iloc[0]
        ap, V = atalay.load_atalay_panel(atalay.DEFAULT_SHOCKS_CSV, atalay.DEFAULT_NETWORK_V_CSV, int(st.M))
        y, sizes = align_sizes_to_panel(ap.y, ap.sizes)
        order = [int(str(i).split("_")[-1]) - 1 for i in y.index]; Vs = V[np.ix_(order, order)]
        yield "Atalay (2017)", hmgp(y.to_numpy(float), (sizes / sizes.sum()).to_numpy(float), int(st.M), int(st.block_length), seed,
                                   eps=y.to_numpy(float).T, network_v=Vs, weight_scale=float(sizes.sum() ** 2))

    if "Stella (2015)" in available:
        ids = stella.load_firm_ids(CALIBRATION)
        _, full_shocks, _, _, V = stella.load_dynamic_panel(CALIBRATION, ids)
        shocks = full_shocks[2:]
        nsz = np.sqrt(np.maximum(np.diag(V), 0))
        ss = pd.read_pickle(summary_path("stella_revtq_firm_network")).iloc[0]
        yield "Stella (2015)", hmgp(shocks.T, nsz / nsz.sum(), int(ss.M), int(ss.block_length), seed, eps=shocks, network_v=V, weight_scale=float(nsz.sum() ** 2))

    if "Herskovic et al. (2020)" in available:
        h = herskovic.build_herskovic_inputs(); sh = pd.read_pickle(summary_path("herskovic_firm_network")).iloc[0]
        yield "Herskovic et al. (2020)", hmgp(h.outcomes.T, h.shares, int(sh.M), int(sh.block_length), seed)

    if "Welburn et al. (2020)" in available:
        w = welburn.build_welburn_inputs()
        sw = pd.read_pickle(CACHE / "welburn_firm_network_M10_summary.pkl").iloc[0]
        yield "Welburn et al. (2020)", hmgp(
            w.outcomes.T, w.shares, int(sw.M), int(sw.block_length), seed,
        )

def rows():
    seed = 20260916
    available = available_items()
    yield from nonnetwork_rows(seed, available)
    yield from network_rows(seed, available)


OFF = 0.13  # symmetric vertical offset of the MGP (below) and hMGP (above) rows around the naive marker


def plot() -> None:
    paper = pd.read_csv(CACHE / "applications_coefficients_expanded_data.csv").set_index("item")
    h = pd.read_csv(CACHE / "hmgp_figure1_data.csv").set_index("item")

    positions, headings, ymax = hierarchy_positions()

    fig, ax = plt.subplots(figsize=(10, 12.5))
    for section, items in HIERARCHY:
        for item in items:
            p = positions[item]; r = paper.loc[item]
            if not np.isfinite(r.estimate): continue
            lo, hi = 100*r.low, 100*r.high
            if np.isfinite(lo) and np.isfinite(hi):
                ax.hlines(p - OFF, 0, max(0, lo), color="#B9D9EE", lw=2.0, zorder=2)
                ax.hlines(p - OFF, lo, hi, color="#1f77b4", lw=2.0, zorder=3)
            ax.plot(100*r.estimate, p - OFF, "o", color="#1f77b4", ms=5.0, zorder=4)
            if item in h.index:
                q = h.loc[item]
                if np.isfinite(q.h_lo) and np.isfinite(q.h_hi):
                    hlo, hhi = 100*max(0, q.h_lo), 100*min(1, q.h_hi)
                    ax.hlines(p + OFF, 0, hlo, color="#C7E9C0", lw=2.0, zorder=2)
                    ax.hlines(p + OFF, hlo, hhi, color="#2ca02c", lw=2.0, zorder=3)
                ax.plot(100*min(1, q.hmgp), p + OFF, "s", color="#2ca02c", ms=5.0, zorder=4)
            if np.isfinite(r.homoskedastic):
                ax.plot(min(100, 100*r.homoskedastic), p, "D", color="#d62728", ms=5.4, clip_on=False, zorder=5)
    ax.set_yticks(list(positions.values()), tick_labels(positions), fontsize=12)
    ax.tick_params(axis="y", length=0, pad=4)
    for lab in ax.get_yticklabels(): lab.set_horizontalalignment("right")
    for section, yp in headings:
        ax.text(-.012, yp, section, transform=ax.get_yaxis_transform(), ha="right", va="center", fontweight="bold", fontsize=12.2, clip_on=False)
    ax.set_xlim(0, 100); ax.set_xticks(range(0, 101, 20), [f"{x}%" for x in range(0, 101, 20)], fontsize=11.5); ax.set_xticks(range(5, 100, 5), minor=True)
    ax.set_ylim(ymax - .25, -.45); ax.set_xlabel("Granular share of aggregate fluctuations", fontsize=12); ax.grid(False)
    for x in range(5, 100, 5):
        c, w, ls = ("0.82", .72, "--") if x % 20 == 0 else ("0.88", .66, "--") if x % 10 == 0 else ("0.91", .62, ":")
        ax.axvline(x, color=c, linestyle=ls, linewidth=w, zorder=0)
    for sp in ax.spines.values(): sp.set_visible(True); sp.set_color("0.25"); sp.set_linewidth(.8)
    ax.legend(handles=[Line2D([0], [0], marker="D", color="#d62728", lw=0, markersize=5.4, label="Naive estimate"),
                       Line2D([0], [0], marker="o", color="#1f77b4", lw=2, markersize=5.0, label="Heteroskedastic ceiling"),
                       Line2D([0], [0], marker="s", color="#2ca02c", lw=2, markersize=5.0, label="Homoskedastic ceiling")],
              loc="upper right", frameon=True, framealpha=.95, edgecolor="0.75", facecolor="white", borderpad=.55, fontsize=11.5)
    fig.subplots_adjust(left=.215, right=.98, top=.988, bottom=.055)
    fig.savefig(OUT / "figure1_hmgp.pdf", dpi=190, facecolor="white", bbox_inches="tight", pad_inches=.03)
    plt.close(fig)
    tab = h.join(paper[["estimate", "high", "homoskedastic"]]).reindex([i for _, items in HIERARCHY for i in items])
    print(tab.assign(hmgp=lambda d: 100*d.hmgp, h_hi=lambda d: 100*d.h_hi, mgp=lambda d: 100*d.estimate, mgp_ucl=lambda d: 100*d.high, naive=lambda d: 100*d.homoskedastic)[["N", "T", "M", "hmgp", "h_hi", "mgp", "mgp_ucl", "naive", "eiggap", "window"]].round(1).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-plot", action="store_true", help="Update the result CSV only.")
    args = parser.parse_args()
    result_path = CACHE / "hmgp_figure1_data.csv"
    out = []
    for item, r in rows():
        r["item"] = item; out.append(r)
        interval = (
            f"90% [{100*r['h_lo']:6.2f}, {100*r['h_hi']:6.2f}]"
            if np.isfinite(r['h_lo']) and np.isfinite(r['h_hi']) else "point only"
        )
        print(f"[hmgp] {item:28s} N={r['N']:5d} T={r['T']:5d} M={r['M']:2d} block={r['block']:2d}  hMGP={100*r['hmgp']:6.2f}%  {interval}  eiggap={r['eiggap']:.2f} window={r['window']}", flush=True)
    columns = ("hmgp", "h_lo", "h_hi", "eiggap", "window", "N", "T", "M", "block", "hhi", "item")
    pd.DataFrame(out, columns=columns).to_csv(result_path, index=False)
    if not args.no_plot:
        plot()
        print(f"[hmgp] wrote {OUT / 'figure1_hmgp.pdf'}")
