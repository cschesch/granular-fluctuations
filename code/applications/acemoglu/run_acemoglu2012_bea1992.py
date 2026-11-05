#!/usr/bin/env python3
"""Apply the Acemoglu et al. 1992 network influences to matched NBER–CES TFP."""

import argparse
import re
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from gradient_bootstrap import bootstrap_family
from mgp_core import (
    ANALYSIS_BOOTSTRAP_DRAWS,
    RUN_OUTPUT_CACHE,
    analysis_block_length,
    build_lpt_grouped_data,
    choose_solver,
    make_panel_rng,
    mgp_share,
)
from applications.acemoglu.run_acemoglu2012_mgp import (
    DATA_ROOT,
    DERIVED,
    DETAILED_IO_MAT,
    EXPECTED_DETAILED_NORMS,
    EXPECTED_HEADERS,
    NBER_MAT,
    load_nber_manufacturing_panel,
)


BEA_ARCHIVE = DATA_ROOT / "bea1992" / "ndn0184.zip"

# BEA 1992 benchmark documentation, p. M–43, identifies commodities
# reassigned to the groups of their primary producers.
REDEFINED_COMMODITIES = {
    "180201": ("180400", "190200"),
    "180202": ("180400", "190200"),
    "180203": ("180400", "190200"),
    "270202": ("270201",),
    "370104": ("370101",),
    "370105": ("370101",),
    "380600": ("380100", "380400", "380501"),
    "381200": ("381100",),
    "381300": ("381100",),
}


def parse_sic_concordance(text: str, sic_codes: pd.Index) -> pd.DataFrame:
    manufacturing = text.split("\nMANUFACTURING\n", 1)[1].split(
        "\nTRANSPORTATION, COMMUNICATIONS, AND UTILITIES\n", 1
    )[0]
    entries = []
    for line in manufacturing.splitlines():
        if re.match(r"^\d{2}\.\d{4}\s", line):
            entries.append(line.strip())
        elif line.strip() and line.startswith(" "):
            entries[-1] += " " + line.strip()
    token = r"\*?\d{3,4}(?:-\d{1,4})?"
    rows = []
    for entry in entries:
        match = re.fullmatch(rf"(\d{{2}}\.\d{{4}})\s+(.+?)\s+({token}(?:,\s*{token})*)", entry)
        if match is None:
            raise ValueError(f"Cannot parse BEA concordance row {entry!r}")
        code, description, sic_tokens = match.groups()
        for value in sic_tokens.split(","):
            start, _, end = value.strip().lstrip("*").partition("-")
            stop = start[:len(start) - len(end)] + end if end else start
            for sic in sic_codes:
                if int(start) <= int(str(sic)[:len(start)]) <= int(stop):
                    rows.append((sic, code.replace(".", ""), description))
    pairs = pd.DataFrame(rows, columns=["sic", "original_io_code", "original_description"])
    np.testing.assert_array_equal(np.sort(pairs.sic), np.sort(sic_codes))
    return pairs


def match_influences(sic_codes: pd.Index) -> tuple[pd.DataFrame, pd.DataFrame]:
    with ZipFile(BEA_ARCHIVE) as archive:
        totals_long = pd.read_csv(
            archive.open("CXCTR.TXT"), sep="\t", header=None,
            names=["supplier", "buyer", "table", "requirement"],
            dtype={"supplier": str, "buyer": str},
        )
        descriptions = pd.read_csv(
            archive.open("io-code.txt"), sep="\t", header=None,
            names=["io_code", "description"], dtype={"io_code": str}, encoding="latin1",
        )
        pairs = parse_sic_concordance(
            archive.read("Sic-IO.txt").decode("latin1").replace("\r\n", "\n"), sic_codes,
        )
    total_requirements = totals_long.pivot(index="supplier", columns="buyer", values="requirement")
    if not total_requirements.index.equals(total_requirements.columns):
        raise ValueError("BEA total requirements must share the same supplier and buyer codes.")
    network = loadmat(DETAILED_IO_MAT)["CC92"]
    bea = total_requirements.iloc[:len(network), :len(network)]
    direct = np.eye(len(network)) - np.linalg.inv(bea)
    # A cutoff just below 1e-6 falls between the largest omitted positive
    # coefficient and the smallest retained one in the deposit.
    labeled = np.where(direct >= 9.99e-7, direct, 0)
    np.testing.assert_allclose(labeled, network, rtol=0, atol=2e-12)
    print(f"BEA direct requirements vs. CC92: max difference {np.abs(labeled - network).max():.3g}", flush=True)
    totals = network.sum(axis=0)
    normalized = network / np.where(totals > 0, totals, 1)
    alpha = 0.3333
    influence = np.linalg.solve(
        np.eye(len(network)) - (1 - alpha) * normalized,
        np.full(len(network), alpha / len(network)),
    )
    np.testing.assert_allclose(np.linalg.norm(influence), EXPECTED_DETAILED_NORMS[4], rtol=1e-12)
    nodes = pd.DataFrame({"io_code": bea.index, "position": np.arange(1, len(network) + 1), "bea_influence": influence})
    nodes = nodes.merge(descriptions, on="io_code", validate="one_to_one")
    pairs["io_code"] = pairs.original_io_code.map(lambda code: REDEFINED_COMMODITIES.get(code, (code,)))
    links = pairs.explode("io_code").merge(nodes, on="io_code", how="left", validate="many_to_one")
    if links.bea_influence.isna().any():
        raise ValueError(f"Unmatched BEA codes: {links.loc[links.bea_influence.isna(), 'io_code'].unique()}")

    # Connected components form a common classification with no SIC industry
    # or BEA commodity appearing in two units.
    sic_idx, sic_codes = pd.factorize(links.sic, sort=True)
    io_idx, io_codes = pd.factorize(links.io_code, sort=True)
    n = len(sic_codes) + len(io_codes)
    graph = coo_matrix((np.ones(len(links)), (sic_idx, len(sic_codes) + io_idx)), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    sic_groups = pd.Series(labels[:len(sic_codes)], index=sic_codes)
    links["component"] = links.sic.map(sic_groups)
    return links, nodes


def estimate(shocks, weights, components, m, b, solver):
    contributions = shocks.mul(weights, axis=0).groupby(components).sum()
    sizes = weights.groupby(components).sum()
    series = contributions.div(sizes, axis=0)
    grouped, *_ = build_lpt_grouped_data(series.to_numpy(), (sizes / sizes.sum()).to_numpy(), m)
    point = mgp_share(grouped, solver)["mgp_share"]
    block = analysis_block_length(shocks.shape[1])
    # Use the same moving blocks across specifications.
    inference = bootstrap_family(
        grouped, b=b, block_length=block, solver=solver,
        rng=make_panel_rng(20260831, f"acemoglu2012_nber_manufacturing_M{m}_gradient"),
    )
    return {
        "sic_industries": len(shocks),
        "units": len(sizes),
        "years": shocks.shape[1],
        "groups": m,
        "weight_hhi": float(((sizes / sizes.sum()) ** 2).sum()),
        "ceiling": float(point),
        "lcl05": float(inference["block_gradient_low"]),
        "ucl95": float(inference["block_gradient_high"]),
        "aggregate_variance": float(np.var(grouped.sum(axis=1), ddof=1)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--M", type=int, default=10)
    parser.add_argument("--bootstrap-b", type=int, default=ANALYSIS_BOOTSTRAP_DRAWS)
    args = parser.parse_args()
    panel, _ = load_nber_manufacturing_panel()
    shocks = panel.pivot(index="sic", columns="year", values="primitive_productivity_shock")
    raw = pd.DataFrame(loadmat(NBER_MAT)["data"], columns=EXPECTED_HEADERS)
    raw["SIC"] = raw.SIC.astype(int)
    shipments = raw.pivot(index="SIC", columns="YEAR", values="VSHIP")
    links, nodes = match_influences(shipments.index)
    links["in_balanced_panel"] = links.sic.isin(shocks.index)
    components = links.groupby("sic").component.first().reindex(shocks.index)
    if components.isna().any():
        raise ValueError("Some balanced SIC industries have no common unit.")
    group_influence = links.drop_duplicates("io_code").groupby("component").bea_influence.sum()
    shipments58 = shipments.loc[shocks.index, 1958]
    shipments92 = shipments.loc[shocks.index, 1992]
    if not shipments92.gt(0).all():
        raise ValueError("Balanced SIC industries require positive 1992 shipments.")
    # Compute component TFP with fixed 1992 shipment weights. Require each
    # component to have complete TFP coverage or to be omitted in its entirety.
    if set(components).intersection(links.loc[~links.in_balanced_panel, "component"]):
        raise ValueError("A matched common unit contains a SIC with incomplete TFP.")
    within = shipments92 / shipments92.groupby(components).transform("sum")
    network_weights = within * components.map(group_influence)
    common_shipments58 = within * shipments58.groupby(components).transform("sum")
    links["used_component_influence"] = links.component.map(group_influence).where(links.in_balanced_panel)
    mass = network_weights.sum()
    print(
        f"{len(links)} concordance pairs, {links.io_code.nunique()} BEA nodes, "
        f"{links.sic.nunique()} SICs, {len(shocks)} balanced SICs, "
        f"{components.nunique()} matched units, BEA mass {mass:.6f} "
        f"of {nodes.bea_influence.sum():.6f} in the balanced sample.",
        flush=True,
    )
    solver = choose_solver("auto")
    cases = [
        ("1958 shipments, original", shocks, shipments58, shocks.index),
        ("1958 shipments, common units", shocks, common_shipments58, components),
        ("1992 shipments, common units", shocks, shipments92, components),
        ("1992 BEA influence, common units", shocks, network_weights, components),
        ("1992 BEA influence, SIC units", shocks, network_weights, shocks.index),
    ]
    rows = []
    for name, values, weights, units in cases:
        row = estimate(values, weights, units, args.M, args.bootstrap_b, solver)
        rows.append({"specification": name, **row})
    results = pd.DataFrame(rows)
    DERIVED.mkdir(parents=True, exist_ok=True)
    nodes.to_csv(DERIVED / "bea1992_network_nodes.csv", index=False)
    links.to_csv(DERIVED / "bea1992_sic_io_crosswalk.csv", index=False)
    pd.DataFrame({"component": components, "within_component_share": within, "influence": network_weights}).to_csv(
        DERIVED / "bea1992_sic_influence_allocation.csv"
    )
    results.to_csv(DERIVED / "bea1992_granularity_comparison.csv", index=False)
    sizes = network_weights.groupby(components).sum()
    common = shocks.mul(within, axis=0).groupby(components).sum()
    common = common.rename_axis(columns="year").stack().rename("primitive_productivity_shock").reset_index()
    common["fixed_influence_weight"] = common.component.map(sizes)
    common.to_csv(DERIVED / "bea1992_common_shocks.csv", index=False)
    primary = results.set_index("specification").loc["1992 BEA influence, common units"]
    RUN_OUTPUT_CACHE.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(
        N=len(sizes), T=shocks.shape[1], M=args.M, HHI=primary.weight_hhi,
        start_date=str(shocks.columns.min()), last_date=str(shocks.columns.max()),
        frequency="annual", block_length=analysis_block_length(shocks.shape[1]),
        aggregate_variance=primary.aggregate_variance, mgp_share=primary.ceiling,
        moving_block_gradient_bootstrap_p05=primary.lcl05,
        moving_block_gradient_bootstrap_p95=primary.ucl95,
    )]).to_pickle(RUN_OUTPUT_CACHE / f"acemoglu2012_bea1992_M{args.M}_summary.pkl")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
