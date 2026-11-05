"""FDIC bank loans, TIC Treasury holdings, and SDC corporate issuance."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mgp_core import remove_periodic_means
from wrds_inputs import ANALYSIS_LAST_QUARTER, wrds_path

from .common import RAW, Application, prepare_flows


FDIC = RAW / "fdic_quarterly_financials_1999q4_2025q4.csv"
TIC_CURRENT = RAW / "tic_slt_table3_2020_current.txt"
TIC_HISTORY = RAW / "tic_slt3d_2011_2019.csv"
SDC_ISSUES = wrds_path("sdc_new_issues")


def load_banks() -> Application:
    d = pd.read_csv(FDIC, dtype={"CERT": str, "REPDTE": str})
    d["period"] = pd.PeriodIndex(pd.to_datetime(d["REPDTE"]), freq="Q")
    loans = d.pivot(index="CERT", columns="period", values="LNLSNET")
    periods = pd.period_range("1999Q4", "2025Q4", freq="Q")
    missing = periods.difference(loans.columns)
    if len(missing):
        raise ValueError(f"Missing FDIC quarters: {list(missing)}; refresh the public-data input")
    loans = loans.reindex(columns=periods).apply(pd.to_numeric, errors="coerce")
    loans = loans.loc[(loans > 0).all(axis=1)]
    # Match quarterly Compustat sales: QoQ log growth, then residualize each
    # bank on its own quarter-of-year fixed effects before pooled winsorization.
    y = np.log(loans).diff(axis=1).iloc[:, 1:]
    y = remove_periodic_means(y, 4)
    labels = d.sort_values("period").drop_duplicates("CERT", keep="last").set_index("CERT")["NAME"]
    return Application(
        "Banks", "Net loans and leases growth", "FDIC BankFind Suite quarterly financials",
        "Buch & Neugebauer (2011) and Blanco-Arroyo & Alfarano (2022).",
        y, loans.loc[y.index, pd.Period("1999Q4")], labels, "1999Q4–2025Q4 levels; quarter-on-quarter growth 2000Q1–2025Q4", "Quarterly", "1999Q4",
        "Balanced FDIC-insured-bank panel; quarter-on-quarter log growth with bank-specific quarter-of-year fixed effects removed before pooled 1/99 winsorization. Mergers and exits imply survivor selection; fixed 1999Q4 net-loan weights.",
    )


def load_tic() -> Application:
    columns = ["country", "code", "date", "holdings"]
    current = pd.read_csv(
        TIC_CURRENT, sep="\t", skiprows=9, dtype=str,
        names=[*columns, "net", "lt", "ltnet", "val", "st", "stnet"],
    )
    historical = pd.read_csv(
        TIC_HISTORY, skiprows=12, names=[*columns, "lt", "st"],
        dtype=str, engine="python", on_bad_lines="skip",
    )
    d = pd.concat([historical[columns], current[columns]], ignore_index=True)
    d["code_num"] = pd.to_numeric(d.code, errors="coerce")
    d["holdings"] = pd.to_numeric(d.holdings.str.replace(",", ""), errors="coerce")
    d = d[d.code_num.notna() & d.date.str.match(r"^20\d\d-\d\d$", na=False)]
    d["period"] = pd.PeriodIndex(d.date, freq="M")
    periods = pd.period_range("2011-12", "2025-12", freq="M")
    d = d[d.period.between(periods[0], periods[-1])]
    levels = d.sort_values("period").drop_duplicates(["code", "period"], keep="last")
    levels = levels.pivot(index="code", columns="period", values="holdings").reindex(columns=periods)
    labels = d.drop_duplicates("code").set_index("code")["country"]
    # TIC geographic totals and memo groups also have codes below 70000.
    country = (levels.index.astype(int) < 70000) & ~labels.reindex(levels.index).str.match(
        r"^(?:Total |Memo:|All |Other |.*Oil Exporters$)"
    )
    world = levels.loc["99996"]
    # Retain the two disjoint published organization groups separately,
    # excluding their combined total (79995). Do not construct a residual.
    organization = levels.index.isin(["72907", "76929"])
    levels = levels.loc[(country | organization) & levels.gt(0).all(axis=1)]
    n_countries = int((levels.index.astype(int) < 70000).sum())
    coverage = levels.sum(axis=0) / world
    # Monthly holdings changes: no overlapping annual differences and no
    # seasonal adjustment. Application retains the existing pooled winsorization.
    y = np.log(levels).diff(axis=1).iloc[:, 1:]
    return Application("Foreign Treasury holders", "US Treasury-holdings growth", "US Treasury International Capital, Table 3",
        "Non-overlapping country/territory holdings and separately reported organization groups.", y, levels.loc[y.index, periods[0]], labels,
        "2011M12–2025M12 levels; month-on-month growth 2012M1–2025M12", "Monthly", "2011M12",
        f"{n_countries} balanced positive country/territory series and {len(levels)-n_countries} separately reported organization groups "
        "(International and Total Regional Orgs.). Geographic totals, memo groups, and Total IROs are excluded; no residual is added. "
        f"Retained holdings cover {coverage.min():.2%}–{coverage.max():.2%} of TIC Grand Total across the sample. "
        "Month-on-month log growth without seasonal adjustment, with pooled 1/99 winsorization and fixed December 2011 holdings weights. "
        "Fixed-share winsorized growth is an approximation to world holdings growth. Holdings are custodial and may not identify ultimate owners.")


def load_bond_issues() -> Application:
    """Quarterly gross US-market USD bond issuance by ultimate parent.

    SDC's total deal amount is recorded in USD millions. Selecting the main
    tranche makes MASTER_DEAL_NO unique and prevents a multi-tranche deal total
    from being counted repeatedly. The security filter excludes
    convertibles, structured/asset-backed securities, government and municipal
    obligations, equity-linked securities, CDs, and deposit/bank notes.
    """
    use = ["MASTER_DEAL_NO", "MASTER_DEAL_DATE", "NINAMES", "CURRENCY", "MARKET", "SECURITY",
           "TOTDOLAMT", "UPCUSIP", "MAIN_TRANCHE", "ASSETTYPE_PRINT", "NATION", "INDUSTRY"]
    d = pd.read_csv(SDC_ISSUES, usecols=[c.lower() for c in use], low_memory=False)
    d.columns = d.columns.str.upper()
    security = d["SECURITY"].fillna("")
    debt = security.str.contains(
        r"Bond|Bonds|\bBd\b|\bBds\b|Note|Notes|\bNt\b|\bNts\b|Debenture|Debs|Debentures|MTN|Medium-Term",
        case=False, regex=True)
    excluded = security.str.contains(
        r"Convert|Cvt|Exchangeable|\bExch\b|Government|Municip|Muni |Treasury|Sovereign|"
        r"Asset|Mtg|Mortgage|Pass|Collat|CLO|CDO|Structured|Struc|Islamic|Sukuk|"
        r"Preferred|Pfd|Deposit|CD Notes|CD Nts|Equity|Shares|"
        r"Units|Warrant|Rights|Tax Rev|Revenue Bond|Rev Bonds|Ind Dev|Gen Obl|Housing Bd",
        case=False, regex=True)
    keep = (
        d["CURRENCY"].eq("US")
        & d["MARKET"].isin(["U.S. Public", "U.S. Private"])
        & d["MAIN_TRANCHE"].eq("Y")
        & d["ASSETTYPE_PRINT"].isna()
        & d["NATION"].eq("United States")
        & ~d["INDUSTRY"].isin({
            "Fedl Credit Agcy", "Mtg Securities", "National Agency", "Regional Gov't",
            "Non-Gov't Agency", "National Gov't", "City Government", "Regional Agency", "City Agency",
        })
        & debt & ~excluded
    )
    d = d.loc[keep].copy()
    d["date"] = pd.to_datetime(d["MASTER_DEAL_DATE"], errors="coerce")
    d["amount"] = pd.to_numeric(d["TOTDOLAMT"], errors="coerce")
    d = d[d["date"].notna() & d["amount"].gt(0) & d["UPCUSIP"].notna()]
    d = d[d["date"].dt.year.between(1990, 2025)]
    if d["MASTER_DEAL_NO"].duplicated().any():
        raise ValueError("SDC main-tranche filter did not produce unique master deals")
    d["period"] = d["date"].dt.to_period("Q")
    periods = pd.period_range("1990Q1", ANALYSIS_LAST_QUARTER, freq="Q")
    flows = d.pivot_table(index="UPCUSIP", columns="period", values="amount", aggfunc="sum")
    flows, mean_flow = prepare_flows(flows.reindex(columns=periods))
    y = remove_periodic_means(flows, 4)
    labels = (d.groupby("UPCUSIP")["NINAMES"].agg(
        lambda z: z.mode().iat[0] if not z.mode().empty else z.iloc[0]))
    return Application(
        "US corporate bond issuers", "Gross USD bond issuance", "LSEG SDC New Issues via WRDS",
        "Issuer-level gross bond flows provide a capital-market analogue to the granular firm and bank applications.",
        y, mean_flow, labels, "1990Q1–2025Q4 issue dates", "Quarterly", "Full-sample mean issuance",
        "USD millions in levels, aggregated by ultimate-parent CUSIP; zero-issuance quarters retained. "
        "Issuer-specific quarter-of-year means are removed without linear detrending or winsorization. "
        "Adjusted flows divided by fixed mean raw issuance are grouped and aggregated with those same mean-issuance shares, preserving the adjusted total up to a common scale. "
        "US-domiciled corporate issuers in US public and private markets, USD issues, and main tranches only. "
        "Federal credit, mortgage-security, government, and agency issuer industries are excluded, as are convertibles, "
        "structured/asset-backed, municipal, equity-linked, CD, and deposit-note securities. Licensed WRDS data.",
        aggregation="sum",
    )
