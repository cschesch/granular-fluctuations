"""Corporate bond issuance from licensed SDC inputs."""

import pandas as pd
from mgp_core import remove_periodic_means
from wrds_inputs import ANALYSIS_LAST_QUARTER, wrds_path
from applications.common import Application, prepare_flows


SDC_ISSUES = wrds_path("sdc_new_issues")


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
