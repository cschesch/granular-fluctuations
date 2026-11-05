"""Shared ordering, labels and positions of the paper's application figures."""

HIERARCHY = [
    ("Firms", ["Sales", "Employment", "Productivity"]),
    ("Returns", ["Stocks", "Mutual funds", "Forbes Billionaires"]),
    ("Financial flows", ["Bank loans", "Corporate bond issuance", "Mutual fund flows", "Treasury demand by country"]),
    ("Trade", ["Export products", "Import products", "Export destinations", "Import origins"]),
    ("Prices and labor", ["Price by item", "Employment by sector"]),
    ("Innovation", ["Patent count by firm", "Patent value by firm", "Patent count by field", "Patent value by field",
                    "Drug approvals by firm"]),
    ("Geography", ["GDP by metro", "GDP by county", "GDP by state"]),
    ("Elections", ["Democratic votes by state", "Democratic votes by county"]),
    ("Energy and climate", ["Power generation by plant", "CO2 emissions by plant", "Oil production by country", "CO2 emissions by country"]),
    ("World economy", ["GDP (PPP) by country", "Mkt cap by country", "Public debt by country", "Private debt by country"]),
    ("Sector networks", ["No I/O network or lags", "Foerster et al. (2011)",
                       "Acemoglu et al. (2012)", "Carvalho–Gabaix (2013)", "Atalay (2017)"]),
    ("Firm networks", ["Stella (2015)", "Herskovic et al. (2020)",
                       "Welburn et al. (2020)"]),
]


# Asterisks mark reconstructions of unavailable empirical inputs.
APPROXIMATE_REPLICATIONS = {
    "Acemoglu et al. (2012)",
    "Carvalho–Gabaix (2013)",
    "Stella (2015)",
    "Herskovic et al. (2020)",
    "Welburn et al. (2020)",
}


def hierarchy_positions():
    positions, headings = {}, []
    y = 0.0
    for section, items in HIERARCHY:
        headings.append((section, y))
        y += .72
        for item in items:
            positions[item] = y
            y += .72
        y += .48
    return positions, headings, y


def tick_labels(positions):
    return [item + ("*" if item in APPROXIMATE_REPLICATIONS else "") for item in positions]
