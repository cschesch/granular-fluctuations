"""Shared ordering, labels and positions of the paper's application figures."""

HIERARCHY = [
    ("Firms", ["Sales", "Employment", "Productivity"]),
    ("Returns", ["Stocks", "Mutual funds", "Forbes Billionaires"]),
    ("Financial flows", ["Bank loans", "Corporate bond issuance", "Mutual fund flows", "Treasury demand by country"]),
    ("Prices and labor", ["Price by item", "Employment by sector"]),
    ("Trade", ["Export products", "Import products", "Export destinations", "Import origins"]),
    ("Innovation", ["Patent count by firm", "Patent value by firm", "Patent count by field", "Patent value by field",
                    "Drug approvals by firm"]),
    ("Geography", ["GDP by state", "GDP by county", "GDP by metro"]),
    ("Elections", ["Democratic votes by state", "Democratic votes by county"]),
    ("Energy and climate", ["Power generation by plant", "CO2 emissions by plant", "CO2 emissions by country", "Oil production by country"]),
    ("World economy", ["GDP (PPP) by country", "Public debt by country", "Private debt by country", "Mkt cap by country"]),
    ("Sector networks", ["No I/O network or lags", "Carvalho–Gabaix (2013)",
                       "Acemoglu et al. (2012)", "Foerster et al. (2011)", "Atalay (2017)"]),
    ("Firm networks", ["Stella (2015)", "Welburn et al. (2020)",
                       "Herskovic et al. (2020)"]),
]


# Asterisks mark reconstructions of unavailable empirical inputs.
APPROXIMATE_REPLICATIONS = {
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


def format_application_rows(ax, positions, headings, ymax, *, fontsize=12, heading_x=-.012):
    """Use the same row labels, section headings and frame in application figures."""
    ax.set_yticks(list(positions.values()), tick_labels(positions), fontsize=fontsize)
    ax.tick_params(axis="y", length=0, pad=4)
    for label in ax.get_yticklabels():
        label.set_horizontalalignment("right")
    for section, y in headings:
        ax.text(heading_x, y, section, transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontweight="bold", fontsize=12.2, clip_on=False)
    ax.set_ylim(ymax - .25, -.45)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("0.25")
        spine.set_linewidth(.8)


def format_share_axis(ax):
    """Percentage axis shared by the heteroskedastic and homoskedastic figures."""
    ax.set_xlim(0, 100)
    ax.set_xticks(range(0, 101, 20), [f"{x}%" for x in range(0, 101, 20)], fontsize=11.5)
    ax.set_xticks(range(5, 100, 5), minor=True)
    ax.set_xlabel("Granular share of aggregate fluctuations", fontsize=12)
    ax.grid(False)
    for x in range(5, 100, 5):
        if x % 20 == 0:
            color, linewidth, linestyle = "0.82", .72, "--"
        elif x % 10 == 0:
            color, linewidth, linestyle = "0.88", .66, "--"
        else:
            color, linewidth, linestyle = "0.91", .62, ":"
        ax.axvline(x, color=color, linestyle=linestyle, linewidth=linewidth, zorder=0)
