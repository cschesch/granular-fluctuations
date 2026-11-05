"""The paper's independent panel applications, labels and source loaders."""

from dataclasses import dataclass
from functools import partial
from typing import Callable

from applications.common import Application
from applications.debt.panel import load_private_debt, load_public_debt
from applications.fdic.panel import load_banks
from applications.sdc.panel import load_bond_issues
from applications.treasury.panel import load_tic
from applications.granular_cities.panel import load_cities
from applications.bea_regional.panel import load_counties, load_states
from applications.patent_firms.panel import load_patents
from applications.epa.panel import load_campd
from applications.carbon_emissions.panel import load_country_co2
from applications.owid_energy.panel import load_oil
from applications.comtrade.panel import load_trade
from applications.maddison.panel import load_countries
from applications.french.panel import load_country_equity_returns
from applications.kpss.panel import patent_value_by_field, patent_count_by_field, patent_value_by_firm
from applications.fda.panel import novel_drugs_by_sponsor


@dataclass(frozen=True)
class PanelSpec:
    label: str
    entity: str
    load: Callable[[], Application]


PANELS = (
    PanelSpec('GDP (PPP) by country', 'Countries', load_countries),
    PanelSpec('Mkt cap by country', 'Country equity markets', load_country_equity_returns),
    PanelSpec('Bank loans', 'Banks', load_banks),
    PanelSpec('GDP by metro', 'Metropolitan areas', load_cities),
    PanelSpec('Export products', 'Exported products', partial(load_trade, "exports", "hs2")),
    PanelSpec('Import products', 'Imported products', partial(load_trade, "imports", "hs2")),
    PanelSpec('Export destinations', 'Export destinations', partial(load_trade, "exports", "partners")),
    PanelSpec('Import origins', 'Import origins', partial(load_trade, "imports", "partners")),
    PanelSpec('GDP by state', 'US states', load_states),
    PanelSpec('GDP by county', 'US counties', load_counties),
    PanelSpec('Power generation by plant', 'Power plants', partial(load_campd, "Gross Load (MWh)", "Power plants", "Gross-load growth")),
    PanelSpec('CO2 emissions by plant', 'Power-plant emissions', partial(load_campd, "CO2 Mass (short tons)", "Power-plant emissions", "CO2-emissions growth")),
    PanelSpec('Patent count by firm', 'Patent assignees', load_patents),
    PanelSpec('Oil production by country', 'Oil-producing countries', load_oil),
    PanelSpec('Public debt by country', 'Public debt by country', load_public_debt),
    PanelSpec('Private debt by country', 'Private nonfinancial debt by country', load_private_debt),
    PanelSpec('Treasury demand by country', 'Foreign Treasury holders', load_tic),
    PanelSpec('CO2 emissions by country', 'Country CO2 emissions', load_country_co2),
    PanelSpec('Corporate bond issuance', 'US corporate bond issuers', load_bond_issues),
)

INNOVATION = (
    PanelSpec('Patent value by firm', 'Patent-value firms', patent_value_by_firm),
    PanelSpec('Patent count by field', 'Patent-count fields', patent_count_by_field),
    PanelSpec('Patent value by field', 'Patent technology fields', patent_value_by_field),
    PanelSpec('Drug approvals by firm', 'Novel-drug sponsors', novel_drugs_by_sponsor),
)


PANEL_ENTITIES = {spec.label: spec.entity for spec in (*PANELS, *INNOVATION)}
PANEL_ENTITIES.update({
    "Mutual funds": "Mutual fund portfolios: returns",
    "Mutual fund flows": "Mutual fund portfolios: flows",
})


def panel_loaders():
    return {spec.label: spec.load for spec in PANELS}


def innovation_loaders():
    return {spec.label: spec.load for spec in INNOVATION}
