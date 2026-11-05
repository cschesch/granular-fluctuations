#!/usr/bin/env python3
"""North America quarterly sales, 1961 onward; standard consolidated industrial accounts.
Domestic-source population (D), USD accounts, all countries and active/inactive firms.
Retain missing/nonpositive sales and fiscal/calendar labels. Company fields are current.
Country, industry and balanced-panel screens belong in the analysis.
The cutoff uses datacqtr, retaining fiscal periods assigned to 2025Q4 even if
their period-end date falls in January 2026. Unlabelled quarters are omitted.
Default observation cutoff: end-2025.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _wrds import download

SQL = """-- Compustat North America: quarterly sales and explicit quarter labels.
-- Active and inactive companies; standard consolidated industrial statements.
-- USD reporting currency; retain all countries, industries and nonpositive sales.
-- Company-table names/classifications/country fields are current attributes.
SELECT
    q.gvkey, q.datadate, q.datacqtr, q.datafqtr,
    q.fyearq, q.fqtr, q.fyr, q.rdq,
    q.indfmt, q.consol, q.popsrc, q.datafmt, q.curcdq,
    q.saleq, q.saley,
    c.conm, c.conml, c.costat, c.fic, c.loc, c.naics, c.sic
FROM comp.fundq AS q
LEFT JOIN comp.company AS c ON q.gvkey = c.gvkey
WHERE q.indfmt = 'INDL'
  AND q.consol = 'C'
  AND q.datafmt = 'STD'
  AND q.popsrc = 'D'
  AND q.curcdq = 'USD'
  AND q.datadate >= %(start_date)s
  AND q.datacqtr <= %(end_quarter)s
ORDER BY q.gvkey, q.datadate;
"""


if __name__ == "__main__":
    download(__file__, "compustat_quarterly_sales", SQL, start_date="1961-01-01")
