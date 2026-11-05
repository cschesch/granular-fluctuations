#!/usr/bin/env python3
"""North America annual accounts; standard consolidated industrial statements (INDL/C/STD/D).
USD accounts, all available years, countries, industries and active/inactive firms.
Retain missing/nonpositive values. sich/naicsh are historical; company fields are current.
Sales, employment and network-application variables share this single source extract.
The cutoff uses fiscal year, including fiscal-2025 accounts ending in early 2026.
Default observation cutoff: end-2025.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _wrds import download

SQL = """SELECT
 a.gvkey,a.datadate,a.fyear,a.fyr,a.indfmt,a.consol,a.popsrc,a.datafmt,a.curcd,
 a.tic,a.cusip,a.sale,a.revt,a.emp,a.at,a.cogs,a.xlr,a.xsga,a.xrd,a.xint,
 a.oibdp,a.oiadp,a.ebit,a.ebitda,a.ni,a.ib,a.dp,a.capx,a.ppent,a.invt,
 a.che,a.act,a.lct,a.dlc,a.dltt,a.lt,a.seq,a.ceq,a.csho,a.prcc_f,
 a.sich,a.naicsh,c.conm,c.conml,c.costat,c.fic,c.loc,c.sic,c.naics,
 c.cik,c.ipodate,c.dldte,c.dlrsn,c.gsector,c.ggroup,c.gind,c.gsubind
 FROM comp.funda a LEFT JOIN comp.company c ON a.gvkey=c.gvkey
 WHERE a.indfmt='INDL' AND a.consol='C' AND a.datafmt='STD' AND a.popsrc='D'
 AND a.curcd='USD'
 AND a.fyear <= %(end_year)s"""


if __name__ == "__main__":
    download(__file__, "compustat_annual", SQL)
