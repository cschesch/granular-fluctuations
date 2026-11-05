#!/usr/bin/env python3
"""Daily stocks from the quarterly CRSP CIZ release, all history and security types.
Include delisted securities; retain native return/price flags and units.
No common-stock screen, SIZ conversion or additional delisting adjustment here.
Dated security classifications are in download_security_history.py.
Default observation cutoff: end-2025.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _wrds import download

SQL = """SELECT permno,permco,dlycaldt,dlyret,dlyretx,
 dlyprc,shrout,dlycap,dlyvol,dlydelflg,dlyretmissflg,dlyprcflg,dlycapflg,
 dlycumfacpr,dlycumfacshr FROM crsp_q_stock.dsf_v2 WHERE dlycaldt <= %(end_date)s"""


if __name__ == "__main__":
    download(__file__, "crsp_daily_stocks_ciz", SQL)
