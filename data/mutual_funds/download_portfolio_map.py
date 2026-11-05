#!/usr/bin/env python3
"""All dated mutual-fund share-class (crsp_fundno) to portfolio (crsp_portno) links.
Retain validity intervals; no share-class aggregation in this download.
Default observation cutoff: end-2025.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _wrds import download

SQL = """SELECT * FROM crsp_q_mutualfunds.crsp_portno_map
 WHERE begdt <= %(end_date)s"""


if __name__ == "__main__":
    download(__file__, "mutual_fund_portfolio_map", SQL)
