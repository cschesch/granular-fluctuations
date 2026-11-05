#!/usr/bin/env python3
"""Survivor-Bias-Free U.S. Mutual Funds: monthly share-class net assets, all history/types.
Retain missing/nonpositive assets; no portfolio aggregation or weight construction.
Default observation cutoff: end-2025.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _wrds import download

SQL = """SELECT crsp_fundno,caldt,mtna
 FROM crsp_q_mutualfunds.monthly_tna WHERE caldt <= %(end_date)s"""


if __name__ == "__main__":
    download(__file__, "mutual_fund_monthly_assets", SQL)
