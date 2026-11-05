#!/usr/bin/env python3
"""Survivor-Bias-Free U.S. Mutual Funds: monthly share-class returns, all history/types.
Retain inactive funds and missing returns; no portfolio aggregation or sample screens.
Default observation cutoff: end-2025.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _wrds import download

SQL = """SELECT crsp_fundno,caldt,mret
 FROM crsp_q_mutualfunds.monthly_returns WHERE caldt <= %(end_date)s"""


if __name__ == "__main__":
    download(__file__, "mutual_fund_monthly_returns", SQL)
