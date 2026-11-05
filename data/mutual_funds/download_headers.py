#!/usr/bin/env python3
"""Current mutual-fund identifiers and attributes, including inactive share classes.
Do not use today's portfolio identifiers as historical mappings; see header_history
and portfolio_map downloads for dated links.
Default observation cutoff: end-2025.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _wrds import download

SQL = """SELECT * FROM crsp_q_mutualfunds.fund_hdr"""


if __name__ == "__main__":
    download(__file__, "mutual_fund_headers", SQL)
