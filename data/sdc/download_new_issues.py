#!/usr/bin/env python3
"""Global SDC New Issues records, all available dates, currencies and security types.
Retain all statuses and tranches. U.S./USD corporate-debt and issuance screens
belong in analysis; do not repeatedly sum a deal total across its tranches.
Default observation cutoff: end-2025.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _wrds import download

SQL = """SELECT * FROM sdc.wrds_ni_details
 WHERE master_deal_date <= %(end_date)s"""


if __name__ == "__main__":
    download(__file__, "sdc_new_issues", SQL)
