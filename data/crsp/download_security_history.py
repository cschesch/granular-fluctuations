#!/usr/bin/env python3
"""CRSP CIZ security identifiers and classifications with their validity intervals.
Use the quarterly stock release; retain all securities and source history.
Join by permno and the applicable date interval when constructing samples.
Default observation cutoff: end-2025.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _wrds import download

SQL = """SELECT * FROM crsp_q_stock.stksecurityinfohist
 WHERE secinfostartdt <= %(end_date)s"""


if __name__ == "__main__":
    download(__file__, "crsp_security_history_ciz", SQL)
