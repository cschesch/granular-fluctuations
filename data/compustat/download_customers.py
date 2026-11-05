#!/usr/bin/env python3
"""Historical Segments customer disclosures, all source dates and customer types.
Use comp_segments_hist_daily for full history, not the narrower comp.seg_customer.
Retain disclosed sales and customer identifiers; these are not automatically buyer gvkeys.
Supplier company attributes are current; no matching or network construction here.
Default observation cutoff: end-2025.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _wrds import download

SQL = """SELECT s.gvkey,s.srcdate,
 s.cid,s.cnms,s.ctype,s.gareac,s.gareat,s.salecs,s.sid,s.stype,
 c.conm,c.conml,c.cik,c.sic,c.naics,c.gsector,c.ggroup,c.gind,c.gsubind,c.fic,c.loc,c.costat
 FROM comp_segments_hist_daily.wrds_seg_customer s LEFT JOIN comp.company c ON s.gvkey=c.gvkey
 WHERE s.srcdate <= %(end_date)s"""


if __name__ == "__main__":
    download(__file__, "compustat_customers", SQL)
