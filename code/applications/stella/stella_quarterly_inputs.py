"""Stella's current-vintage revtq input, separate from project-wide saleq panels.

Legacy reconstruction variable/file names say saleq. They receive revtq values
through this explicit adapter, with no filling from saleq. Deposited code is
never edited. Environment overrides support labelled same-vintage comparisons.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from mgp_core import ROOT
from applications.stella.stella_provenance import file_fingerprints, verify_fingerprints

STATUS = ROOT / 'data/stella2015/wrds_inputs/latest_acquisition_status.json'
REVENUE_FIELD = os.environ.get('STELLA_REVENUE_FIELD', 'revtq')
if REVENUE_FIELD not in {'revtq', 'saleq'}:
    raise ValueError('STELLA_REVENUE_FIELD must be revtq or an explicitly labelled saleq comparison')
if os.environ.get('STELLA_QUARTERLY_FILE'):
    QFILE = Path(os.environ['STELLA_QUARTERLY_FILE']).resolve()
else:
    status = json.loads(STATUS.read_text())
    if status.get('status') != 'downloaded' or not status.get('revtq_available'):
        raise ValueError('A verified Stella revenue extract is required; no saleq fallback')
    QFILE = STATUS.parent / status['file']


def read_quarterly(columns, *, path=None, field=None):
    """Load the source revenue into the legacy saleq slot without changing missingness."""
    path = QFILE if path is None else Path(path)
    field = REVENUE_FIELD if field is None else field
    source_columns = [field if c == 'saleq' else c for c in columns]
    frame = pd.read_csv(path, usecols=source_columns, dtype={'gvkey': str})
    if field != 'saleq' and 'saleq' in columns:
        frame = frame.rename(columns={field: 'saleq'})
    return frame


def input_description():
    return {'quarterly_source': str(QFILE.relative_to(ROOT)) if QFILE.is_relative_to(ROOT) else str(QFILE),
            'input_revenue_field': REVENUE_FIELD,
            'revenue_missingness': 'Source field only; no filling from saleq',
            'legacy_names': 'saleq variable and filenames are compatibility names for the stated input_revenue_field'}


def verify_input_provenance(provenance):
    expected = provenance['input_file_fingerprints']
    current = file_fingerprints([QFILE], ROOT)
    if provenance.get('input_revenue_field') != REVENUE_FIELD or any(expected.get(k) != v for k,v in current.items()):
        raise ValueError('ADF/adjusted histories use a different quarterly file or revenue field; rebuild them')
    verify_fingerprints(expected, ROOT)
