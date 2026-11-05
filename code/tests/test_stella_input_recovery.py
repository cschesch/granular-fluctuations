#!/usr/bin/env python3
"""Revenue missingness, cache isolation, and literal seasonal-exception checks."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from applications.stella import stella_quarterly_inputs as inputs
from applications.stella.prepare_stella_sales import x13_spec
from applications.stella.stella_provenance import file_fingerprints


class InputRecoveryChecks(unittest.TestCase):
    def test_revenue_missingness_is_not_filled_from_sales(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'quarterly.csv'
            pd.DataFrame({'gvkey':['000001','000002'],'saleq':[100.,200.],
                          'revtq':[np.nan,190.]}).to_csv(path,index=False)
            revenue = inputs.read_quarterly(['gvkey','saleq'], path=path, field='revtq')
            self.assertTrue(pd.isna(revenue.saleq.iloc[0]))
            self.assertEqual(revenue.saleq.iloc[1],190.)
            self.assertEqual(revenue.gvkey.iloc[0],'000001')

    def test_absent_revenue_field_does_not_fall_back(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'quarterly.csv'
            pd.DataFrame({'gvkey':['000001'],'saleq':[100.]}).to_csv(path,index=False)
            with self.assertRaises(ValueError):
                inputs.read_quarterly(['gvkey','saleq'], path=path, field='revtq')

    def test_same_file_different_field_invalidates_cached_adjustment(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'quarterly.csv'
            path.write_text('gvkey,revtq,saleq\n000001,10,10\n')
            manifest={'input_file_fingerprints':file_fingerprints([path],inputs.ROOT),
                      'input_revenue_field':'saleq'}
            with patch.object(inputs,'QFILE',path), patch.object(inputs,'REVENUE_FIELD','revtq'):
                with self.assertRaises(ValueError):
                    inputs.verify_input_provenance(manifest)

    def test_literal_tid_exception_changes_only_outlier_request(self):
        values=np.exp(np.arange(80)/100)
        start=pd.Period('1962Q1',freq='Q')
        regular=x13_spec(values,'fixture',start,'sax12')
        special=x13_spec(values,'fixture',start,'sax12_no_outlier')
        self.assertEqual(special,regular.replace('outlier{types=(all) method=addone}\n',''))
        self.assertIn('transform{function=log}',special)
        self.assertIn('seasonalma=msr',special)
        self.assertNotIn('automdl{',special)


if __name__=='__main__':
    unittest.main()
