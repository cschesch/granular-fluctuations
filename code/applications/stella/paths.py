"""Cache locations for the Stella application used in the paper."""

from project_paths import ROOT

CALIBRATION = ROOT / "code/.cache/firm_networks/stella_revtq"
UNBALANCED = CALIBRATION.with_name("stella_revtq_unbalanced")
