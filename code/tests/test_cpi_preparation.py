"""Check prepared CPI weights and aggregation identities."""

from applications.bls_cpi import check_cpi_sanity as cpi
from tests.support import require_files

TEST_GROUP = "integration"
TEST_CACHE_INPUTS = (".cache/derived/bls_cpi/*.csv", ".cache/derived/bls_cpi/*.parquet",
                     ".cache/derived/bls_cpi/panels/*")


def test_prepared_cpi():
    require_files(*(cpi.BCPI / name for name in (
        "cpi_price_long.parquet", "cpi_weight_series_1967_onward.csv",
        "cpi_weight_series_1967_onward_wide.parquet", "panels/cpi_1967_leaves.csv",
        "panels/cpi_1967_inflation.parquet",
    )))
    cpi.main()
