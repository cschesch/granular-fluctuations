"""Replay the selected nonregular population from its saved certificate."""
import numpy as np
import inference_mc as mc
from mgp_core import RUN_OUTPUT_CACHE, lpt_buckets
from tests.support import require_files

TEST_GROUP = "integration"
TEST_CACHE_INPUTS = (".cache/run_outputs/selected_nonregular_zipf_r3.npz",)


def test_selected_population():
    path = RUN_OUTPUT_CACHE / "selected_nonregular_zipf_r3.npz"
    require_files(path)
    saved = np.load(path)
    rng = np.random.default_rng(20260901)
    variance = np.exp(.5 * rng.standard_normal(1000))
    variance /= variance.mean()
    first_loading = rng.standard_normal((1000, 3))
    raw = rng.standard_normal((1000, 3))
    assert np.array_equal(variance, saved["sigma2_e"])
    assert np.array_equal(raw, saved["raw_loadings"])
    sizes = mc.power_law_sizes(1000, 1.0)
    granular = float((sizes**2) @ variance)
    scale = granular * .97 / (.03 * np.sum((sizes @ first_loading)**2))
    buckets, _ = lpt_buckets(sizes**2, 10)
    grouping = np.zeros((10, 1000))
    for j, bucket in enumerate(buckets):
        grouping[j, bucket] = sizes[bucket]
    loading = grouping @ (np.sqrt(scale) * first_loading)
    original_covariance = loading @ loading.T + np.diag((grouping**2) @ variance)
    assert mc._population_regularity(original_covariance) is True
    assert mc._population_regularity(saved["sigma_grouped_normalized"]) is False
