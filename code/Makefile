# Paper pipeline for Granularity in High Dimensions.
.DEFAULT_GOAL := all

PYTHON ?= python3
export OPENBLAS_NUM_THREADS := 1
export OMP_NUM_THREADS := 1
export MPLCONFIGDIR := $(CURDIR)/.cache/matplotlib

.PHONY: all applications variations simulations empirical test check collect syncdraft verify

all:
	mkdir -p .cache
	touch .cache/paper-start
	$(PYTHON) paper_applications.py
	$(PYTHON) paper_outputs.py collect --since .cache/paper-start

applications variations simulations empirical:
	$(PYTHON) paper_applications.py $@

verify:
	$(PYTHON) verify_release.py

check:
	$(PYTHON) paper_outputs.py check

collect:
	$(PYTHON) paper_outputs.py collect

# Separate from all so that only an explicit invocation updates Overleaf.
syncdraft:
	$(PYTHON) paper_outputs.py sync

test:
	$(PYTHON) test_core.py
	$(PYTHON) test_gradient_bootstrap_checkpoint.py
	$(PYTHON) -m applications.bls_cpi.check_cpi_sanity
	$(PYTHON) test_campd_panel.py
	$(PYTHON) test_tic_panel.py
	$(PYTHON) test_debt_baselines.py
	$(PYTHON) test_mc_new.py
	$(PYTHON) test_grouped_giv.py
	$(PYTHON) test_factor_rank_figure1_cache.py
	$(PYTHON) test_atalay_elasticity_surface.py
