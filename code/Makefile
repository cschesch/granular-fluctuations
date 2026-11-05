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
	$(PYTHON) -m tests.test_core
	$(PYTHON) -m tests.test_gradient_bootstrap_checkpoint
	$(PYTHON) -m applications.bls_cpi.check_cpi_sanity
	$(PYTHON) -m tests.test_campd_panel
	$(PYTHON) -m tests.test_tic_panel
	$(PYTHON) -m tests.test_debt_baselines
	$(PYTHON) -m tests.test_mc_new
	$(PYTHON) -m tests.test_grouped_giv
	$(PYTHON) -m tests.test_factor_rank_figure1_cache
	$(PYTHON) -m tests.test_atalay_elasticity_surface
