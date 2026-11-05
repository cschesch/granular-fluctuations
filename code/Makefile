# Paper pipeline for Granularity in High Dimensions.
.DEFAULT_GOAL := all

PYTHON ?= python3
export OPENBLAS_NUM_THREADS := 1
export OMP_NUM_THREADS := 1
export MPLCONFIGDIR := $(CURDIR)/.cache/matplotlib

.PHONY: all strict applications variations simulations empirical test test-fast test-numerical test-integration smoke check check-results collect syncdraft verify

all:
	mkdir -p .cache
	touch .cache/paper-start
	$(PYTHON) run_replication.py
	$(PYTHON) paper_outputs.py collect --since .cache/paper-start

strict:
	mkdir -p .cache
	touch .cache/paper-start
	$(PYTHON) run_replication.py --strict
	$(PYTHON) paper_outputs.py collect --strict --since .cache/paper-start

applications variations simulations empirical:
	$(PYTHON) run_replication.py $@

verify:
	$(PYTHON) verify_release.py

check:
	$(PYTHON) paper_outputs.py check

check-results:
	$(PYTHON) check_results.py

collect:
	$(PYTHON) paper_outputs.py collect

# Separate from all so that only an explicit invocation updates Overleaf.
syncdraft:
	$(PYTHON) paper_outputs.py sync

test:
	$(PYTHON) run_tests.py

test-fast test-numerical test-integration:
	$(PYTHON) run_tests.py $(patsubst test-%,%,$@)

smoke:
	$(PYTHON) smoke.py
