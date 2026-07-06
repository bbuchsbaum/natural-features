PYTHON ?= python
PIP ?= pip

.PHONY: install test test-smoke test-media test-nightly test-external import-snl \
	generate-tier-a validate-tier-a fetch-tier-b prepare-tier-b generate-golden \
	lint release-check ci-smoke ci-media ci-nightly setup-mfa-conda \
	benchmark-tier-a benchmark-gate

install:
	$(PIP) install -e ".[dev,storage,vision]"

test:
	pytest -q

test-smoke:
	pytest -q -m smoke

test-media:
	pytest -q -m media

# Nightly runs include dedicated nightly tests plus media diagnostics.
test-nightly:
	pytest -q -m "nightly or media"

test-external:
	NF_ENABLE_EXTERNAL_DATA=1 pytest -q -m external

generate-tier-a:
	$(PYTHON) scripts/generate_tier_a_stimuli.py

validate-tier-a:
	$(PYTHON) scripts/validate_tier_a_stimuli.py

fetch-tier-b:
	$(PYTHON) scripts/fetch_tier_b_stimuli.py --allow-missing-sha

prepare-tier-b:
	$(PYTHON) scripts/prepare_tier_b_clips.py

generate-golden:
	$(PYTHON) scripts/generate_golden_references.py

lint:
	ruff check src tests scripts

release-check:
	$(PYTHON) scripts/release_check.py

import-snl:
	$(PYTHON) scripts/import_snl_dataset.py

setup-mfa-conda:
	./scripts/setup_mfa_conda.sh mfa

benchmark-tier-a:
	$(PYTHON) -m natural_features.cli.main speech-benchmark \
		--manifest tests/benchmarks/manifests/tier_a_alignment_manifest.json \
		--backend auto \
		--json

benchmark-gate:
	$(PYTHON) scripts/check_alignment_benchmark_gate.py --report $(REPORT)

ci-smoke: validate-tier-a test-smoke

ci-media: validate-tier-a test-media

ci-nightly: validate-tier-a test-nightly
