# CKPT-001 reproduction entry points.  Override PYTHON/ARTIFACT_ROOT if needed.
PYTHON ?= python
SEEDS ?= 1,2,3,4,5
PROFILE ?= smoke
ARTIFACT_ROOT ?= artifacts/reproduce-all

# These must be visible before Python/Torch initialize. They make the contract
# explicit; the generated manifest records their effective values.
export PYTHONHASHSEED ?= 0
export CUBLAS_WORKSPACE_CONFIG ?= :4096:8

.PHONY: reproduce foundation-analysis test lint typecheck docs clean-artifacts

reproduce:
	$(PYTHON) -m hga reproduce-all --artifact-root "$(ARTIFACT_ROOT)" --seeds "$(SEEDS)" --profile "$(PROFILE)"

foundation-analysis:
	$(PYTHON) -m hga foundation-analysis --out "$(ARTIFACT_ROOT)/foundation_analysis.json" --markdown "$(ARTIFACT_ROOT)/foundation_analysis.md"

test:
	$(PYTHON) -m pytest

lint:
	ruff check .

typecheck:
	mypy hga mimari egitim

docs:
	mkdocs build --strict

clean-artifacts:
	rm -rf artifacts/reproduce-all
