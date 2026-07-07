PYTHON ?= $(shell if command -v uv >/dev/null 2>&1; then uv run --group docs python -c 'import sys; print(sys.executable)' 2>/dev/null || command -v python; else command -v python; fi)
QUARTODOC ?= $(shell if command -v uv >/dev/null 2>&1; then printf 'uv run --group docs quartodoc'; else printf 'quartodoc'; fi)
QUARTO_PYTHON ?= $(PYTHON)
export QUARTO_PYTHON
QUARTODOC_CONFIG = _quartodoc.yml

.PHONY: lint quarto quarto-preview

lint:
	cd .. && $(PYTHON) tools/docs_lint.py

quarto: lint
	$(QUARTODOC) build --config $(QUARTODOC_CONFIG) && quarto render

quarto-preview:
	$(QUARTODOC) build --config $(QUARTODOC_CONFIG) && quarto preview --no-browser
