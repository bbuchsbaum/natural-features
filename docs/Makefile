QUARTO_PYTHON ?= $(shell command -v python)
export QUARTO_PYTHON
QUARTODOC_CONFIG = _quartodoc.yml

.PHONY: lint quarto quarto-preview

lint:
	cd .. && python tools/docs_lint.py

quarto: lint
	quartodoc build --config $(QUARTODOC_CONFIG) && quarto render

quarto-preview:
	quartodoc build --config $(QUARTODOC_CONFIG) && quarto preview --no-browser
