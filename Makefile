.PHONY: help install install-dev install-api install-audio install-models \
        test test-cov lint typecheck demo serve clean

PYTHON ?= python

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package + lightweight deps (no GPU stack)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[api,audio,dev]"

install-dev: install  ## Alias: dev install
	@true

install-api:  ## Install only the FastAPI server subset
	$(PYTHON) -m pip install -e ".[api]"

install-audio:  ## Install only the audio I/O subset
	$(PYTHON) -m pip install -e ".[audio]"

install-models:  ## Install the FULL heavy ML stack (GPU box only)
	$(PYTHON) -m pip install -e ".[models]"
	@echo ""
	@echo "Reminder: this pulls torch, transformers, speechbrain, onnxruntime,"
	@echo "xformers, funasr, accelerate, bitsandbytes, etc. — ~5 GB of wheels."
	@echo "Only run this on the box where you will actually run inference."

test:  ## Run the test suite (stub mode; no weights, no GPU)
	$(PYTHON) -m pytest -q -m "not gpu and not slow"

test-cov:  ## Run tests with coverage
	$(PYTHON) -m pytest -m "not gpu and not slow" --cov=vajravoice --cov-report=term-missing

test-all:  ## Run ALL tests, including GPU-marked (needs install-models)
	$(PYTHON) -m pytest -q

lint:  ## Lint with ruff
	$(PYTHON) -m ruff check vajravoice tests

typecheck:  ## Type-check with mypy
	$(PYTHON) -m mypy vajravoice

demo:  ## Run the stub-mode CLI demo (no GPU, no weights)
	$(PYTHON) -m scripts.synthesize --text "Namaskar, aapan ek navin prakalp baddal bolu." --config configs/stub.yaml

assets:  ## Regenerate the SVG visualizations in assets/
	$(PYTHON) scripts/generate_assets.py

animations:  ## Regenerate the animated GIFs in assets/
	$(PYTHON) scripts/generate_animations.py

media: assets animations  ## Regenerate all visual assets (SVGs + GIFs)

serve:  ## Start the FastAPI server (stub mode by default)
	$(PYTHON) -m uvicorn vajravoice.api.server:app --host 0.0.0.0 --port 8000 --reload

clean:  ## Remove build artifacts and caches
	rm -rf build dist *.egg-info .pytest_cache .coverage .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
