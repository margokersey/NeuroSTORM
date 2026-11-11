.PHONY: sandbox-gl sandbox sandbox-env sandbox-test hooks test test-unit test-integration test-cov test-fast lint format clean help

help:  ## Show this help message
	@echo "NeuroSTORM Test Commands"
	@echo "========================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

test:  ## Run all tests
	pytest tests/ -v

test-unit:  ## Run only unit tests
	pytest tests/ -v -m unit

test-integration:  ## Run only integration tests
	pytest tests/ -v -m integration

test-fast:  ## Run tests excluding slow ones
	pytest tests/ -v -m "not slow"

test-cov:  ## Run tests with coverage report
	pytest tests/ -v --cov=. --cov-report=html --cov-report=term

test-models:  ## Test model loading only
	pytest tests/test_model_loading.py -v

test-data:  ## Test data loading
	pytest tests/test_atlas_masking.py -v

lint:  ## Run linting checks
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

format:  ## Format code with black and isort
	black .
	isort .

format-check:  ## Check code formatting without modifying
	black --check --diff .
	isort --check-only --diff .

clean:  ## Clean test artifacts
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf **/__pycache__
	rm -rf **/*.pyc
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

install-test:  ## Install test dependencies
	pip install pytest pytest-cov pytest-xdist flake8 black isort

ci:  ## Run CI checks locally
	@echo "Running linting..."
	@make lint
	@echo "\nRunning tests..."
	@make test
	@echo "\nCI checks complete!"

hooks:  ## Enable repo git hooks (blocks large files/checkpoints/stray outputs)
	git config core.hooksPath .githooks

# ---- Local (no-GPU) sandbox: see notes/local_sandbox.md ----
sandbox-env:  ## Create the laptop conda env (neurostorm-sandbox, CPU/MPS, no CUDA)
	conda env create -f scripts/sandbox/environment-local.yml

sandbox:  ## Tiny synthetic-data end-to-end train/val/test run on CPU (~10 s)
	bash scripts/sandbox/run_smoke.sh

sandbox-test:  ## pytest suite with the pure-PyTorch mamba_ssm shim on the path
	PYTHONPATH=scripts/sandbox/shims:. pytest tests/ -m "not slow"

sandbox-gl:  ## Run the Great Lakes 5-fold/LOOCV/sbatch launchers on synthetic data (~10 min)
	bash scripts/sandbox/run_greatlakes_checks.sh all
