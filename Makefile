.PHONY: help install test lint format reproduce bench clean

help:
	@echo "Targets:"
	@echo "  install   - uv sync (dev extras)"
	@echo "  test      - run pytest"
	@echo "  lint      - ruff check"
	@echo "  format    - ruff format"
	@echo "  reproduce - regenerate headline figures from cached results"
	@echo "  bench     - run Triton kernel benchmark (requires CUDA)"
	@echo "  clean     - remove caches"

install:
	uv sync --extra dev

test:
	uv run pytest -q

lint:
	uv run ruff check src tests

format:
	uv run ruff format src tests

reproduce:
	uv run python scripts/make_figures.py

bench:
	uv run python scripts/bench_kernel.py

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
