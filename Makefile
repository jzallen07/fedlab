.PHONY: sync lint test sim-smoke docker-smoke central-train

sync:
	uv sync --extra dev

lint:
	uv run ruff check src tests

test:
	uv run pytest

sim-smoke:
	uv run python scripts/sim_smoke.py

docker-smoke:
	uv run python scripts/docker_smoke.py

central-train:
	uv run python scripts/run_centralized_training.py
