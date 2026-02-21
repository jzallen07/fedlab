.PHONY: lint test sim-smoke docker-smoke central-train

lint:
	ruff check src tests

test:
	pytest

sim-smoke:
	python scripts/sim_smoke.py

docker-smoke:
	python scripts/docker_smoke.py

central-train:
	python scripts/run_centralized_training.py
