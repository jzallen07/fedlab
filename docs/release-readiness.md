# R01 Release Readiness Checklist

Date (UTC): 2026-02-21
Scope: MVP release gate verification for FedForge local PoC.

## Gate Status

| Gate | Description | Status | Evidence |
|---|---|---|---|
| G0 | Foundation gate: scaffold + tooling + CI-equivalent checks green | PASS | `uv run ruff check src tests` and `uv run pytest` passed; logs in local shell output. Hardware validation log: `artifacts/release/r01/g0_hardware_validation.log` |
| G1A | FL core gate: FedAvg simulation + non-IID partitioning on BloodMNIST | PASS | `artifacts/release/r01/g1a_partition_bloodmnist.log`, `artifacts/partitions/bloodmnist_train_moderate_3clients_seed20260221.json`, `artifacts/release/r01/g1a_sim_smoke.log`, `artifacts/sim-smoke/summary.json` |
| G1B | Dataset extension gate: DermaMNIST profile run | PASS | `artifacts/release/r01/g1b_dermamnist_profile.log`, `artifacts/profiles/dermamnist_extension_cpu/reference_metrics.json` |
| G1C | Dataset extension gate: PathMNIST profile run | PASS | `artifacts/release/r01/g1c_pathmnist_profile.log`, `artifacts/profiles/pathmnist_extension_cpu/reference_metrics.json` |
| G2 | Integration gate: local distributed runtime smoke | PASS | `artifacts/release/r01/g2_local_distributed.log`, `artifacts/release/r01/local-distributed-server/summary.json`, `artifacts/release/r01/local-distributed-server/round_metrics.jsonl` |
| G3 | Realtime gate: monitor-api + dashboard telemetry flow | PASS | `artifacts/release/r01/g3_g5_docker_smoke.log` (`docker-smoke: ok`, websocket broadcast assertion path) |
| G4 | Control gate: start/pause/resume/stop end-to-end | PASS | `artifacts/release/r01/g3_g5_docker_smoke.log` (`control-run-id` and successful control sequence) |
| G5 | Demo gate: compose stack + reproducible docs flow | PASS | `artifacts/release/r01/g3_g5_docker_smoke.log` (compose up/down successful), `artifacts/release/r01/g5_notebook_exec.log` (analysis notebook command from README executed successfully) |

## Commands Executed

```bash
uv run ruff check src tests
uv run pytest
uv run python scripts/validate_hardware_modes.py --dataset all
uv run python scripts/partition_medmnist.py --dataset bloodmnist --root data/medmnist --split train --num-clients 3 --seed 20260221 --preset moderate --image-size 28 --output-dir artifacts/partitions
uv run python scripts/sim_smoke.py
uv run python scripts/run_phase_profile.py --profile configs/profiles/dermamnist_extension.yaml
uv run python scripts/run_phase_profile.py --profile configs/profiles/pathmnist_extension.yaml
uv run python scripts/run_local_distributed.py --num-clients 2 --rounds 1 --model-id hf-internal-testing/tiny-random-DeiTForImageClassification --train-examples 4 --eval-examples 2 --image-size 30 --train-mode head_only --device cpu --run-id r01-local --output-dir artifacts/release/r01/local-distributed-server
uv run python scripts/docker_smoke.py --project-name fedlab --skip-build
uv run python scripts/execute_notebook.py notebooks/convergence_analysis.ipynb
```

## Notes

- BloodMNIST partition check used `--image-size 28` to align with the repo-cached MedMNIST files in `data/medmnist/*.npz`.
- No code changes were required for runtime logic; this chunk captures release validation evidence and gate closure.

## Verdict

R01 PASS: All gates `G0-G5` and `G1A/G1B/G1C` are green.
