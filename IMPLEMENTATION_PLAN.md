# FedForge Implementation Plan
Date: 2026-02-21
Status: Planning document (implementation has not started)
Audience: Two engineers executing in parallel with shared checkpoints

## 1. Objective
Build FedForge from empty repo to a portfolio-grade federated learning system with:
1. A working FL pipeline (simulation, local distributed, Docker Compose).
2. Deterministic non-IID data partitioning and repeatable metrics.
3. A modern TypeScript dashboard showing real-time node and round activity.

## 1.1 Decisions Locked (From Product Direction)
1. Frontend is React with TypeScript and Vite.
2. Frontend lives in the same repository (`apps/dashboard`).
3. Deployment target is local-only proof of concept, using production-style architecture principles.
4. Dashboard includes realistic operator controls (start, pause, resume, stop run).
5. Telemetry and run-state persistence use lightweight SQLite (no heavy external datastore required).
6. Use MedMNIST as the vision data source family, including `BloodMNIST`, `DermaMNIST`, and `PathMNIST`.
7. Use pretrained `facebook/deit-tiny-patch16-224` (~5M params) as the default transformer model path.
8. Keep dataset rollout phased for local CPU reliability: Phase A `BloodMNIST`, Phase B `DermaMNIST`, Phase C `PathMNIST`.
9. Dataset/tooling choices must remain free to use (no paid data or paid infrastructure requirements).
10. ML training pipeline must use Hugging Face tooling (`transformers` and ecosystem) rather than hand-rolled direct PyTorch training loops.

## 2. Execution Protocol (Mandatory Per Chunk)
Every chunk in this document must be completed in this exact sequence:
1. Pick up one unclaimed chunk that is not blocked by dependencies.
2. Do only the scoped work for that chunk.
3. Validate using the validation criteria listed for that chunk.
4. Commit using the commit snippet format, mark chunk complete, move to next.

Rules:
1. One chunk equals one commit.
2. If a chunk grows beyond commit size, split it before implementation.
3. Do not start chunks with unmet dependencies.
4. If blocked for more than 30 minutes, mark chunk BLOCKED with reason and move to another unblocked chunk.

## 3. Team Model and Lanes
Engineer 1 primary lane:
1. Platform and runtime foundations (`F01`, `F02`, `F05`).
2. FL server runtime and strategy integration (`A01`-`A06`, `I03`).
3. Monitoring/control plane (`M01`-`M06`) including SQLite persistence and run-control APIs.
4. Deployment and operations integration (`I04`) and release/quality gates with pair checkpoints.

Engineer 2 primary lane:
1. HF ML pipeline ownership (`B00`-`B07`) including DeiT model path, MedMNIST data pipeline, and `Trainer` orchestration.
2. Data quality and hardware validation (`D00`, `D00A`) plus phased dataset run profiles (`D01`-`D03`).
3. Frontend dashboard implementation (`U01`-`U09`) including realtime graph, metrics, and operator controls.
4. Analysis and reporting artifacts (`I06`) and pair checkpoints for final documentation/release.

Shared lane:
1. Integration checkpoints and e2e validation (`I01`, `I05`).
2. Documentation and release readiness (`I07`, `R01`).
3. Interface contracts (event schema, control API, metrics format) must be jointly reviewed at each gate boundary.

## 4. Architecture Baseline
### 4.1 FL Runtime Plane
1. `server` runs Flower strategy (FedAvg baseline) and global aggregation.
2. `client` processes train locally on shard data and upload model updates.
3. FL data exchange remains model weights and metrics only, never raw client datasets.

### 4.2 Telemetry and Dashboard Plane
1. Add `monitor-api` service (FastAPI) with:
2. REST endpoint for latest snapshot/history.
3. WebSocket endpoint for real-time event stream.
4. Control endpoints for operator actions (`start`, `pause`, `resume`, `stop`).
5. SQLite-backed event and run-state persistence.
6. Server and clients emit structured events to monitor-api.
7. Dashboard subscribes to WebSocket and renders live graph and metrics.

### 4.3 Frontend Stack (Required)
1. Vite + React + TypeScript.
2. Tailwind CSS.
3. shadcn/ui components.
4. Visualization libraries:
5. `reactflow` for node-edge network graph with animated flow.
6. `recharts` for rounds, loss, accuracy, latency charts.
7. Optional `framer-motion` for panel transitions and subtle activity animations.

### 4.4 Local-Only Operating Model
1. System is optimized for local compose-based execution, not production hardening.
2. Service boundaries (FL runtime, monitor API, UI) mirror real distributed design principles.
3. Security and auth are intentionally minimal for local learning workflows.

### 4.5 Dataset and Model Plan (Locked)
1. Dataset family: MedMNIST with three tracked datasets.
2. Phase A dataset: `BloodMNIST` (MVP baseline and fastest iteration loop).
3. Phase B dataset: `DermaMNIST` (mid-size extension run).
4. Phase C dataset: `PathMNIST` (largest run for stronger pathology narrative).
5. Model baseline: pretrained `facebook/deit-tiny-patch16-224` with classification head replacement.
6. Training strategy for laptop:
7. Start with frozen backbone + train head only.
8. Then optionally unfreeze last transformer block for short fine-tuning.
9. Input strategy: dataset samples are resized to model input resolution with deterministic transforms.

### 4.6 ML Pipeline Standard (Required)
1. Data loading and split handling use Hugging Face-compatible dataset pipeline patterns.
2. Preprocessing uses `AutoImageProcessor` aligned with DeiT expected normalization and resize behavior.
3. Training orchestration uses `transformers.Trainer` (or HF-compatible trainer abstraction), not a custom PyTorch loop.
4. Metrics computation uses `evaluate` with explicit accuracy/loss tracking per round and per client.
5. Runtime acceleration config uses `accelerate` defaults suitable for local CPU runs.
6. Reproducibility requirements:
7. Fixed seeds for data split, trainer, and client selection.
8. Versioned run config files for each dataset phase (`BloodMNIST`, `DermaMNIST`, `PathMNIST`).
9. Data preprocessing best practices:
10. Per-dataset class distribution report.
11. Train/val/test split audit and leakage checks.
12. Persisted preprocessing metadata and label mappings for traceability.

### 4.7 Mac Hardware Policy (CPU-Safe First)
1. Default execution mode is CPU-safe and deterministic for all FL smoke and baseline runs.
2. Optional acceleration mode uses Apple Metal (`mps`) when available on Apple Silicon.
3. Device selection order: `mps` when explicitly enabled and available, otherwise CPU.
4. Multi-client FL runs in parallel should default to CPU to avoid `mps` single-device contention.
5. `mps` mode is allowed for single-client debug/profiling and optional extension runs.
6. Set `PYTORCH_ENABLE_MPS_FALLBACK=1` in launch environment for graceful unsupported-op fallback.
7. If `mps` run fails stability checks, pipeline must auto-fallback to CPU and continue.

### 4.8 CPU-Safe Training Profiles (Default)
1. Use these as initial `TrainingArguments` presets for local laptop runs.
2. Global defaults:
3. `per_device_train_batch_size=8`
4. `per_device_eval_batch_size=8`
5. `gradient_accumulation_steps=2`
6. `num_train_epochs=1` per federated round
7. `learning_rate=5e-4` (head-only) and `1e-4` (last-block unfreeze)
8. `weight_decay=0.01`
9. `warmup_ratio=0.05`
10. `logging_steps=10`
11. `eval_strategy="epoch"`
12. `save_strategy="no"` during local round training
13. `dataloader_num_workers=0` (safe on macOS)
14. `max_grad_norm=1.0`
15. `seed` fixed per run config
16. Precision policy:
17. CPU baseline uses full precision (`fp32`).
18. `bf16` and `fp16` are disabled by default on Mac local runs.
19. MPS-specific override policy:
20. Keep conservative batch size (`4-8`).
21. Start with head-only fine-tuning.
22. Do not enable distributed training in `mps` mode.

## 5. Target Repository Layout Addendum
```text
fedlab/
├── IMPLEMENTATION_PLAN.md
├── pyproject.toml
├── docker-compose.yml
├── Dockerfile
├── src/
│   ├── server/
│   ├── client/
│   ├── common/
│   ├── ml/
│   │   ├── config.py
│   │   ├── data.py
│   │   ├── preprocess.py
│   │   ├── trainer.py
│   │   └── metrics.py
│   └── monitor/
│       ├── app.py
│       ├── schema.py
│       ├── db.py
│       ├── store.py
│       ├── ws.py
│       └── control.py
├── apps/
│   └── dashboard/
│       ├── src/
│       └── package.json
├── scripts/
├── tests/
│   ├── unit/
│   └── integration/
└── notebooks/
```

## 6. Event Contract (MVP)
Required event types:
1. `node_heartbeat`
2. `round_started`
3. `model_dispatched`
4. `client_train_started`
5. `client_train_completed`
6. `client_update_uploaded`
7. `aggregation_started`
8. `aggregation_completed`
9. `round_completed`
10. `node_error`
11. `run_requested`
12. `run_paused`
13. `run_resumed`
14. `run_stopped`

Required fields:
1. `event_id`
2. `ts`
3. `round`
4. `node_id`
5. `role` (`server` or `client`)
6. `status`
7. `latency_ms` (optional where applicable)
8. `payload_bytes` (optional where applicable)
9. `metrics` (optional dictionary)
10. `run_id` (required for run lifecycle events)

## 7. Milestones and Gates
1. G0 Foundation gate: scaffold + tooling + CI green.
2. G1A FL core gate: simulation runs with FedAvg and non-IID partitioning on `BloodMNIST`.
3. G1B Dataset extension gate: successful run profile on `DermaMNIST`.
4. G1C Dataset extension gate: successful run profile on `PathMNIST`.
5. G2 Integration gate: local distributed runtime passes smoke tests.
6. G3 Realtime gate: monitor-api + dashboard show live training flow with persisted telemetry.
7. G4 Control gate: dashboard controls can start/pause/resume/stop runs end-to-end.
8. G5 Demo gate: Docker Compose stack runs all services and docs are reproducible.

## 8. Dependency Waves
Wave 0:
1. Foundation and repo scaffolding.

Wave 1A:
1. Server/platform lane.

Wave 1B:
1. Client/data lane.

Wave 1C:
1. Dashboard foundation lane.

Wave 2:
1. FL integration and telemetry wiring.

Wave 3:
1. Docker integration, e2e validation, docs, release checklist.

Key joins:
1. `I01` depends on `A03` and `B07`.
2. `M04` depends on `A03` and `B07`.
3. `U06` depends on `U04` and `M05`.
4. `U09` depends on `M06` and `U06`.
5. `D02` depends on `D01`.
6. `D03` depends on `D02`.
7. `I05` depends on `A06`, `M06`, `U07`, and `U09`.
8. `R01` depends on all gate-critical chunks.

## 9. Chunk Backlog (Commit-Sized To-Do Checklist)
Status keys:
1. `[ ]` not started
2. `[~]` in progress
3. `[x]` complete
4. `[!]` blocked

### Wave 0: Foundation
| Status | ID | Owner | Depends On | Work Chunk | Validation | Commit Snippet |
|---|---|---|---|---|---|---|
| [x] | F01 | Eng1 | None | Create core directories (`src/server`, `src/client`, `src/common`, `src/ml`, `src/monitor`, `scripts`, `tests`, `notebooks`, `apps/dashboard`). | Directory checks pass. | `chore(F01): scaffold project directories` |
| [x] | F02 | Eng1 | F01 | Add `pyproject.toml` with runtime and dev dependencies for FL, monitoring, and HF ML stack (`transformers`, `datasets`, `evaluate`, `accelerate`). | Editable install succeeds. | `chore(F02): initialize project metadata and hf runtime deps` |
| [x] | F03 | Eng2 | F02 | Add lint/test tooling (`ruff`, `pytest`) and minimal test scaffold. | `ruff` and `pytest` run clean. | `chore(F03): add lint and test harness` |
| [x] | F04 | Eng2 | F03 | Add task runner commands (`make lint`, `make test`, `make sim-smoke`, `make docker-smoke`). | Make targets execute successfully. | `chore(F04): add make targets for dev workflow` |
| [x] | F05 | Eng1 | F04 | Add CI workflow for lint and test gates. | CI config validates and local equivalent passes. | `chore(F05): add ci workflow for lint and tests` |

### Wave 1A: Server and Platform Lane
| Status | ID | Owner | Depends On | Work Chunk | Validation | Commit Snippet |
|---|---|---|---|---|---|---|
| [x] | A01 | Eng1 | F05 | Implement server app entrypoint with CLI/config loading. | Server help command works. | `feat(A01): add server app entrypoint` |
| [x] | A02 | Eng1 | A01 | Implement FedAvg strategy wrapper and round hooks. | Strategy unit tests pass. | `feat(A02): implement fedavg strategy hooks` |
| [x] | A03 | Eng1 | A02 | Add server metric aggregation and artifact writer (`json/csv/checkpoint`). | Artifact files generated in test run. | `feat(A03): add server metrics and artifact outputs` |
| [x] | A04 | Eng1 | F05 | Add Dockerfile for runtime image used by server and clients. | Docker image builds cleanly. | `chore(A04): add runtime dockerfile` |
| [x] | A05 | Eng1 | A04 | Add local distributed launch script for server and N clients. | Local multi-process smoke run passes. | `feat(A05): add local distributed launcher` |
| [x] | A06 | Eng1 | A05 | Add initial `docker-compose.yml` for server + 3 clients. | Compose config validates. | `chore(A06): add compose stack for fl runtime` |

### Wave 1B: Client, Model, and Data Lane
| Status | ID | Owner | Depends On | Work Chunk | Validation | Commit Snippet |
|---|---|---|---|---|---|---|
| [x] | B00 | Eng2 | F05 | Add HF training config manifests for all dataset phases (Blood/Derma/Path) with CPU-safe defaults and optional Mac `mps` overrides. | Config validation tests pass for all manifests and device modes. | `feat(B00): add hf training manifests with cpu-safe and mps profiles` |
| [x] | B01 | Eng2 | F05 | Implement shared model module using pretrained `facebook/deit-tiny-patch16-224` with replaceable classifier head and freeze/unfreeze modes. | Model unit tests pass for forward shape and mode switching. | `feat(B01): add deit-tiny pretrained model module` |
| [x] | B02 | Eng2 | F05 | Add MedMNIST data script for `BloodMNIST`, `DermaMNIST`, and `PathMNIST` download/prep. | Script validates all three datasets can be prepared locally. | `feat(B02): add medmnist tri-dataset preparation script` |
| [x] | B03 | Eng2 | B02 | Implement deterministic non-IID partitioning script supporting dataset selection and client/site skew presets. | Partition determinism tests pass for all three datasets. | `feat(B03): add multi-dataset non-iid partitioner` |
| [x] | B04 | Eng2 | B02,B03 | Implement HF dataset preprocessing pipeline with `AutoImageProcessor`, deterministic transforms, and label mapping artifacts. | Preprocessing unit tests pass for all three datasets. | `feat(B04): add hf preprocessing pipeline for medmnist` |
| [x] | B05 | Eng2 | B00,B01,B04 | Implement training/evaluation pipeline using `transformers.Trainer` and `evaluate` metrics. | Trainer smoke test passes on CPU with head-only mode. | `feat(B05): add hf trainer pipeline for deit` |
| [x] | B06 | Eng2 | B05 | Implement Flower client wrapper around HF trainer lifecycle and model state serialization. | Client app CLI/help works and local train step emits expected metrics. | `feat(B06): add flower client wrapper for hf trainer` |
| [x] | B07 | Eng2 | B06 | Implement simulation runner script for N rounds and N clients with dataset flag and phase presets. | Simulation smoke run passes on `BloodMNIST`. | `feat(B07): add phase-aware simulation runner` |

### Wave 1C: Frontend Foundation Lane
| Status | ID | Owner | Depends On | Work Chunk | Validation | Commit Snippet |
|---|---|---|---|---|---|---|
| [x] | U01 | Eng2 | F05 | Scaffold Vite React TypeScript app in `apps/dashboard`. | Build completes with no errors. | `chore(U01): scaffold dashboard app with vite react ts` |
| [x] | U02 | Eng2 | U01 | Configure Tailwind and shadcn/ui baseline and theme tokens. | App renders styled components. | `chore(U02): configure tailwind and shadcn` |
| [x] | U03 | Eng2 | U02 | Build dashboard shell layout (header, graph panel, metrics panel, event stream panel). | Responsive layout checks pass. | `feat(U03): add dashboard shell layout` |
| [x] | U04 | Eng2 | U03 | Add topology visualization with `reactflow` using mock node/edge states and animations. | Graph renders and state transitions animate. | `feat(U04): add realtime topology graph with mock data` |
| [x] | U05 | Eng2 | U03 | Add `recharts` metrics widgets for round KPIs and trend charts (mock data). | Charts render and update from mocked store. | `feat(U05): add metrics charts with recharts` |

### Wave 2: FL and Telemetry Integration
| Status | ID | Owner | Depends On | Work Chunk | Validation | Commit Snippet |
|---|---|---|---|---|---|---|
| [ ] | I01 | Pair | A03,B07 | Integrate end-to-end FL simulation from server and clients with persisted metrics. | Simulation integration test passes. | `feat(I01): integrate e2e fl simulation pipeline` |
| [x] | I02 | Eng2 | I01 | Add unit tests for HF model/data/preprocess/trainer boundaries and determinism. | Unit suite passes. | `test(I02): expand hf pipeline unit coverage` |
| [ ] | I03 | Eng1 | I01,I02 | Add integration test for multi-round simulation with artifact assertions. | Integration suite passes. | `test(I03): add multi-round simulation integration test` |
| [x] | M01 | Eng1 | F05 | Implement `src/monitor/app.py` with FastAPI bootstrap and health route. | Service boots and health endpoint passes. | `feat(M01): add monitor api bootstrap` |
| [x] | M02 | Eng1 | M01 | Define telemetry schemas and validator layer in `src/monitor/schema.py`. | Schema tests pass. | `feat(M02): add telemetry schema validation` |
| [x] | M03 | Eng1 | M02 | Implement SQLite persistence layer and run-state tables for monitor service. | DB schema tests and startup initialization pass. | `feat(M03): add sqlite-backed telemetry persistence` |
| [x] | M04 | Eng1 | M03,A03,B07 | Instrument server and client event emission hooks for all required event types. | Emitted events visible in monitor logs. | `feat(M04): add telemetry emission hooks to fl runtime` |
| [x] | M05 | Eng1 | M04 | Add WebSocket broadcast manager for live event stream. | WS clients receive streamed events. | `feat(M05): add websocket event broadcast` |
| [x] | M06 | Eng1 | M03,M04 | Add monitor control endpoints and command routing for start/pause/resume/stop actions. | Control API contract tests pass. | `feat(M06): add run control endpoints and routing` |
| [x] | U06 | Eng2 | U04,M05 | Implement dashboard WS client and REST snapshot hydration. | UI receives and displays live events. | `feat(U06): connect dashboard to monitor api` |
| [x] | U07 | Eng2 | U06,U05 | Wire live telemetry into topology and metrics state store. | Node states and charts update in real time. | `feat(U07): wire realtime telemetry into dashboard views` |
| [x] | U08 | Eng2 | U07 | Add event log table and node detail drawer for drill-down debugging. | UI interaction tests pass. | `feat(U08): add event log and node detail inspector` |
| [x] | U09 | Eng2 | U06,M06 | Add dashboard control panel for run start/pause/resume/stop with action feedback. | UI can trigger and display control state transitions. | `feat(U09): add operator control panel` |

### Wave 2B: Dataset Phase Validation
| Status | ID | Owner | Depends On | Work Chunk | Validation | Commit Snippet |
|---|---|---|---|---|---|---|
| [x] | D00 | Eng2 | B04 | Add data quality and preprocessing audit pipeline (class balance, split integrity, leakage checks) for all three datasets. | Audit report artifacts generated for Blood/Derma/Path. | `feat(D00): add medmnist preprocessing and data quality audits` |
| [x] | D00A | Eng2 | B05 | Add hardware-mode validation script for `cpu` and optional `mps` detection/fallback behavior. | Validation shows deterministic CPU baseline and successful fallback behavior. | `test(D00A): add cpu and mps fallback validation` |
| [ ] | D01 | Eng2 | I03,D00,D00A | Establish baseline FL run profile for `BloodMNIST` (rounds, clients, batch size) and store reference metrics. | Repeatable baseline run completes on laptop CPU. | `feat(D01): add bloodmnist baseline run profile` |
| [ ] | D02 | Eng2 | D01 | Add and validate `DermaMNIST` run profile with tuned local CPU settings. | DermaMNIST profile run completes and metrics export is valid. | `feat(D02): add dermamnist extension run profile` |
| [ ] | D03 | Eng2 | D02 | Add and validate `PathMNIST` run profile with constrained rounds for local CPU runtime. | PathMNIST profile run completes and metrics export is valid. | `feat(D03): add pathmnist extension run profile` |

### Wave 3: Docker, E2E, Documentation, Release
| Status | ID | Owner | Depends On | Work Chunk | Validation | Commit Snippet |
|---|---|---|---|---|---|---|
| [x] | I04 | Eng1 | A06,M06,U06 | Extend compose stack with monitor-api and dashboard services plus SQLite volume mount. | Compose stack starts all services cleanly with persisted monitor DB. | `chore(I04): add monitor dashboard and sqlite volume to compose` |
| [ ] | I05 | Pair | I04,U07,U09 | Add Docker e2e smoke script that runs rounds and verifies dashboard event receipt and control actions. | E2E smoke command passes. | `test(I05): add docker realtime and control e2e smoke test` |
| [ ] | I06 | Eng2 | I01 | Create convergence notebook reading generated artifacts and telemetry exports. | Notebook executes end-to-end. | `feat(I06): add convergence and per-client analysis notebook` |
| [ ] | I07 | Pair | I05,I06,D03 | Update README with setup, runbook, architecture, troubleshooting, and screenshots. | Fresh clone runbook validated by both engineers. | `docs(I07): finalize runbook and dashboard docs` |
| [ ] | R01 | Pair | I07 | Execute release readiness checklist and mark MVP complete. | All gates G0-G5 and G1A/G1B/G1C green. | `chore(R01): complete mvp release readiness checklist` |

## 10. Interdependency Summary
1. Frontend mock dashboard work (`U01-U05`) can start immediately after `F05`.
2. Core FL integration (`I01`) must complete before telemetry event wiring stabilizes.
3. Telemetry stream (`M05`) is the contract boundary for live dashboard integration (`U06` onward).
4. Control-plane contract (`M06`) is required before dashboard controls (`U09`) can be integrated.
5. Data quality audits (`D00`) must complete before phase-profile benchmarking (`D01-D03`).
6. Compose-level realtime demo (`I04-I05`) is blocked until backend telemetry/control and frontend websocket/control views are complete.
7. Hardware validation (`D00A`) must complete before baseline profile benchmarking (`D01-D03`).

## 11. Dashboard UX Expectations (Non-Negotiable)
1. Modern, clean visual hierarchy and spacing.
2. Real-time animated node-edge graph for training and aggregation flow.
3. Low-latency event updates with clear status colors for each node.
4. Chart panel with global and per-client trend lines.
5. Event timeline suitable for debugging and demo narration.
6. Operator controls with clear run-state feedback and disabled-state safety while actions are in flight.

## 12. Risks and Mitigations
1. Risk: Event volume overwhelms UI rendering.
2. Mitigation: Throttle UI updates and batch events on client.
3. Risk: Contract drift between runtime and dashboard.
4. Mitigation: Shared typed event schema and contract tests.
5. Risk: Docker orchestration complexity increases flakiness.
6. Mitigation: Keep service startup health checks and deterministic smoke scripts.
7. Risk: Control commands create race conditions with round lifecycle.
8. Mitigation: Centralize run-state transitions in monitor-api with explicit finite-state rules.
9. Risk: HF DeiT training on CPU is slower than expected for largest dataset phase.
10. Mitigation: Default to head-only fine-tuning, reduced rounds, and phase-specific CPU budgets.
11. Risk: `mps` op coverage/stability differences on Mac affect repeatability.
12. Mitigation: Keep CPU as baseline, enable `PYTORCH_ENABLE_MPS_FALLBACK=1`, and treat `mps` as optional acceleration mode.

## 13. Definition of Done (MVP + Dashboard Addendum)
1. `docker compose up` runs server, 3 clients, monitor-api, and dashboard.
2. Dashboard shows live node status and round activity during training.
3. Dashboard can start, pause, resume, and stop runs from the control panel.
4. FL metrics and artifacts plus monitor telemetry are persisted in local outputs and SQLite.
5. Automated lint, unit, integration, and docker smoke checks are green.
6. README documents reproducible local setup and demo flow.
7. ML training pipeline is fully HF-based (`transformers` + ecosystem) with no custom direct PyTorch training loop.
8. CPU-safe baseline profiles are documented and pass on local Mac hardware, with optional `mps` mode validated.
