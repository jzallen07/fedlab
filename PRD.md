# FedForge — Federated Learning Portfolio Project

**Status:** Draft  
**Created:** 2026-02-12  
**Author:** Zack (with Argent)

---

## 1. Overview

FedForge is a portfolio/tutorial project demonstrating **federated learning** (FL) — a decentralized machine learning approach where models are trained across multiple clients without sharing raw data. Only model updates (weights/gradients) are exchanged with a central server.

**Goals:**
- Build a working FL pipeline from scratch
- Demonstrate privacy-preserving ML concepts
- Create a visually impressive, explainable demo
- Learn the FL ecosystem (Flower framework, aggregation strategies, non-IID handling)

---

## 2. What Is Federated Learning?

### Core Concept

```
┌────────────────────────────────────────────────────────────┐
│                      FL SERVER                              │
│  • Holds global model                                       │
│  • Sends model to selected clients each round               │
│  • Receives weight updates (deltas or full weights)         │
│  • Aggregates (FedAvg: weighted average by # samples)       │
│  • Broadcasts new global model                              │
└──────────────────────┬─────────────────────────────────────┘
                       │
          ┌────────────┼────────────┐
          │            │            │
     ┌────▼────┐  ┌────▼────┐  ┌────▼────┐
     │Client 0 │  │Client 1 │  │Client 2 │  ...
     │─────────│  │─────────│  │─────────│
     │Local    │  │Local    │  │Local    │
     │Dataset  │  │Dataset  │  │Dataset  │
     └─────────┘  └─────────┘  └─────────┘
```

### The FL Loop (Per Round)

1. **Server** sends current global model weights to selected clients
2. **Clients** load weights into local model
3. **Clients** train for N epochs on their own local data (data never leaves)
4. **Clients** return updated weights (or deltas) to server
5. **Server** aggregates updates (e.g., FedAvg = weighted average by dataset size)
6. **Server** now has improved global model
7. Repeat for R rounds

### Why Federated Learning?

| Benefit | Explanation |
|---------|-------------|
| **Privacy** | Raw data never leaves the client. Only model updates move. |
| **Compliance** | Enables GDPR/HIPAA-compliant ML across organizations |
| **Bandwidth** | Don't ship massive datasets; only weights cross the network |
| **Edge Intelligence** | Train on data where it's generated (phones, hospitals, sensors) |

---

## 3. Project Ideas (Pick One or Combine)

### Option A: Federated Hospital Diagnostic Model ⭐ RECOMMENDED

**Concept:** Multiple simulated "hospitals" each hold local chest X-rays or patient data. Train a classifier (pneumonia detection, readmission risk) locally, federate to a central model.

**Why it's good:** 
- Directly relevant to healthcare AI
- Clear privacy/compliance story
- Visually compelling (medical imaging)

**Datasets:**

| Dataset | Description | Access |
|---------|-------------|--------|
| [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) | 224k chest X-rays, 14 pathology labels (Stanford) | Free, registration |
| [NIH Chest X-ray](https://nihcc.app.box.com/v/ChestXray-NIHCC) | 112k images, 14 disease labels | Free download |
| [Synthea](https://synthetichealth.github.io/synthea/) | Synthetic patient generator (tabular) | Open source |

**Federation Simulation:** Split dataset by random partition or by label distribution (non-IID) to simulate different hospital populations.

---

### Option B: Federated Wearable Health

**Concept:** Simulate 100+ "users" with step count, heart rate, sleep data. Train a wellness/risk model locally on each, federate for population-level insights.

**Why it's good:**
- Consumer health angle
- Natural per-user federation boundary
- Easy to visualize

**Datasets:**

| Dataset | Description | Access |
|---------|-------------|--------|
| [PPG-DaLiA](https://archive.ics.uci.edu/ml/datasets/PPG-DaLiA) | PPG heart rate during daily activities, 15 subjects | UCI, free |
| [WESAD](https://archive.ics.uci.edu/ml/datasets/WESAD) | Wrist/chest sensors, stress detection, 15 subjects | UCI, free |
| [PAMAP2](https://archive.ics.uci.edu/ml/datasets/PAMAP2+Physical+Activity+Monitoring) | IMU + heart rate, 18 activities, 9 subjects | UCI, free |
| [Sleep-EDF](https://physionet.org/content/sleep-edfx/1.0.0/) | Polysomnography (sleep stages), 197 recordings | PhysioNet, free |

**Federation Simulation:** Each subject = one client. Natural non-IID (people have different activity patterns).

---

### Option C: Federated LLM Fine-Tuning (Advanced)

**Concept:** Multiple "organizations" have private document corpora. Fine-tune a small LLM (or LoRA adapters) locally on each, federate the adapter weights.

**Why it's good:**
- Cutting edge (most FL demos are vision/tabular)
- Shows LLM-era thinking
- Impressive for portfolio

**Datasets:**

| Dataset | Description | Federation Angle |
|---------|-------------|-----------------|
| [20 Newsgroups](https://scikit-learn.org/stable/datasets/real_world.html#newsgroups-dataset) | Classic text classification, 20 topics | Each newsgroup = one "org" |
| [The Pile](https://pile.eleuther.ai/) (subsets) | 800GB diverse text, 22 sub-corpora | Each sub-corpus = different org |
| [PubMed Abstracts](https://pubmed.ncbi.nlm.nih.gov/download/) | Biomedical literature | Split by journal or MeSH category |
| [Enron Emails](https://www.cs.cmu.edu/~enron/) | Corporate emails | Split by employee/department |

**Federation Simulation:** Each topic/domain = one client. Fine-tune small model (GPT-2, Phi-2) or LoRA adapters.

---

### Option D: Federated MNIST/EMNIST (Baseline/Tutorial)

**Concept:** Classic intro project. Distribute handwritten digits across N clients with non-IID splits (each client only sees certain digits).

**Why it's good:**
- Simple enough to implement in a weekend
- Well-documented in FL literature
- Good for learning before tackling harder projects

**Datasets:**

| Dataset | Description |
|---------|-------------|
| [MNIST](http://yann.lecun.com/exdb/mnist/) | 70k handwritten digits (0-9) |
| [EMNIST](https://www.nist.gov/itl/products-and-services/emnist-dataset) | Extended MNIST with letters |

**Federation Simulation:** Pathological non-IID: each client gets only 2-3 digit classes. Tests FL robustness.

---

## 4. Technical Architecture

### Framework: Flower (flwr)

[Flower](https://flower.dev/) is the recommended FL framework:
- Python-native, great DX
- Supports PyTorch, TensorFlow, JAX
- Built-in simulation mode for fast iteration
- Real gRPC for distributed deployment
- Active community, good docs

**Alternatives:** TensorFlow Federated (more verbose), PySyft (privacy-focused), FedML (research-oriented)

### Development Progression

| Stage | Mode | Description |
|-------|------|-------------|
| **1. Simulation** | `flwr.simulation` | Everything in one Python process. Fast iteration, debugging. |
| **2. Multi-process** | Local gRPC | Server + N client processes on localhost. Real networking. |
| **3. Docker Compose** | Containers | Server + N client containers. Portfolio demo ready. |
| **4. Distributed** | Cloud/Edge | Optional: actual VMs, Raspberry Pis, cloud instances. |

### Docker Architecture (Target Demo)

```
┌─────────────────────────────────────────────────────────┐
│                   Local Docker Network                   │
│                                                         │
│   ┌─────────────┐                                       │
│   │   SERVER    │  ← "Company HQ"                       │
│   │  (container)│                                       │
│   │             │                                       │
│   │ Global model│                                       │
│   │ Aggregation │                                       │
│   └──────┬──────┘                                       │
│          │ gRPC                                         │
│   ┌──────┴──────┬─────────────┐                        │
│   ▼             ▼             ▼                        │
│ ┌─────┐      ┌─────┐      ┌─────┐                      │
│ │ C1  │      │ C2  │      │ C3  │  ← "Field Sites"    │
│ │     │      │     │      │     │    (containers)      │
│ │Data │      │Data │      │Data │                      │
│ │Shard│      │Shard│      │Shard│                      │
│ └─────┘      └─────┘      └─────┘                      │
└─────────────────────────────────────────────────────────┘
```

**Key properties:**
- Data stays in each client container (never shared)
- Only model weights cross the Docker network
- Containers are isolated — can't peek at each other
- Trivially extensible to remote deployment

### Proposed Directory Structure

```
fedforge/
├── README.md
├── PRD.md                      # This document
├── pyproject.toml
├── docker-compose.yml
│
├── src/
│   ├── server/
│   │   ├── __init__.py
│   │   ├── app.py              # FL server entry point
│   │   ├── strategy.py         # FedAvg or custom aggregation
│   │   └── config.py
│   │
│   ├── client/
│   │   ├── __init__.py
│   │   ├── app.py              # FL client entry point
│   │   ├── train.py            # Local training loop
│   │   └── data.py             # Data loading + partitioning
│   │
│   └── common/
│       ├── __init__.py
│       ├── model.py            # Shared model architecture
│       └── utils.py
│
├── data/
│   └── .gitkeep                # Downloaded datasets
│
├── scripts/
│   ├── download_data.py
│   ├── partition_data.py       # Split dataset into N shards
│   ├── run_simulation.py       # One-command simulation
│   └── run_distributed.sh      # Spin up Docker stack
│
├── notebooks/
│   └── analysis.ipynb          # Visualize convergence
│
└── Dockerfile
```

---

## 5. Key Concepts to Demonstrate

### Must-Have

- [ ] Basic FL loop (server ↔ client weight exchange)
- [ ] FedAvg aggregation
- [ ] Non-IID data distribution (clients have different data)
- [ ] Convergence visualization (accuracy over rounds)
- [ ] Docker-based multi-container demo

### Nice-to-Have

- [ ] Differential privacy (add noise to updates)
- [ ] Secure aggregation (encrypted updates)
- [ ] Client selection strategies (not all clients every round)
- [ ] Comparison: federated vs. centralized training
- [ ] Model compression (reduce communication cost)

---

## 6. Success Criteria

1. **Working Demo:** `docker-compose up` spins up server + 3 clients, runs FL training, produces a trained model
2. **Visualization:** Jupyter notebook showing convergence curves, per-client metrics
3. **Documentation:** Clear README explaining FL concepts, how to run, architecture
4. **Extensibility:** Easy to swap datasets/models for different project variants

---

## 7. Resources & References

### Frameworks
- [Flower Docs](https://flower.dev/docs/)
- [Flower Tutorials](https://flower.dev/docs/framework/tutorial-series-what-is-federated-learning.html)
- [TensorFlow Federated](https://www.tensorflow.org/federated)

### Papers
- [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629) — Original FedAvg paper
- [Advances and Open Problems in Federated Learning](https://arxiv.org/abs/1912.04977) — Comprehensive survey

### Tutorials
- [Flower Quickstart (PyTorch)](https://flower.dev/docs/framework/tutorial-quickstart-pytorch.html)
- [Building a Federated Learning System](https://blog.openmined.org/federated-learning-of-a-rnn-on-raspberry-pis/)

---

## 8. Next Steps

1. [ ] Pick primary project variant (recommend: Option A - Hospital Diagnostic)
2. [ ] Set up Python environment with Flower + PyTorch
3. [ ] Download and explore dataset
4. [ ] Implement simulation mode first
5. [ ] Graduate to Docker deployment
6. [ ] Write documentation and analysis notebook

---

*This PRD will be refined as the project progresses.*
