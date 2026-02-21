# FedForge 🔥

A federated learning portfolio project demonstrating privacy-preserving machine learning across distributed clients.

## Status

🚧 **Planning Phase** — See [PRD.md](./PRD.md) for project spec.

## Quick Concept

```
     ┌─────────────┐
     │   SERVER    │  Aggregates model updates
     └──────┬──────┘
            │
   ┌────────┼────────┐
   ▼        ▼        ▼
┌─────┐  ┌─────┐  ┌─────┐
│ C1  │  │ C2  │  │ C3  │   Clients train locally
│Data │  │Data │  │Data │   Data never leaves
└─────┘  └─────┘  └─────┘
```

**Data stays local. Only model weights move.**

## Project Variants

| Option | Domain | Dataset |
|--------|--------|---------|
| Hospital Diagnostic | Healthcare | CheXpert, NIH Chest X-ray |
| Wearable Health | Consumer IoT | PPG-DaLiA, WESAD |
| LLM Fine-Tuning | NLP | 20 Newsgroups, Pile subsets |
| MNIST Baseline | Tutorial | MNIST/EMNIST |

## Tech Stack

- **Framework:** [Flower](https://flower.dev/)
- **ML:** PyTorch
- **Deployment:** Docker Compose

---

*Created 2026-02-12*
