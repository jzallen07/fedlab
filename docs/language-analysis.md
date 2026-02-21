# FedForge Language Analysis: Python vs TypeScript vs Rust vs Zig

**Date:** 2026-02-21
**Context:** Evaluating which language to use for building FedForge end-to-end, based on the requirements in [PRD.md](../PRD.md).

---

## Project Requirements Summary

FedForge requires:
- **FL Server:** Central aggregation server (FedAvg and variants)
- **FL Clients:** Local CNN training on medical imaging data (chest X-rays)
- **Communication:** gRPC between server and clients
- **Deployment:** Docker Compose (1 server + N client containers)
- **ML Training:** CNN image classification (ResNet, EfficientNet, etc.)
- **Data Handling:** Non-IID dataset partitioning, data loading pipelines
- **Visualization:** Convergence curves, per-client metrics (Jupyter notebooks)
- **Privacy (nice-to-have):** Differential privacy, secure aggregation
- **Framework:** The PRD specifies Flower (flwr) + PyTorch

The PRD does not explicitly mandate Python, but the proposed directory structure (`.py` files, `pyproject.toml`) and framework choices (Flower, PyTorch) are Python-native.

---

## Quick Comparison

| Dimension | Python | TypeScript | Rust | Zig |
|-----------|--------|------------|------|-----|
| **FL Framework** | Flower (mature, 6.6k stars) | None exist | None mature (RustFL is proof-of-concept) | None exist |
| **ML Training** | PyTorch 2.9 (dominant) | TensorFlow.js (capable, fewer examples) | Burn 0.20 (functional) or tch-rs (libtorch wrapper) | Zigrad (experimental) or C interop |
| **gRPC** | grpcio 1.78 (Flower handles it) | @grpc/grpc-js (9.5M downloads/wk) | Tonic (production-grade, 188M downloads) | gRPC-zig (72 stars) or C wrapper |
| **Medical Imaging** | pydicom, MONAI, torchvision | No DICOM; basic via tfjs | dicom-rs + image crate | No DICOM, zigimg lacks JPEG |
| **Differential Privacy** | Opacus (official Flower integration) | Unmaintained npm packages | OpenDP (Rust core) | Nothing exists |
| **Docker Image Size** | 800MB-2GB+ (with PyTorch) | 200-400MB (with tfjs-node) | 10-50MB (static binary) | 5-20MB (static binary) |
| **Visualization** | Matplotlib, Jupyter (native) | D3.js, Chart.js (browser-native) | No native option | Nothing exists |
| **Time to Working Demo** | Weeks | Months | Months | Many months |
| **Community/Tutorials** | Hundreds of FL tutorials | ~16 FL repos on GitHub | ~3 FL repos total | 0 FL projects found |
| **Language Stability** | Stable (3.13+) | Stable (ES2024+) | Stable (2021 edition) | Pre-1.0 (0.15.2), breaking changes expected |

---

## 1. Python

### Overview

Python is the natural language for this project. The PRD was written with Python in mind, and the entire FL ecosystem is Python-first.

### Pros

**Ecosystem dominance for FL and ML:**
- **Flower** (flwr v1.26.1) is the recommended FL framework with 6,600+ GitHub stars. It provides built-in strategies (FedAvg, FedProx, FedAdam, FedAdagrad, FedYogi, FedMedian, FedTrimmedAvg), a simulation engine (Ray-backed Virtual Client Engine scaling to 15M clients), and `flwr-datasets` with non-IID partitioners (Dirichlet, Pathological, Shard, IID, etc.).
- **PyTorch** (v2.9.0) is the dominant framework for medical imaging research. Chest X-ray classification papers overwhelmingly use PyTorch with ResNet-50, EfficientNet, or DenseNet architectures.
- **MONAI** (by NVIDIA/King's College London) is a PyTorch-based medical imaging toolkit with GPU-accelerated I/O, domain-specific transforms, and its own FL APIs.
- **Opacus** (v1.5.4, by Meta) provides differential privacy for PyTorch with as few as 2 lines of code change. There is an [official Flower + Opacus integration example](https://flower.ai/docs/examples/opacus.html).

**Mature data science stack:**
- NumPy 2.4.1, pandas 3.0.0, scikit-learn 1.8.0, Matplotlib 3.10.8 are all current and fully compatible.
- Jupyter notebooks are the standard for ML visualization and analysis.

**Extensive learning resources:**
- Flower has official quickstart tutorials for PyTorch, a "Flower in 30 Minutes" notebook, and a community Slack with 6,800+ members.
- Kaggle has working examples of Flower + PyTorch FL on CIFAR-10 and other datasets.
- Multiple published tutorials cover federated chest X-ray classification specifically.

**Developer experience:**
- `uv` (Astral, Rust-based) provides 10-100x faster package installation than pip.
- Pyright/ty for type checking, Ruff for linting.
- Flower simulation mode allows single-process debugging with breakpoints.

### Cons

**Performance bottlenecks:**
- The GIL limits true multithreaded parallelism for CPU-bound tasks. However, this is largely mitigated for FedForge: each FL client runs in its own Docker container (separate process), and PyTorch releases the GIL during C extension calls (CUDA kernels, BLAS ops).
- Python 3.13 introduced experimental free-threaded mode; Python 3.14 matures this further.
- Serialization overhead: Flower handles weight serialization (NumPy arrays to protobuf) internally, but for large models (ResNet-50, ~25M parameters), per-round serialization is noticeable.

**Docker image sizes:**
- PyTorch + CUDA images can reach 7.6 GB. CPU-only images are significantly smaller but still 800MB-2GB.
- Mitigation: multi-stage builds, `python:3.x-slim` base, CPU-only PyTorch for the portfolio demo (training small CNNs on chest X-ray subsets does not require GPU).

**Flower-specific issues:**
- Simulation mode has [known issues on Windows](https://github.com/adap/flower/issues/5512) (Ray + Python 3.13 incompatibility).
- Frequent API changes: recent migration from TaskIns/TaskRes to message-based communication, CLI config format changes. Upgrading versions requires migration effort.
- No built-in secure aggregation (DP is supported via Opacus, but cryptographic secure aggregation is not native).

**grpcio caveats:**
- Streaming RPCs are slower in Python than other languages due to extra thread creation.
- No `fork()` support (gRPC core uses multithreading internally), which can cause issues with multiprocessing-based FL simulations.
- Large binary wheel sizes.

### Verdict

Python is the **overwhelmingly practical choice**. Every component of FedForge has a mature, well-documented library in Python. The "time to working demo" is measured in weeks, not months. The performance and image size concerns are real but manageable for a portfolio project.

---

## 2. TypeScript

### Overview

TypeScript offers strong type safety and excellent developer tooling, but the FL and ML ecosystems are dramatically thinner than Python's.

### Pros

**ML training is possible via TensorFlow.js:**
- TensorFlow.js (v4.22.0, ~19k GitHub stars) supports both training and inference in Node.js.
- `@tensorflow/tfjs-node` uses native TensorFlow C library bindings, giving near-native performance for computation kernels. The bottleneck is JavaScript-to-C bridge marshaling, not the math itself.
- `@tensorflow/tfjs-node-gpu` provides CUDA acceleration on Linux (10-100x speedup for large models).
- Supports Conv2D, MaxPool, transfer learning, and the full Keras-like layers API.

**gRPC support is excellent:**
- `@grpc/grpc-js` (v1.14.3) has 9.5M weekly downloads, is maintained by Google, and is written in TypeScript. Full type safety with generated TypeScript interfaces from `.proto` files.

**Type safety advantages:**
- TypeScript interfaces for model weight structures, FL protocol messages, and configuration objects catch errors at compile time.
- Monorepo-friendly: server and client can share type definitions.
- End-to-end typed gRPC with generated interfaces.

**Browser-native visualization:**
- D3.js, Chart.js, Plotly.js provide rich interactive visualization without Jupyter.
- A web dashboard for FL training progress could be a compelling portfolio differentiator.

**Docker deployment is mature:**
- `node:22-alpine` base images are ~80-100MB.
- Multi-stage builds can reduce images by 70% vs naive builds.

### Cons

**No FL framework exists:**
- Flower has no JavaScript/TypeScript SDK. The only Google-affiliated browser FL project (PAIR-code/federated-learning) is abandoned (~157 stars, last meaningful update ~2019).
- Only 16 out of 1,551 "federated-learning" GitHub repos are in TypeScript (1%). Python has 891 (57%).
- You would need to build the FL orchestration layer (FedAvg, client selection, weight exchange protocol, non-IID partitioning) entirely from scratch.

**Weak ML ecosystem for medical imaging:**
- All published chest X-ray CNN training examples are in Python TensorFlow or PyTorch. No TensorFlow.js examples for medical imaging were found.
- No DICOM library in JavaScript/TypeScript.
- Browser memory limits make large image dataset training challenging (50k images at 224x224 in float32 requires ~8GB).

**Differential privacy is essentially unsupported:**
- The `differential-privacy` npm package is inactive (no releases in 12+ months).
- OpenMined's `differentialprivacy-ts` appears abandoned.
- You would need to implement noise mechanisms manually (feasible but not turn-key).

**Data handling is less mature:**
- Danfo.js (~5k stars) is the closest thing to pandas but is "generally slower and may not be as feature-rich."
- NumJs (~2.3k stars) is minimally maintained.
- TensorFlow.js tensors cover most weight manipulation needs but lack the breadth of NumPy/SciPy.

**tfjs-node-gpu is Linux-only:**
- No GPU acceleration for Node.js on macOS or Windows. This limits development workflow.

### Verdict

TypeScript is **technically feasible** but requires significantly more engineering effort. The ML training layer (tfjs-node) works, gRPC is excellent, and the type safety is genuinely valuable. However, building the entire FL protocol from scratch instead of using Flower adds months of work. The strongest argument for TypeScript is **portfolio differentiation** -- very few people have built FL systems in TypeScript, so a working demo would stand out. The strongest argument against is the dramatically higher effort with no ecosystem to lean on.

---

## 3. Rust

### Overview

Rust offers strong performance, memory safety, and excellent deployment characteristics. The ML ecosystem is functional but less mature than Python's, and no production FL framework exists.

### Pros

**Strong ML training library (Burn):**
- **Burn** (v0.20.0, ~9.1k GitHub stars) is a pure-Rust deep learning framework with autograd, Conv2d, MaxPool2d, Dropout, Linear layers, and optimizers (SGD, Adam). It has working MNIST CNN training examples.
- GPU support via CubeCL: NVIDIA CUDA, AMD ROCm HIP, Apple Metal, WebGPU, Vulkan. CPU with SIMD acceleration.
- A 2025 benchmark showed Burn+CUDA achieving 97% of PyTorch+CUDA performance with lower memory overhead.
- **tch-rs** (v0.23.0) provides Rust bindings to libtorch (PyTorch's C++ backend), giving full PyTorch semantics if needed.

**Production-grade gRPC (Tonic):**
- Tonic (v0.14.3, ~11.6k stars, 188M+ downloads) is explicitly designed as "a core building block for production systems." First-class async/await with Tokio, streaming, TLS, health checking.
- prost (Protocol Buffers) integrates seamlessly with Tonic for codegen from `.proto` files.
- Performance comparable to or exceeding Go's gRPC implementation.

**Excellent Docker/deployment story:**
- Static binaries with musl: 5-15MB for a web service. `FROM scratch` images total 10-50MB.
- A 4-container FedForge stack (1 server + 3 clients) could total ~40-120MB vs 3-8GB for Python equivalents.
- Startup in milliseconds (no interpreter, no GC warmup).
- Deterministic memory usage without garbage collector overhead.

**Medical imaging support exists:**
- **dicom-rs** (v0.9.0): Pure Rust DICOM implementation with pixel data decoding.
- **image** crate: PNG, JPEG, TIFF, WebP, and many more formats. Resize with Lanczos3 filtering.
- **imageproc**: Edge detection, filtering, morphology, contrast adjustment.

**Differential privacy via OpenDP:**
- OpenDP (v0.12.0) has its core library written in Rust with Python/R bindings. Composable DP mechanisms suitable for FL.

**Data handling:**
- **ndarray**: Rust's NumPy equivalent with Rayon parallelism, BLAS support, and an [ndarray for NumPy users](https://docs.rs/ndarray/latest/ndarray/doc/ndarray_for_numpy_users/index.html) guide.
- **Polars** (v0.53.0): DataFrame library 3-10x faster than pandas, based on Apache Arrow.

**Existing FL proof-of-concepts:**
- **RustFL**: Async FL with tch-rs, differential privacy, and Shamir's secret sharing. v0.1.0 on crates.io.
- **candle-fl**: FedAvg implementation using Hugging Face Candle.
- **Xaynet**: Masked FL with homomorphic encryption (unmaintained but architecturally interesting).

### Cons

**No mature FL framework:**
- Flower has no Rust SDK. The existing Rust FL projects (RustFL, candle-fl, Xaynet) are proof-of-concept or unmaintained.
- You must build the FL orchestration layer from scratch, or use a hybrid approach (Python Flower + Rust training via PyO3).

**ML ecosystem described as "experimental":**
- [arewelearningyet.com](https://www.arewelearningyet.com/) describes the Rust ML ecosystem as "ripe for experimentation, but not very complete yet."
- No equivalent of `pip install torch; import torch` simplicity. Expect more boilerplate and less tutorial coverage.
- Burn is the most promising option but is still pre-1.0 and actively evolving.

**Developer experience friction:**
- Clean builds: 2-5+ minutes depending on dependencies. Incremental builds: 5-30 seconds.
- Borrow checker challenges for ML workloads: shared mutable state (model weights), graph structures (computation graphs), complex lifetimes all create friction. Common workarounds: `Arc<Mutex<T>>`, `Rc<RefCell<T>>`, or using the ML framework's abstractions.
- No Jupyter-equivalent interactive workflow (though `evcxr` Jupyter kernel exists).

**No visualization libraries:**
- No native plotting/charting library in Rust. Would need to export data and use external tools (Python matplotlib, web-based charts, or gnuplot).

**Steep learning curve:**
- Ownership, lifetimes, trait bounds require significant investment to internalize.
- Debug tooling for ML is less mature than Python's (no easy tensor inspection like `print(tensor.shape)`).

### Hybrid Approach (PyO3)

A viable middle ground: use Python + Flower for FL orchestration, with Rust (via PyO3) for performance-critical training loops. This is proven in production by Polars, Hugging Face tokenizers, Ruff, and Pydantic v2. Reports of 10-11x throughput improvements when replacing Python hot paths with Rust.

### Verdict

Rust is the **strongest alternative to Python** for this project. Burn provides functional CNN training, Tonic provides excellent gRPC, and the deployment story is compelling. However, building the FL layer from scratch and the steeper learning curve add significant time. The pure-Rust approach is a **differentiated portfolio project** -- few people have built FL systems in Rust. The hybrid approach (Python Flower + Rust training via PyO3) offers the best of both worlds but adds integration complexity.

---

## 4. Zig

### Overview

Zig is a pre-1.0 systems language with excellent C interop and deployment characteristics, but virtually no ML or FL ecosystem.

### Pros

**Exceptional deployment characteristics:**
- Static binaries as small as ~300KB. Cross-compilation built into the toolchain (`zig build -Dtarget=x86_64-linux-musl`).
- `FROM scratch` Docker images with zero runtime dependencies.
- The smallest possible container images of any option evaluated.

**First-class C interop:**
- `@cImport`/`@cInclude` directly imports C headers without writing bindings. The compiler translates C to Zig.
- Can link libtorch, TensorFlow Lite, OpenBLAS, DCMTK, or any C library directly.
- Zig ships as a drop-in C/C++ compiler and can compile C/C++ code alongside Zig code.

**Comptime features (theoretical ML benefits):**
- Compile-time tensor shape verification could catch dimension mismatches before runtime.
- Zero-cost generics more principled than C++ templates.
- Potential for statically verified neural network layer compatibility.

**Memory safety improvements over C:**
- Bounds checking, optional types, error unions, debug allocator (use-after-free detection).
- Cleaner syntax and fewer footguns than C++.
- Dramatically simpler build system than CMake.

**Notable production users:**
- Bun (JavaScript runtime), TigerBeetle (financial database), ZML (AI inference).

### Cons

**Language is not stable:**
- Latest release: 0.15.2 (October 2025). 1.0 estimated mid-to-late 2026.
- Breaking changes between minor versions (0.14 to 0.15 had breaking API changes).
- VSCode extension was archived November 2025 during the GitHub-to-Codeberg migration.

**No ML training ecosystem:**
- The most mature Zig ML project, **ZML** (~1.9k stars, backed by a company), is **inference-only**. It cannot be used for the local training step in FL.
- **Zigrad** and **Zorch** are experimental tensor libraries with autograd support, but neither is stable or well-tested for CNN training.
- No established CNN training pipeline exists in Zig.

**Critical gaps for medical imaging:**
- **zigimg** (the main image library) does not support JPEG read/write.
- No DICOM library exists in Zig.
- Chest X-ray datasets are typically in DICOM or JPEG format, making this a hard blocker.

**No gRPC ecosystem:**
- **gRPC-zig** has only 72 GitHub stars. Protobuf support exists (zig-protobuf) but is less battle-tested.
- The safer approach (wrapping gRPC C Core via C interop) adds significant complexity.

**No FL framework or community:**
- Zero federated learning projects found in the Zig ecosystem.
- ~10-15 ML-related projects total across the entire ecosystem.

**No visualization:**
- No plotting or charting libraries exist in Zig.

**No differential privacy or secure aggregation libraries.**

**Package management limitations:**
- No official central registry (unlike npm, crates.io, PyPI). Packages fetched from Git URLs.
- Community registries (Zigistry) are unofficial.

**ZLS limitations:**
- Cannot resolve complex comptime expressions; misses many type errors that the compiler catches.

### Verdict

Zig is **not viable for this project as a primary language**. Nearly every component would need to be built from scratch or wrapped via C interop: ML training, image loading, gRPC, visualization, privacy mechanisms. The areas where Zig excels (deployment, binary size, cross-compilation) represent perhaps 5-10% of the total project effort.

The most viable Zig-based approach would be a hybrid architecture: Zig for the FL server/networking layer, calling into C libraries for ML training. But at that point, you are effectively writing a C/C++ ML system with a Zig wrapper, and the added complexity is hard to justify.

---

## Recommendation Summary

### For a portfolio project (FedForge's stated goal):

| Priority | Language | Rationale |
|----------|----------|-----------|
| **1st** | **Python** | 10x faster path to a working demo. Flower + PyTorch + Opacus + MONAI give you everything the PRD requires out of the box. The FL ecosystem is Python-first. |
| **2nd** | **Rust** | Best alternative if you want to demonstrate systems-level skills. Burn + Tonic + dicom-rs cover the key requirements. The deployment story (tiny containers, millisecond startup) is genuinely impressive. Consider the hybrid approach (Python Flower + Rust training via PyO3) for faster delivery. |
| **3rd** | **TypeScript** | Feasible with tfjs-node + @grpc/grpc-js, and the browser-native visualization angle is compelling. But no FL framework means building the orchestration layer from scratch. |
| **4th** | **Zig** | Not recommended. Pre-1.0 language with no ML training ecosystem, no JPEG/DICOM support, no FL community, and no visualization. Nearly everything must be built from scratch or wrapped via C interop. |

### If the goal were "most impressive portfolio piece":

A **Python + Rust hybrid** (Flower for FL orchestration, Rust for training via PyO3, tiny Docker images) would demonstrate the widest range of skills while remaining practical to deliver.

### If the goal were "fastest to a working demo":

**Python, unambiguously.** The PRD's proposed architecture (Flower + PyTorch + Docker Compose) can produce a working demo with `docker-compose up` in the shortest timeline.

---

## Key Version Numbers (as of 2026-02-21)

| Package | Version | Language |
|---------|---------|----------|
| Flower (flwr) | 1.26.1 | Python |
| PyTorch | 2.9.0 | Python |
| Opacus | 1.5.4 | Python |
| grpcio | 1.78.1 | Python |
| TensorFlow.js | 4.22.0 | TypeScript/JS |
| @grpc/grpc-js | 1.14.3 | TypeScript/JS |
| Burn | 0.20.0 | Rust |
| Tonic | 0.14.3 | Rust |
| tch-rs | 0.23.0 | Rust |
| dicom-rs | 0.9.0 | Rust |
| Polars | 0.53.0 | Rust |
| Zig | 0.15.2 | Zig |
| ZML | N/A (inference-only) | Zig |

---

*This analysis was compiled from web research conducted on 2026-02-21. All version numbers and ecosystem assessments reflect the state of each ecosystem at that date.*
