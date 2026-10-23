# PoolForge Architectural Decisions Log (ADR)

This document tracks all design decisions, trade-offs, and defaults chosen during the development of PoolForge.

---

### ADR 001: Pinned Python Stack and Cross-Platform Portability
- **Date**: 2026-10-23
- **Context**: PoolForge must run seamlessly on both Linux and Windows student laptops without native compilation hurdles or Unix-only API dependencies (`os.fork`, `fcntl`, etc.).
- **Decision**:
  - Python 3.10+ runtime target.
  - Core frameworks: PyTorch for neural network execution and autograd, FastAPI + Uvicorn for asynchronous WebSocket and REST routing, Pydantic v2 for strict message serialization, and `psutil` for cross-platform OS hardware telemetry.
  - Cross-platform pathing via `pathlib.Path` and `os.path.normpath`.
- **Consequences**: Easy setup across mixed student OS environments; student viva examiners can inspect code running on Windows or Linux laptops without modification.

---

### ADR 002: Transport Layer Abstraction (`RelayTransport` vs `DirectTCPTransport`)
- **Date**: 2026-10-23
- **Context**: Home Wi-Fi networks and college campus NATs block incoming peer-to-peer connections without router configuration or port forwarding.
- **Decision**:
  - Default transport is **`RelayTransport`**: outbound binary WebSocket frames routed via the coordinator. Requires zero NAT traversal, port forwarding, or VPN.
  - High-performance option is **`DirectTCPTransport`**: point-to-point raw TCP sockets with framer for LAN, Wi-Fi hotspots, or Tailscale environments.
  - Both adhere to an abstract `Transport` interface so the pipeline training engine is decoupled from the physical transport.
- **Trade-off**: Relay adds ~2-15ms coordinator hop latency, which our benchmarks will explicitly measure and document. Direct TCP reduces step latency on LAN.

---

### ADR 003: Memory Estimation and Contiguous Layer Scheduling
- **Date**: 2026-10-23
- **Context**: Deep learning model splitting requires contiguous pipeline parallelism to pass activations and gradients sequentially.
- **Decision**:
  - Layer memory is estimated deterministically as:
    $$\text{Memory} = (\text{Parameters} \times 4 \times 4) + (\text{Activation Shape} \times 4 \times \text{Batch Size})$$
    (Accounting for 32-bit floats for weights, gradients, Adam first moments, and Adam second moments).
  - Scheduler allocates contiguous layer blocks $[\text{start}, \text{end})$ proportional to each node's reported free RAM cap.
  - If total pool RAM is insufficient for the model, the coordinator fails early with `"pool too small"` before allocating memory.
