# PoolForge ⚡

> **Pools laptops' spare RAM/compute to train AI models and host apps together.**  
> An MCA Final Year Capstone Project.

[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue)]()
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey)]()

---

## 📖 Overview
PoolForge allows two or more student laptops on separate networks to pool their spare memory and compute into a single virtual cluster.
- **Distributed Pipeline Parallelism**: AI models are split by layer across nodes so no single laptop holds the full model in memory.
- **Zero Config NAT Traversal**: Outbound WebSocket relay connects machines behind campus Wi-Fi or home routers without port forwarding or VPNs.
- **Live Real-Time Dashboard**: Control training, watch loss convergence, monitor per-node RAM and CPU, and inspect network throughput live.
- **Edge App Hosting**: Deploy lightweight microservices and web apps onto pool nodes with automatic Cloudflare tunneling.

---

## 🚀 Quick Start (Under 10 Minutes)

### 1. Prerequisites
- Python 3.10+
- Git

### 2. Installation
```bash
git clone https://github.com/poolforge/poolforge.git
cd poolforge

# Setup virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e agent
```

### 3. Run Coordinator & Dashboard
```bash
# Starts coordinator on http://localhost:8000
python3 -m coordinator.server
```

### 4. Join Worker Nodes
```bash
# In another terminal / laptop:
poolforge join <ROOM_CODE> --server http://localhost:8000
```

---

## 🧪 Running Tests
```bash
make test
# or
pytest -v tests
```

---

## 📂 Project Structure
```text
poolforge/
├── agent/            # Python CLI worker (runs pipeline stages, reports stats)
├── coordinator/      # FastAPI + SQLite + WebSocket control plane & relay
├── dashboard/        # React + Vite + Recharts live web interface
├── common/           # Shared schemas, tensor framing wire format, model specs
├── benchmarks/       # Capacity, latency, and throughput benchmark scripts
├── deploy/           # Dockerfile and cloud deployment templates
├── docs/             # Technical docs, viva viva demo guide, limitations
├── scripts/          # Helper scripts for multi-agent local simulation
└── tests/            # Test suite (unit, integration, equivalence)
```

---

## ⚖️ Honest Engineering Trade-offs
1. **RAM is not literally merged**: Nodes hold partitions of the model weights and Adam optimizer states. Communication latency between layers is bounded by the network connection.
2. **Step Latency**: Step time increases over WAN/relay compared to single-machine training. The pool allows models that physically exceed a single laptop's RAM to train successfully.
