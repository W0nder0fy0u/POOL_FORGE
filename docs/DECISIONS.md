# Design decisions

Defaults and trade-offs chosen while building PoolForge, with the reason for each.
(Format: decision - why - consequence.)

## Platform and packaging

**D1. Python 3.10+, PyTorch on CPU, FastAPI + uvicorn, SQLite, React + Vite + Recharts.**
These are the most common, well-documented choices, they run on Windows and Linux,
and they need no system services (no Postgres, Redis or GPU). No Unix-only APIs are
used (no `fork`, `signal.SIGKILL`, `fcntl`).

**D2. `pip install -e .` from the repo root (not `pip install -e agent`).**
The agent imports shared code from `common/`, which lives outside `agent/`, and
setuptools cannot package a folder outside its project directory. One root
`pyproject.toml` gives the `poolforge` command and the shared code in one step.
The Join page shows this command.

**D3. The coordinator never imports torch.**
Split planning uses `common/specs.py`, which is pure Python. That keeps the Docker
image around 150 MB instead of about 2 GB, and lets it run on free hosting tiers. A test
(`test_scaffold.py`) enforces this.

**D4. Exact versions in `requirements.txt`; looser ranges in `pyproject.toml`.**
`requirements.txt` reproduces the tested environment, while `pip install -e .` on a
friend's laptop accepts any recent torch they already have.

## Security

**D5. The join link carries only the 6-character room code. The "secret join token" is a
per-node token issued at join time.** The spec asks for "room code + secret join token" and
for the link to carry "the code only". We resolve that like this:
- The code (36^6, about 2.2 billion values) plus a join rate limit (20 per minute per IP) acts
  as the invite.
- `POST /join` returns a random **node token**. The control and relay WebSockets
  reject any connection without a valid token, so nodes outside the room cannot
  relay.
- The admin can **lock** the room once everyone has joined.
- The creator gets a separate **admin token** for Start/Stop/Deploy.

Tokens are stored as SHA-256 hashes and compared in constant time.

**D6. Admin token in an HTTP header (`X-Admin-Token`), never in a URL.**
URLs end up in logs and browser history. The dev script's `#admin=` fragment is
never sent to the server, and the page removes it immediately. Uvicorn's WebSocket
log lines are filtered to redact `token=`.

**D7. No pickle anywhere on the network.** Tensors travel as a JSON header plus raw bytes;
control messages are JSON validated by pydantic (`extra="forbid"`). Checkpoints are
local files, loaded with `torch.load(weights_only=True)`.

**D8. Hosting runs only allow-listed apps** (`common/apps.py`). The API takes an app
*id*. The command line is built only from the allow-list on the agent itself.
Docker containers get a 256 MB memory limit.

**D9. Encryption passphrase never reaches the coordinator.** Each agent derives an
AES-256-GCM key with PBKDF2 (200k iterations, room code as salt). The relay only sees
ciphertext. If one node has a different passphrase (or none), the job fails with a
clear message instead of hanging.

## Training engine

**D10. GPipe-style schedule: all micro-batch forwards, then all backwards.** It is
the simplest correct pipeline schedule to explain. 1F1B would use less activation
memory but is harder to reason about in a viva.

**D11. Layer i is initialised with seed `seed + i`.** A layer gets the same initial
weights whether it is built alone on laptop B or inside the full model on one machine.
That is what makes the split-vs-single loss test exact (equal within 1e-5).

**D12. Frames carry `job_id#attempt` and are matched by (kind, step, micro-batch).**
Frames can arrive out of order, and after a resume stale frames from the old attempt
may still be in flight. The mailbox ignores them.

**D13. Compute runs in a worker thread (`asyncio.to_thread`).** A big layer can take
seconds. The event loop must keep answering heartbeats and receiving frames, otherwise
the coordinator would think the laptop died.

**D14. Simulated latency delays incoming frames; it does not block the sender.** A real
network delays delivery without freezing the sending laptop. Sleeping in `send()` would
make micro-batching look useless for the wrong reason.

**D15. Step time = time between consecutive finished steps on a node.** For the first
step, timing starts when its input arrives. This measures the pipeline's real pace and
excludes one-off setup such as downloading MNIST.

**D16. Optimizers: Adam (default) and SGD with momentum 0.9.** The memory formula
assumes Adam (4 copies of every parameter), so the estimate is an upper bound for SGD.

## Scheduling

**D17. Memory estimate = params x 4 B x 4 + activation x batch x 4 B (spec formula),
with 10% of each node's free RAM held back.** The formula doesn't count the PyTorch
runtime (several hundred MB per process), so it is a planning estimate, not a
guarantee. The capacity benchmark compares it with measured RAM.

**D18. Biggest node first, contiguous ranges proportional to free RAM.** If
proportional shares don't fit (e.g. one huge layer), a second pass lets each node take
as many layers as fit. If nothing fits, the error says "pool too small" and names the
node and layers.

**D19. Direct TCP is used only if every stage's agent runs `--transport tcp`; otherwise
relay.** A mixed pool still works, just without the speed-up. A shared per-room secret
handshake stops strangers on the LAN from injecting tensors.

## Robustness

**D20. Node drop: pause, wait `POOLFORGE_DROP_GRACE_S` (default 30 s), then resume
from the newest checkpoint every stage has saved, or fail cleanly.** Agents keep their
last 3 checkpoints and report each one, so the coordinator always knows a step that all
stages can restore.

**D21. Agents reconnect with backoff 1, 2, 4, 8, 10, 10... seconds.** Training reports
made while offline are queued and sent later; stale RAM/CPU stats are dropped.

**D22. Heartbeat every 2 s; a node silent for 10 s is disconnected by the sweeper.**

## Benchmarks

**D23. Benchmarks drive the real system (separate processes, REST API) on one
machine, with 4 PyTorch threads per simulated laptop.** This makes single-vs-pool
comparisons fair on one 8-thread PC. Localhost has near-zero latency, so the
`--simulate-latency-ms` runs stand in for real Wi-Fi/Internet.

**D24. Synthetic data in benchmarks.** It removes the MNIST download from timings and
works offline. The dashboard default model still trains on real MNIST.

## UI

**D25. Orange theme: orange + white (light) and orange + black (dark)**, following the
OS setting with a toggle. Primary orange-red (#FF4500, Reddit-inspired) and warm coral
(#D97757, Claude-inspired). Chart series use a validated colour-blind-safe order that
starts with orange. Every status shows a text label, never colour alone.
