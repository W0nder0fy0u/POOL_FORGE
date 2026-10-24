# Limitations (honest list)

**RAM is not literally merged.** Each laptop holds only its own layers. Together the
pool can *store* a model bigger than one laptop's RAM, but a program cannot see
"16 GB" when you have two 8 GB laptops. Only models built as layer lists (as in our
model zoo) can be split.

**Every step pays the network.** A 2-stage step needs at least two network trips:
activations forward and gradients back. With 50 ms one-way latency that adds roughly
100 ms or more to every step (see `benchmarks/results/e_latency.png`). For a model that
fits on one laptop, PoolForge is **slower** than training on that laptop alone. Its
value is fitting models that otherwise don't fit.

**Pipeline stages wait for each other.** With one micro-batch, only one laptop
computes at a time; the others idle. Micro-batching hides some of this, but there is
always a "bubble" at the start and end of each step. Pooling does not give an N-times
speed-up.

**Relay overhead.** By default every tensor crosses the coordinator twice (laptop to
server, server to laptop). That costs bandwidth on the server, extra latency, and on
free hosting tiers it may be throttled. Direct TCP avoids this but only works when the
laptops can reach each other (same Wi-Fi, hotspot or Tailscale).

**Memory estimates are estimates.** The formula counts weights, gradients, two Adam
states and one activation per layer. It does not count the PyTorch runtime (several
hundred MB per process), temporary buffers, or the dataset on the first stage. The
scheduler keeps 10% headroom, but a node can still run out of memory.
`--ram-cap-gb` / `--vram-cap-gb` only change what the agent *reports*; they do not enforce a limit.

**GPU sharing does not merge VRAM.** Each GPU holds only its own layers; activations
still travel over the network every step. A GPU makes its stage much faster, but a
slow network or a CPU-only stage can still be the bottleneck. The CUDA context also
uses a few hundred MB of VRAM that the estimate does not count.

**Voice calls are peer to peer (mesh).** Every person sends audio to every other
person, which is fine for about 6 people. Browsers only allow the microphone on
https:// or http://localhost, so on a LAN use the Render URL (or localhost) for calls.
Without a TURN server, calls can fail on strict networks (college Wi-Fi, mobile data).
Chat history is kept in memory (last 100 messages) and is lost on restart.

**Trust model.**
- Anyone with the room code can join until the admin locks the room. A malicious
  member can send wrong activations or gradients and spoil training (there is no
  verification of results).
- The coordinator operator can see all tensors unless `--encrypt-passphrase` is used,
  and can always see metadata (shapes, sizes, timing).
- Hosted apps run on members' laptops. They are limited to the allow-list, but a
  hosted app is still reachable by anyone who gets its URL.
- Rate limiting uses `X-Forwarded-For` behind a proxy, which a client can fake.
- No user accounts. Whoever holds the admin token is the admin.

**Not suitable for real-time workloads.** PoolForge cannot pool RAM for games,
browsers or other live programs. Training is a batch job that tolerates waiting;
interactive programs do not.

**Robustness limits.**
- If the **coordinator** restarts, running jobs are marked `interrupted` (rooms and
  history survive if `data/` is persistent). Jobs do not auto-restart.
- Resume works if the dropped laptop comes back with the same `--node-id` and the same
  folder (its checkpoints are local). A laptop that never returns means the job fails.
  Its layers are not re-assigned to the others automatically.
- Only one job per room at a time; one stage per node.

**Scale and scope.**
- One stage per GPU: a laptop with two GPUs only uses the first one.
- The model zoo is small: MLPs and CNNs made of linear/conv/pool/activation layers.
  No transformers, skip connections or branching models.
- Hosting: only the allow-listed sample apps and images. Docker apps were not tested on
  the development machine (no Docker there); only the "Docker missing" path is covered
  by tests. Cloudflare quick tunnels are for demos (no uptime guarantee).
- Developed on Linux. Windows is checked by CI: `.github/workflows/tests.yml` runs the
  full test suite on Windows and Linux on every push, including the friend install flow
  (fresh venv, `pip install` from the server, run `poolforge`). Those CI machines have no
  GPU, so GPU training was tested on Linux (NVIDIA) only; on Windows it uses the same
  PyTorch CUDA calls. Windows users with an NVIDIA GPU must install the CUDA build of
  PyTorch (the Join page has a checkbox for this); plain `pip install torch` on Windows is CPU-only.
