# Viva demo script (7 steps, ~10 minutes)

Two ways to run it:
- **Two laptops** (best): the coordinator on Render/Railway (see `docs/DEPLOY.md`),
  or on laptop A with laptop B on the same Wi-Fi.
- **One laptop**: `bash scripts/dev_two_agents.sh` (Windows:
  `scripts\dev_two_agents.ps1`) starts a coordinator plus two agents capped at 1 GB and 2 GB.

**Before the viva:** run `make test` once (all green). Train one MNIST job so the
dataset is downloaded. Open `benchmarks/results/results.md` in a tab. Have
`docs/screenshots/` ready in case the network fails.

---

### 1. Create a pool (1 min)
Open the coordinator URL and click **Create Pool**. Show the join link, the room code
and the admin token.
> *Say:* "The link carries only the room code. The admin token stays in my browser
> and is needed for Start, Stop and Deploy."

### 2. A friend joins with one command (1 min)
On laptop B, open the join link. It shows `poolforge join CODE --server URL`. Run it.
> *Say:* "The agent connects **outbound** over a WebSocket, so it works behind home
> routers with no port forwarding. It gets its own secret node token."

### 3. Live nodes (1 min)
Open **Nodes**: both laptops appear with hostname, free RAM and CPU, updating every 2 s.
Point at `--ram-cap-gb`.
> *Say:* "I capped this laptop at 1 GB to simulate a small machine."

### 4. The model doesn't fit one laptop (2 min)
**Training** -> model `large_mlp`, depth 8 (with the dev script's 1 GB + 2 GB caps):
- width **6144** (~4.0 GB) -> **Preview split** -> "pool too small": even the pool can't fit it.
- width **4608** (~2.3 GB) -> **Preview split** -> a 2-stage plan. The biggest laptop has
  only ~1.8 GB usable, so this model fits **only because of pooling**. Show the split
  plan: contiguous layers per laptop, sized by free RAM.
> *Say:* "Each laptop builds only its own layers. The full model never exists in one
> memory." (Code: `build_layers(specs, start, end)` in `common/models.py`.)

### 5. Train across both laptops (2 min)
Pick `mnist_cnn` (fast, real MNIST) and click **Start**. Show the loss falling, epoch,
step time, steps/sec, per-node RAM and the split plan. Open **Network**: bytes per
step, send time, MB/s per node. Click **Pause** and then **Resume**.
> *Say (the autograd trick):* "Laptop A sends `out.detach()`. Laptop B calls
> `requires_grad_()` on what it receives, computes the loss and calls `backward()`,
> then sends `inp.grad` back. Laptop A continues with `out.backward(grad)`. Each laptop
> updates its own weights with its own optimizer."
> *Proof:* `tests/test_split_training.py` shows split loss equals single-process loss
> within 1e-5.

### 6. Robustness (1 min)
While training, close laptop B's terminal (Ctrl+C). The dashboard shows **waiting**.
Run the same `poolforge join ... --node-id <same id>` again. Training resumes from the
last checkpoint. (If it stays away for more than 30 s, the job fails cleanly.)
> *Say:* "Stages checkpoint every K steps. The coordinator restarts all stages from
> the newest checkpoint they all have."

### 7. Honest results + hosting (2 min)
Open **Reports**, then export CSV. Show `benchmarks/results/results.md`:
- capacity: the pool trains a 1.75x bigger model,
- but it is 18% slower on localhost, and about 2x slower with 50 ms latency.

Optional: **Deploy** -> Static website -> URL (Cloudflare tunnel if `cloudflared` is
installed, otherwise IP:port) -> open it -> Stop.
> *Say:* "Only allow-listed apps can run. The network can never send a command line."

---

## Likely questions

| Question | Short answer |
|---|---|
| Is RAM really combined? | No. Each laptop holds part of the model; together they *store* more. See LIMITATIONS.md. |
| Why is it slower? | Every step waits for activations and gradients to cross the network (benchmark e). |
| Why not data parallelism? | Data parallelism copies the WHOLE model to every laptop, which doesn't help when the model is too big. |
| How do you know it's correct? | Loss-equivalence test against single-process training with the same seed (`test_split_training.py`). |
| Why not pickle? | Unpickling network data can execute code. We send a JSON header plus raw bytes (`common/protocol.py`). |
| What if a laptop dies? | Pause, wait 30 s, resume from a common checkpoint, or fail cleanly (`coordinator/hub.py`). |
| Security? | Node tokens, admin token, hashed in SQLite, rate limits, pydantic validation, allow-listed apps, optional AES-GCM. |
| Micro-batching? | Splits a batch so stages overlap. Small gain with latency, a loss without it (benchmark f). |
