# PoolForge protocol

Three channels, all opened **outbound** by the agent, so no port forwarding is needed:

| Channel | URL | Content |
|---|---|---|
| REST | `POST /api/rooms/{code}/join` | join, receive a node token |
| Control WebSocket | `/ws/agent/{code}/{node_id}?token=T` | small JSON messages (text frames) |
| Relay WebSocket | `/ws/relay/{code}/{node_id}?token=T` | tensor frames (binary) |
| Direct TCP (optional) | `host:port` advertised by the agent | tensor frames (binary) |

Browsers use `/ws/dashboard/{code}` (read-only events) and REST with the
`X-Admin-Token` header for commands.

## 1. Joining

```
POST /api/rooms/X7K2AB/join
{"hostname": "laptop", "os": "Windows 11", "node_id": "laptop-a"}     # node_id optional

200 {"node_id": "laptop-a", "node_token": "<secret>", "tcp_secret": "<room secret>", "heartbeat_s": 2.0}
403 room locked   404 unknown room   409 node id already online   422 invalid body   429 rate limited
```

## 2. Tensor frame (data plane)

```
+-----------------+--------------------------+------------------------------+
| 4 bytes         | N bytes                  | M bytes                      |
| N, big-endian   | JSON header (UTF-8)      | raw tensor bytes             |
+-----------------+--------------------------+------------------------------+

header = {"kind": "activation" | "gradient" | "labels",
          "dtype": "float32" | "int64" | ...,   "shape": [64, 2048],
          "job_id": "job-1a2b3c4d#0",            # job id + attempt number
          "step": 17, "microbatch": 0,
          "num_bytes": 524288, "encrypted": false}
```

The receiver validates every header field, checks that
`num_bytes == len(payload)` and that `prod(shape) * itemsize == len(raw)`, then
rebuilds the tensor with `torch.frombuffer(...).reshape(shape)`. Limits: header at most
64 KB, payload at most 512 MB. **No pickle.**

With `--encrypt-passphrase`, the raw bytes are replaced by
`[12-byte nonce][AES-256-GCM ciphertext + 16-byte tag]` and `encrypted` is `true`.

### Envelopes

| Link | Sender writes | Receiver gets |
|---|---|---|
| Relay | `[4B len][target id][frame]` | `[4B len][sender id][frame]` (coordinator rewrites it) |
| TCP | first message `[4B len][4B id-len][id][tcp_secret]`, then `[4B total][4B id-len][sender id][frame]` | same |

The relay looks up the target **only inside the sender's room**. If the target's relay
socket is briefly disconnected, up to 64 frames (at most 60 s old) are queued for it.

### Messages per training step (2 stages, M micro-batches)

```
stage 0  --activation(step,m)-->  stage 1      x M
stage 0  --labels(step,m)------>  stage 1      x M
stage 0  <--gradient(step,m)----  stage 1      x M
```

## 3. Control messages (JSON over the control WebSocket)

Every message is validated with pydantic; unknown fields or types are rejected and
ignored. The full definitions are in `common/schemas.py`.

### Agent -> coordinator

| type | when | main fields |
|---|---|---|
| `stats` | every 2 s | ram_total_gb, ram_free_gb (min(real, cap)), proc_ram_mb, cpu_percent, status, transport, tcp_host/port, docker, cloudflared |
| `step` | after every step, from every stage | job_id, stage, step, epoch, loss (last stage only), step_time_ms, bytes_sent/received, avg_send_ms, proc_ram_mb |
| `checkpoint` | after saving | job_id, stage, step |
| `job_state` | started / done / error | job_id, stage, state, detail |
| `app_state` | hosting progress | deployment_id, status (starting/running/stopped/error), url, detail |

### Coordinator -> agent

| type | fields |
|---|---|
| `assign` | job_id, stage_index, num_stages, start_layer, end_layer, specs, input_shape, num_classes, prev_node, next_node, transport, peers, hp{batch_size, microbatches, epochs, steps_per_epoch, lr, optimizer, checkpoint_every, seed, dataset}, resume_step, attempt |
| `control` | action: pause / resume / stop, job_id, reason |
| `run_app` | deployment_id, app_id (allow-listed id only) |
| `stop_app` | deployment_id |

## 4. Dashboard events (`/ws/dashboard/{code}`)

On connect the browser receives a `snapshot` (nodes, job, metrics, node_metrics,
deployments, locked). After that it receives `node`, `job` (with a `message` for
the log), `step`, `node_step`, `deployment` and `locked`.

## 5. REST summary

| Method | Path | Auth |
|---|---|---|
| GET | `/health`, `/api/models`, `/api/apps` | none |
| POST | `/api/rooms` | none (10 per minute per IP) |
| GET | `/api/rooms/{code}` | room code |
| POST | `/api/rooms/{code}/join` | room code (20 per minute per IP) |
| POST | `/api/rooms/{code}/admin/check`, `/lock`, `/plan`, `/jobs`, `/jobs/{id}/{pause,resume,stop}` | admin |
| GET | `/api/rooms/{code}/jobs`, `/jobs/{id}`, `/jobs/{id}/export?format=csv\|json` | room code |
| GET / POST | `/api/rooms/{code}/deployments`, `/deployments/{id}/{stop,restart}` | POST = admin |

Interactive docs: `http://<coordinator>/docs`.
