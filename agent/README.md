# PoolForge agent

The `poolforge` command. Install from the repo root with `pip install -e .`, then:

```bash
poolforge join <ROOM_CODE> --server https://<coordinator-host>
```

Useful flags: `--ram-cap-gb N` (pretend to have less RAM), `--transport tcp` (direct
LAN/Tailscale links), `--node-id NAME`, `--simulate-latency-ms MS`,
`--encrypt-passphrase PASS`. Run `poolforge join -h` for all of them.

Files: `cli.py` (arguments), `worker.py` (connection, heartbeats, messages),
`trainer.py` (runs one pipeline stage), `hosting.py` (Deploy tab apps).
