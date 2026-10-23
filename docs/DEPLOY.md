# Deploying PoolForge

Only the **coordinator** is hosted. Agents run on each person's laptop and connect
out to it.

## Option 1: Docker (any server or your own PC)

```bash
docker build -f deploy/Dockerfile -t poolforge-coordinator .
docker run -d -p 8000:8000 -v poolforge-data:/app/data --name poolforge poolforge-coordinator
```

Open `http://<server>:8000`. The image is ~165 MB (no PyTorch: the coordinator never
trains). Podman works too (`podman build ...` / `podman run ...`); with rootless podman
use `http://127.0.0.1:8000` instead of `localhost`.

Optional environment variables (`-e NAME=value`): `POOLFORGE_PUBLIC_URL` (the URL
printed in join commands), `POOLFORGE_REPO_URL` (git URL shown on the Join page),
`POOLFORGE_DROP_GRACE_S`. See `.env.example`.

## Option 2: Render (free, public HTTPS URL)

1. Push the repo to GitHub.
2. Render dashboard -> **New -> Blueprint** -> pick the repo (uses `deploy/render.yaml`).
   Or **New -> Web Service -> Docker**, with Dockerfile path `deploy/Dockerfile`.
3. Wait for the build, then open `https://<name>.onrender.com`.

Free plan notes: the service sleeps after ~15 minutes idle (the first visit wakes it in
~30 s), and there is no persistent disk, so rooms reset on redeploy. That's fine for a
demo; for a long-running pool, add a disk (commented in `render.yaml`).

## Option 3: Railway

See [`deploy/railway.md`](../deploy/railway.md).

## Friends on other networks

1. Deploy with option 2 or 3 to get a public HTTPS URL.
2. Open it, **Create Pool**, and send the join link.
3. Each friend: install Python 3.10+, then
   `git clone <repo> && cd poolforge && pip install -e .`, then run the command shown
   on the join page: `poolforge join CODE --server https://<your-app>.onrender.com`.
   No router settings are needed: everything is outbound HTTPS/WSS.

Without Python: build `poolforge-agent` with PyInstaller (`deploy/poolforge-agent.spec`,
build on Windows for a Windows `.exe`) and share the folder. Run it the same way:
`poolforge-agent join CODE --server URL`.

## Same Wi-Fi / Tailscale (faster)

Start every agent with `--transport tcp`. The coordinator then tells the agents to
send tensors directly to each other (about 25% faster in our benchmark). With
Tailscale, add `--advertise-host <tailscale IP>`. Firewalls must allow the agent's
TCP port (`--tcp-port` to fix it).
