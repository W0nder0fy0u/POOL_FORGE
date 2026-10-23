# Deploying PoolForge

Only the **coordinator** is hosted. Agents run on each person's laptop and connect
out to it.

## Option 1: Docker on your own PC or server

```bash
docker compose up --build        # builds the Dockerfile, runs on http://localhost:8000
docker compose down              # stop (rooms/history kept in the poolforge-data volume)
```

Without compose: `docker build -t poolforge-coordinator .` then
`docker run -d -p 8000:8000 -v poolforge-data:/app/data poolforge-coordinator`.
The image is ~165 MB (no PyTorch: the coordinator never trains). Podman works the same
(`podman build` / `podman run`); with rootless podman use `http://127.0.0.1:8000`.

Optional environment variables: `POOLFORGE_PUBLIC_URL` (URL printed in join commands),
`POOLFORGE_REPO_URL` (git URL shown on the Join page), `POOLFORGE_DROP_GRACE_S`.
See `.env.example`.

## Option 2: Render (free, public HTTPS URL)

Pick ONE of these ways.

### 2-native. Plain Python, no Docker (simplest)

1. Push the repo to GitHub.
2. Render dashboard -> **New -> Web Service** -> pick the repo, then set:
   - **Language**: Python 3
   - **Build Command**: `pip install -r requirements-coordinator.txt`
   - **Start Command**: `python -m coordinator.server`
   - **Instance Type**: Free
   - **Environment**: `PYTHON_VERSION` = `3.12.8`, optionally `POOLFORGE_REPO_URL` = your GitHub URL
   - **Health Check Path** (Advanced): `/health`
   (Or **New -> Blueprint**: `render.yaml` in the repo root has exactly these settings.)
3. Open `https://<name>.onrender.com/health` -> `{"status":"ok"}`.

The dashboard is pre-built in `dashboard/dist`, so no Node.js step is needed. If you
change the dashboard, run `cd dashboard && npm run build` and commit `dist/`.

### 2a. Docker: Render builds the image from GitHub

1. Push the repo to GitHub.
2. Render dashboard -> **New -> Web Service** -> pick the repo -> **Language: Docker**
   (it builds the `Dockerfile` in the repo root). Set `POOLFORGE_REPO_URL` to your GitHub URL.
3. Wait for the build, then open `https://<name>.onrender.com/health` -> `{"status":"ok"}`.

### 2b. Docker: you build the image, Render just runs it

1. Create a free account on [hub.docker.com](https://hub.docker.com).
2. Build, self-test and push in one command (asks for your Docker Hub login once):
   ```bash
   bash scripts/docker_publish.sh <dockerhub-username>
   ```
   -> `docker.io/<dockerhub-username>/poolforge:latest`
3. Render dashboard -> **New -> Web Service -> Existing Image** -> paste that image URL.
   - Instance type: **Free**
   - Health check path: `/health` (Settings -> Health Checks)
   - Environment: optional `POOLFORGE_REPO_URL` = your GitHub URL
   Render sets `PORT` itself; the coordinator listens on it automatically.
4. To update later: run the publish script again, then on Render click
   **Manual Deploy -> Deploy latest reference**.

### Free plan notes
The service sleeps after ~15 minutes idle (the first visit wakes it in ~30 s), and
there is no persistent disk, so rooms reset on redeploy or restart. That's fine for a
demo; create the pool on the day you test.

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
