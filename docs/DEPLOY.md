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
`POOLFORGE_DROP_GRACE_S`, and `POOLFORGE_TURN_URL` / `_USER` / `_PASS` (TURN relay for
voice calls on strict networks; without it calls use free Google STUN only).
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
   - **Environment**: `PYTHON_VERSION` = `3.12.8`
   - **Health Check Path** (Advanced): `/health`
   (Or **New -> Blueprint**: `render.yaml` in the repo root has exactly these settings.)
3. Open `https://<name>.onrender.com/health` -> `{"status":"ok"}`.

Friends never need the GitHub repo (keep it private): they download the PoolForge app
from the Join page. See [Desktop app](#desktop-app-what-friends-install) to set up the downloads.

The dashboard is pre-built in `dashboard/dist`, so no Node.js step is needed. If you
change the dashboard, run `cd dashboard && npm run build` and commit `dist/`.

### 2a. Docker: Render builds the image from GitHub

1. Push the repo to GitHub.
2. Render dashboard -> **New -> Web Service** -> pick the repo -> **Language: Docker**
   (it builds the `Dockerfile` in the repo root).
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
   Render sets `PORT` itself; the coordinator listens on it automatically.
4. To update later: run the publish script again, then on Render click
   **Manual Deploy -> Deploy latest reference**.

### Free plan notes
The service sleeps after ~15 minutes idle (the first visit wakes it in ~30 s), and
there is no persistent disk, so rooms reset on redeploy or restart. That's fine for a
demo; create the pool on the day you test.

## Option 3: Railway

See [`deploy/railway.md`](../deploy/railway.md).

## Desktop app (what friends install)

Friends download one file (about 20 MB), open it, paste the join link and click **Join**.
Nothing else to install: on first run the app downloads a private Python and PyTorch
(the CUDA build if it finds an NVIDIA GPU, about 2.5 GB; otherwise the CPU build,
about 200 MB) into `%LOCALAPPDATA%\PoolForge` or `~/.local/share/poolforge`. Later runs
start in seconds. Unticking **Share my NVIDIA GPU** in the app skips the GPU download and
keeps the GPU out of the pool (the agent runs with `--device cpu`); an already installed
GPU build is reused. `POOLFORGE_TORCH=cpu` forces the CPU build.

The agent inside the app is compiled to native code with Nuitka (no `.py` files, no
docstrings), and the coordinator is never included. That stops casual reading of
your code, but like any program on someone else's computer it can still be
reverse-engineered by an expert. The coordinator code only ever runs on your server.

**Build and publish** (the Windows `.exe` must be built on Windows, so GitHub does it):

1. Follow the steps at the top of `.github/workflows/release-app.yml` once: a public
   `poolforge-app` repo for the downloads, a token, and two Actions variables.
2. `git tag v1.0.1 && git push origin v1.0.1` builds `PoolForge.exe` and
   `PoolForge-linux` and publishes them as a release of that public repo.
3. The home page and every Join page show the download buttons, on any device. They
   link to the latest release of `W0nder0fy0u/poolforge-app` (`APP_RELEASES` in
   `coordinator/server.py`). To point them somewhere else, set
   `POOLFORGE_APP_URL_WINDOWS` and `POOLFORGE_APP_URL_LINUX` on Render.

Local build (Python 3.12, e.g. `uv venv --python 3.12`):
`pip install nuitka==4.3.4 zstandard patchelf` then
`python deploy/build_app.py --server https://<your-app>.onrender.com` -> `dist-app/`.

The Windows app is not code-signed, so SmartScreen shows "Windows protected your PC"
the first time (More info -> Run anyway). Signing needs a paid certificate.

**Developers** can still use `pip install -e .` from the repo. Set
`POOLFORGE_SERVE_SOURCE=1` on the coordinator to re-enable the old
`/download/poolforge.tar.gz` pip install (it hands out the agent's readable source,
never the coordinator's).

## Friends on other networks

1. Deploy with option 2 or 3 to get a public HTTPS URL.
2. Open it, **Create Pool**, and send the join link.
3. Each friend downloads the app from the link, opens it and pastes the link.
   No router settings are needed: everything is outbound HTTPS/WSS.

## Same Wi-Fi / Tailscale (faster)

Start every agent with `--transport tcp`. The coordinator then tells the agents to
send tensors directly to each other (about 25% faster in our benchmark). With
Tailscale, add `--advertise-host <tailscale IP>`. Firewalls must allow the agent's
TCP port (`--tcp-port` to fix it).
