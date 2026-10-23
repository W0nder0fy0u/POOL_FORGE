# Deploying the coordinator on Railway

1. Push this repository to GitHub.
2. On [railway.app](https://railway.app): **New Project -> Deploy from GitHub repo** -> pick it.
3. In the service **Settings**:
   - **Builder**: Dockerfile. Set the variable `RAILWAY_DOCKERFILE_PATH=deploy/Dockerfile`
     (Variables tab) so Railway uses our Dockerfile.
   - **Networking -> Generate Domain** to get `https://<name>.up.railway.app`.
   - Optional: **Volumes -> Add volume** mounted at `/app/data` so rooms and run
     history survive restarts.
4. Railway injects `PORT`; the coordinator listens on it automatically.
5. Open the domain, click **Create Pool**, and share the join link. Friends run
   `poolforge join <CODE> --server https://<name>.up.railway.app`.

Health check path: `/health`. WebSockets need no extra settings.
