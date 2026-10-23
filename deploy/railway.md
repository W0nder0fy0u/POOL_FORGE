# Deploying PoolForge Coordinator on Railway

[Railway](https://railway.app) provides a free-tier PaaS that can host the PoolForge
coordinator with a persistent SQLite disk in under five minutes.

---

## One-Click Deployment Steps

### 1. Fork / Push the Repository
Push the PoolForge repo to GitHub (or GitLab).

### 2. Create a New Railway Project
1. Go to [railway.app](https://railway.app) and sign in with GitHub.
2. Click **New Project → Deploy from GitHub repo**.
3. Select your PoolForge fork.

### 3. Set the Start Command
In the Railway service settings set:

```
Start Command: uvicorn coordinator.server:app --host 0.0.0.0 --port $PORT
```

Railway automatically injects `$PORT`; the coordinator respects it.

### 4. Add a Persistent Volume (for SQLite)
1. In the service panel click **Add Volume**.
2. Set mount path to `/app/data`.
3. The coordinator writes `poolforge.db` under `/app/data` automatically (controlled
   by the `DATABASE_URL` environment variable below).

> **Note**: Without a persistent volume, the SQLite database is wiped on every
> redeploy and all room/job history is lost.  For a stateless demo you can skip this
> step; for anything longer-lived add the volume.

### 5. Configure Environment Variables

| Variable | Required | Description |
|---|---|---|
| `DATABASE_URL` | No (default: `sqlite:///./data/poolforge.db`) | SQLite path or future Postgres URL |
| `SECRET_KEY` | Recommended | Random string for token signing; Railway can generate one |
| `LOG_LEVEL` | No (default: `INFO`) | Python logging level |
| `PORT` | **Auto-injected by Railway** | Do not set manually |

To add variables: **Settings → Variables → Add Variable**.

### 6. Deploy
Click **Deploy**. Railway will:
1. Run `pip install -r requirements.txt` (or use `deploy/render.yaml`-style custom commands).
2. Start `uvicorn coordinator.server:app …`.
3. Provide a public HTTPS URL like `https://poolforge-coordinator.up.railway.app`.

### 7. Connect Agents
On participant laptops:
```bash
poolforge join <ROOM_CODE> --server https://poolforge-coordinator.up.railway.app
```

---

## Cost Notes
- Railway **free tier**: 500 compute hours/month and 1 GB ephemeral storage — sufficient
  for demos and viva sessions.
- Persistent volumes start at \$0.25 / GB / month on the Hobby plan.
- The coordinator itself uses ~50 MB RAM; the free tier handles it comfortably.
