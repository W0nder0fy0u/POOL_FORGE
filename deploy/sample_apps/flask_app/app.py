"""Sample Flask app for the PoolForge Deploy tab.

Shows which pool node is serving it and counts visits.
Run by hand:  python app.py --port 5000
"""

import argparse
import platform
import socket
import time

from flask import Flask, jsonify

app = Flask(__name__)
STARTED = time.time()
visits = 0

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>PoolForge Flask app</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{{font-family:system-ui,sans-serif;background:#0f172a;color:#e2e8f0;display:grid;place-items:center;min-height:100vh;margin:0}}
.card{{background:#1e293b;padding:2rem 2.5rem;border-radius:12px;max-width:420px}}h1{{color:#38bdf8;margin-top:0}}code{{color:#a5f3fc}}</style>
</head><body><div class="card"><h1>Hello from the pool!</h1>
<p>This Flask app is running on node <code>{host}</code> ({os}).</p>
<p>Visits: <b>{visits}</b> &middot; up {uptime:.0f}s</p>
<p><a style="color:#38bdf8" href="/api/info">/api/info</a> (JSON)</p></div></body></html>"""


@app.get("/")
def index():
    global visits
    visits += 1
    return PAGE.format(host=socket.gethostname(), os=platform.system(), visits=visits, uptime=time.time() - STARTED)


@app.get("/api/info")
def info():
    return jsonify(host=socket.gethostname(), os=platform.system(), visits=visits, uptime_s=round(time.time() - STARTED, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    app.run(host=args.host, port=args.port)
