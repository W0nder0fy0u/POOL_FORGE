"""Allow-list of apps the Deploy tab may run on a pool node.

Security rule: the network can only ever say "run app X" where X is a key of
APPS. Agents look the id up here; there is no way to send a command line,
image name or script from the dashboard. To offer a new app, edit this file
(on the agent machines) and restart the agents.

Two kinds of apps:
  * "sample": a small sample app shipped in deploy/sample_apps/, started as
              a Python subprocess (works without Docker).
  * "docker": an allow-listed public image, run with `docker run`.
"""

from typing import Any, Dict, List

APPS: Dict[str, Dict[str, Any]] = {
    "static_site": {
        "name": "Static website",
        "description": "Sample static site (deploy/sample_apps/static_site), served by Python.",
        "kind": "sample",
        "sample": "static_site",
    },
    "flask_app": {
        "name": "Flask web app",
        "description": "Sample Flask app (deploy/sample_apps/flask_app); needs `pip install flask` on the node.",
        "kind": "sample",
        "sample": "flask_app",
    },
    "nginx": {
        "name": "Nginx (Docker)",
        "description": "Official nginx:1.27-alpine image serving its welcome page.",
        "kind": "docker",
        "image": "nginx:1.27-alpine",
        "container_port": 80,
    },
    "whoami": {
        "name": "whoami (Docker)",
        "description": "traefik/whoami:v1.10 - tiny server that echoes request info.",
        "kind": "docker",
        "image": "traefik/whoami:v1.10",
        "container_port": 80,
    },
}


def get_app(app_id: str) -> Dict[str, Any]:
    """Return the app config, or raise PermissionError if it isn't allow-listed."""
    if app_id not in APPS:
        raise PermissionError(f"App '{app_id}' is not allow-listed. Allowed: {sorted(APPS)}")
    return APPS[app_id]


def catalog() -> List[Dict[str, Any]]:
    return [{"app_id": k, "name": v["name"], "description": v["description"], "kind": v["kind"]} for k, v in APPS.items()]
