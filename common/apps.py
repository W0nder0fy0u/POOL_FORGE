"""PoolForge Allow-Listed Edge Application Catalog.

Security Rule: Only allow-listed pre-approved applications may be launched by agents.
Arbitrary shell execution over the network is strictly prohibited.
"""

from typing import Any, Dict, List

ALLOW_LISTED_APPS: Dict[str, Dict[str, Any]] = {
    "static_site": {
        "id": "static_site",
        "name": "Static Portfolio & Showcase Site",
        "description": "Zero-dependency lightweight HTTP static file server.",
        "default_port": 8080,
        "entrypoint_type": "python_module",
        "module": "http.server",
        "docker_image": "python:3.11-alpine",
    },
    "flask_api": {
        "id": "flask_api",
        "name": "Flask Microservice API",
        "description": "REST API service returning node health and cluster status.",
        "default_port": 5000,
        "entrypoint_type": "python_script",
        "script_path": "deploy/sample_apps/flask_app/app.py",
        "docker_image": "python:3.11-slim",
    },
}


def get_allow_listed_app(app_id: str) -> Dict[str, Any]:
    """Retrieve app configuration if allow-listed; otherwise raise PermissionError."""
    app_cfg = ALLOW_LISTED_APPS.get(app_id)
    if not app_cfg:
        allowed = list(ALLOW_LISTED_APPS.keys())
        raise PermissionError(
            f"Unauthorized app '{app_id}'. Only allow-listed apps may be deployed: {allowed}"
        )
    return app_cfg
