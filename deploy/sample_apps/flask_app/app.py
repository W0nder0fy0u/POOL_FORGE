"""Sample Lightweight Microservice for PoolForge Edge Hosting."""

import json
import os
import socket
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

PORT = int(os.environ.get("PORT", 5000))


class MicroserviceHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        payload = {
            "service": "poolforge-edge-microservice",
            "hostname": socket.gethostname(),
            "timestamp": time.time(),
            "status": "healthy",
            "message": "Hello from PoolForge edge hosting!",
        }
        self.wfile.write(json.dumps(payload, indent=2).encode("utf-8"))

    def log_message(self, format, *args):
        # Quiet standard output
        pass


def run():
    server = HTTPServer(("0.0.0.0", PORT), MicroserviceHandler)
    print(f"Microservice running on port {PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    run()
