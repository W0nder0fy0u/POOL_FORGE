"""PoolForge Agent CLI Interface.

Usage:
    poolforge join <ROOM_CODE> --server http://localhost:8000
    poolforge join <ROOM_CODE> --server http://localhost:8000 --ram-cap-gb 2.0 --transport tcp
"""

import argparse
import asyncio
import logging
import sys
from agent.worker import AgentWorker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="poolforge",
        description="PoolForge Agent CLI: pool laptop memory & compute into a virtual AI cluster.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-command to execute")

    # Command: join
    join_parser = subparsers.add_parser("join", help="Join a collaborative pool room")
    join_parser.add_argument("room", help="6-character room code")
    join_parser.add_argument(
        "--server",
        default="http://localhost:8000",
        help="Coordinator server URL (default: http://localhost:8000)",
    )
    join_parser.add_argument(
        "--ram-cap-gb",
        type=float,
        default=None,
        help="Simulate smaller laptop RAM capacity in GB (e.g. 1.0 or 2.0)",
    )
    join_parser.add_argument(
        "--transport",
        choices=["relay", "tcp"],
        default="relay",
        help="Data plane transport: 'relay' (default, WAN WebSocket) or 'tcp' (LAN direct socket)",
    )
    join_parser.add_argument(
        "--simulate-latency-ms",
        type=float,
        default=0.0,
        help="Simulate artificial network latency in milliseconds per step",
    )
    join_parser.add_argument(
        "--node-id",
        type=str,
        default=None,
        help="Custom identifier for this worker node",
    )

    # Command: version
    subparsers.add_parser("version", help="Print version information")

    args = parser.parse_args()

    if args.command == "version":
        print("PoolForge Agent CLI v0.1.0")
        return 0

    if args.command == "join":
        worker = AgentWorker(
            room_code=args.room,
            server_url=args.server,
            node_id=args.node_id,
            ram_cap_gb=args.ram_cap_gb,
            transport_type=args.transport,
            simulate_latency_ms=args.simulate_latency_ms,
        )
        print(f"PoolForge Agent starting... Connecting to {args.server} for room {args.room.upper()}")
        if args.ram_cap_gb:
            print(f"Simulating RAM cap: {args.ram_cap_gb:.1f} GB")
        try:
            asyncio.run(worker.start())
        except KeyboardInterrupt:
            print("\nShutting down PoolForge agent...")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
