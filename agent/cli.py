"""`poolforge` command line.

    poolforge join X7K2AB --server https://my-pool.onrender.com
    poolforge join X7K2AB --server http://localhost:8000 --ram-cap-gb 1 --node-id laptop-a
    poolforge join X7K2AB --server http://192.168.1.5:8000 --transport tcp
    poolforge version
"""

import argparse
import asyncio
import logging
import os
import sys

from common import __version__


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="poolforge", description="PoolForge agent: share this laptop's RAM/CPU with a pool.")
    sub = p.add_subparsers(dest="command")

    j = sub.add_parser("join", help="join a pool and wait for work")
    j.add_argument("room", help="6-character room code from the join link")
    j.add_argument("--server", default=os.environ.get("POOLFORGE_SERVER", "http://localhost:8000"), help="coordinator URL")
    j.add_argument("--ram-cap-gb", type=float, default=None,
                   help="report at most this much free RAM (simulate a small laptop)")
    j.add_argument("--transport", choices=["relay", "tcp"], default=os.environ.get("POOLFORGE_TRANSPORT", "relay"),
                   help="relay (default, works anywhere) or tcp (direct, LAN/Tailscale only)")
    j.add_argument("--advertise-host", default=None, help="IP other nodes use to reach this one (tcp; default: auto)")
    j.add_argument("--tcp-port", type=int, default=0, help="TCP listen port (tcp; default: random free port)")
    j.add_argument("--simulate-latency-ms", type=float, default=0.0,
                   help="add this much one-way delay to every incoming tensor (benchmarks)")
    j.add_argument("--node-id", default=None, help="fixed node name (letters, digits, - _ .)")
    j.add_argument("--encrypt-passphrase", default=os.environ.get("POOLFORGE_PASSPHRASE"),
                   help="AES-encrypt tensors; every node in the room must use the same passphrase")
    j.add_argument("--threads", type=int, default=None, help="limit PyTorch CPU threads")
    j.add_argument("--workdir", default=".", help="where checkpoints/ and data/ are stored")
    j.add_argument("-v", "--verbose", action="store_true")

    sub.add_parser("version", help="print the version")

    # Used internally by the Deploy tab when running as a PyInstaller .exe.
    st = sub.add_parser("serve-static")
    st.add_argument("--port", type=int, required=True)
    st.add_argument("--bind", default="127.0.0.1")
    st.add_argument("--dir", required=True)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "version":
        print(f"PoolForge agent {__version__}")
        return 0
    if args.command == "serve-static":
        import functools
        from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

        handler = functools.partial(SimpleHTTPRequestHandler, directory=args.dir)
        ThreadingHTTPServer((args.bind, args.port), handler).serve_forever()
        return 0
    if args.command != "join":
        build_parser().print_help()
        return 1

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    if args.encrypt_passphrase:
        try:
            import cryptography  # noqa: F401
        except ImportError:
            print("--encrypt-passphrase needs: pip install cryptography", file=sys.stderr)
            return 2
    if args.threads:
        import torch

        torch.set_num_threads(args.threads)

    from agent.worker import Agent, JoinError

    agent = Agent(
        room=args.room, server=args.server, node_id=args.node_id, ram_cap_gb=args.ram_cap_gb,
        transport=args.transport, advertise_host=args.advertise_host, tcp_port=args.tcp_port,
        simulate_latency_ms=args.simulate_latency_ms, encrypt_passphrase=args.encrypt_passphrase,
        workdir=args.workdir,
    )
    print(f"PoolForge agent {__version__}: joining room {args.room.upper()} at {args.server}")
    if args.ram_cap_gb:
        print(f"  reporting at most {args.ram_cap_gb} GB free RAM (--ram-cap-gb)")
    print("  press Ctrl+C to leave the pool")
    try:
        asyncio.run(agent.run())
    except KeyboardInterrupt:
        print("\nLeft the pool. Bye!")
    except JoinError as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
