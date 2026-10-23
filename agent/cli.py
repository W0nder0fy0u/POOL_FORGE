"""PoolForge Agent CLI Entrypoint."""

import sys


def main():
    print("PoolForge Agent CLI v0.1.0")
    if len(sys.argv) > 1 and sys.argv[1] == "version":
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
