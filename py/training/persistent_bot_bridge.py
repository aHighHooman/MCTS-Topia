from __future__ import annotations

import argparse
import socket
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Stdio bridge for the persistent Tribes RL bot server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    with socket.create_connection((args.host, args.port), timeout=30.0) as sock:
        reader = sock.makefile("r", encoding="utf-8", newline="\n")
        writer = sock.makefile("w", encoding="utf-8", newline="\n")
        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                continue
            writer.write(line + "\n")
            writer.flush()
            response = reader.readline()
            if not response:
                break
            print(response.strip(), flush=True)


if __name__ == "__main__":
    main()
