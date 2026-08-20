"""Launch the post-evaluation dashboard.

    python scripts/dashboard.py                 # http://127.0.0.1:8765
    python scripts/dashboard.py --port 8080
    python scripts/dashboard.py --host 0.0.0.0  # reachable on the local network

Two modes, kept separate: Frozen Study Replay reads the committed experimental
records and needs no models at all; Live Assistant runs Arm D on a new question
and needs Ollama.

Governed by pre-registration amendment 1.27. This is a demonstrator built after
the evaluation. It contributes no evidence to any hypothesis, it does not write
into any frozen run directory, and nothing it shows in live mode has been
scored.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sme_assistant.demo.server import build_server  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1",
                        help="interface to bind. Defaults to localhost, because "
                             "this serves internal documents and has no "
                             "authentication.")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)

    server = build_server(ROOT, host=args.host, port=args.port)
    state = server.RequestHandlerClass.state  # type: ignore[attr-defined]
    print(f"  Dashboard on http://{args.host}:{args.port}")
    print(f"  Frozen replay : {'ready' if state.replay_ready else state.replay_error}")
    status = state.live_status()
    print(f"  Live assistant: {'ready' if status.get('ready') else status.get('detail')}")
    print("  Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
