"""A stdlib HTTP server for the dashboard.

No web framework. The artefact's core depends on nothing beyond the standard
library and Ollama, and a demonstrator that adds Flask to make a Raspberry Pi
serve three pages would spend that claim for very little. ``http.server`` is
enough for one user on one machine, which is what this is for.

Bound to localhost by default. This serves an interface over an organisation's
internal documents; it has no authentication, and the privacy argument the
dissertation makes would sit badly with a demonstrator listening on every
interface.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from . import render
from .live import LiveAssistant, LiveUnavailable
from .replay import ReplayUnavailable, load_replay_library

#: The reported test split. Stated here so that the loader's check against the
#: run manifests is not the only statement of it; the gold question set is not
#: opened by the demonstrator and this number is not read from it.
TEST_QUESTIONS = 68


class DashboardState:
    """Lazily built, so that neither mode's absence disables the other.

    A machine with no Ollama must still be able to open the replay, and a
    machine whose frozen runs are missing must still be able to run live. Each
    side records why it is unavailable rather than raising at start-up.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.replay_error: str | None = None
        self.live_error: str | None = None
        self._live: LiveAssistant | None = None
        self._live_lock = threading.Lock()
        try:
            from ..evaluation.analysis import FROZEN_QUALITY_RUNS
            # The expected size of the test split is asserted here as well as
            # checked inside the loader against the manifests. Two independent
            # statements of the same number is the point: a run whose manifest
            # and answers agree on 40 questions would satisfy the loader's
            # internal check and fail this one.
            self.library = load_replay_library(root / "results" / "runs",
                                               FROZEN_QUALITY_RUNS,
                                               expected_questions=TEST_QUESTIONS)
        except (ReplayUnavailable, OSError, KeyError) as exc:
            self.library = None
            self.replay_error = str(exc)

    @property
    def replay_ready(self) -> bool:
        return self.library is not None

    def live(self) -> LiveAssistant | None:
        """Built on first use. Loading the index costs a second or two."""
        with self._live_lock:
            if self._live is None and self.live_error is None:
                try:
                    self._live = LiveAssistant.build(self.root)
                except Exception as exc:  # noqa: BLE001 - shown, not raised
                    self.live_error = str(exc)
            return self._live

    def live_status(self) -> dict[str, Any]:
        assistant = self.live()
        if assistant is None:
            return {"ready": False,
                    "detail": self.live_error or "live mode is unavailable"}
        return assistant.model_status()


class Handler(BaseHTTPRequestHandler):
    state: DashboardState

    server_version = "SMEAssistantDashboard/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        """Quiet by default.

        The default handler logs every request to stderr, which on a Pi during
        a demonstration is a stream of noise over the thing being demonstrated.
        """

    def _send(self, body: str, status: int = 200) -> None:
        payload = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        """Live questions arrive by POST.

        A question typed into a GET form ends up in the URL, and from there in
        the browser history and in any proxy or server log between here and the
        page. This serves an organisation's internal documents, and a query
        about someone's sick pay or disciplinary record does not belong in a
        log line. Replay stays on GET because its parameter is a question
        identifier from a fixed public list, and a shareable link to a
        particular comparison is useful.
        """
        parsed = urlparse(self.path)
        if parsed.path.rstrip("/") != "/live":
            self._send(render.page(
                "Not found", "Not found", "replay", "",
                "<header><h1>Not found</h1></header>"), 404)
            return
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length).decode("utf-8") if length else ""
        self._live(parse_qs(raw))

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        route = parsed.path.rstrip("/") or "/"
        if route == "/":
            status = self.state.live_status()
            self._send(render.landing(
                self.state.replay_ready, self.state.replay_error,
                bool(status.get("ready")), status.get("detail")))
        elif route == "/replay":
            self._replay(query)
        elif route == "/live":
            # A GET renders the empty form and executes nothing, whatever it
            # was given. Amendment 1.28 said live questions had moved to POST
            # when only the form's method attribute had changed: a question
            # pasted into the address bar was still answered, and still landed
            # in the browser history and every log between here and the page.
            # The query is discarded rather than echoed, because reflecting it
            # would put it back on the page it was kept off.
            self._live({}, executable=False)
        else:
            self._send(render.page(
                "Not found", "Not found", "replay", "",
                "<header><h1>Not found</h1><p class='muted'>"
                "<a href='/'>Back to the mode selector</a>.</p></header>"), 404)

    def _replay(self, query: dict[str, list[str]]) -> None:
        library = self.state.library
        if library is None:
            self._send(render.page(
                "Replay unavailable", "Frozen experimental replay", "replay",
                "The committed run records could not be read.",
                "<header><h1>Replay unavailable</h1><p class='muted'>"
                f"{self.state.replay_error or ''}</p>"
                "<p><a href='/'>Back to the mode selector</a>.</p></header>"),
                503)
            return
        ids = [q.question_id for q in library.questions]
        wanted = (query.get("q") or [ids[0]])[0]
        selected = library.by_id(wanted) or library.questions[0]
        self._send(render.replay_page(library, selected, ids))

    def _live(self, query: dict[str, list[str]], *,
              executable: bool = True) -> None:
        status = self.state.live_status()
        question = (query.get("q") or [""])[0].strip()
        answer, error = None, None
        if question and executable:
            if not status.get("ready"):
                error = str(status.get("detail") or "live mode is unavailable")
            else:
                try:
                    answer = self.state.live().answer(question)
                except (LiveUnavailable, Exception) as exc:  # noqa: BLE001
                    error = str(exc)
        if not executable:
            question = ""
        self._send(render.live_page(question or None, answer, error, status))


def build_server(root: Path | str, host: str = "127.0.0.1",
                 port: int = 8765) -> ThreadingHTTPServer:
    state = DashboardState(Path(root))
    handler = type("BoundHandler", (Handler,), {"state": state})
    return ThreadingHTTPServer((host, port), handler)
