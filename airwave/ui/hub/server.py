"""The hub's HTTP server - Python standard library only.

No FastAPI, no uvicorn: ``requirements.txt`` stays at six lines, which is a
stated product goal. What that costs is a thread per open connection, and what
it buys is that a clone of this repo runs the hub with nothing extra installed.
For a single-user localhost dashboard that trade is obviously correct.

**It binds to 127.0.0.1 and nothing else.** This server streams the user's
webcam. Binding to 0.0.0.0 would put a live video feed of their room on every
network they join, so the host is not configurable from the CLI.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import socket
import threading
import time
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .state import HubState

log = logging.getLogger("airwave.hub")

STATIC_DIR = Path(__file__).parent / "static"
HOST = "127.0.0.1"
DEFAULT_PORT = 8760
BOUNDARY = "airwaveframe"
STREAM_FPS = 20.0
"""Cap the MJPEG rate. The pipeline may run faster; a browser cannot usefully
show more, and every extra frame is a JPEG encode plus a socket write."""


class HubHandler(BaseHTTPRequestHandler):
    """One handler per request. Routes are explicit - there are only five."""

    protocol_version = "HTTP/1.1"
    server_version = "Airwave"
    sys_version = ""

    def __init__(self, *args, state: HubState, **kwargs) -> None:
        self.state = state
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------- plumbing

    def log_message(self, format: str, *args) -> None:
        """Silence the default stderr access log; it drowns the app's output."""
        log.debug("%s - %s", self.address_string(), format % args)

    def _send(self, body: bytes, content_type: str, *, status: int = 200, cache: str = "no-store") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload, *, status: int = 200) -> None:
        self._send(json.dumps(payload).encode("utf-8"), "application/json", status=status)

    # --------------------------------------------------------------- routing

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's API
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        try:
            if path == "/":
                self._serve_static("index.html")
            elif path.startswith("/static/"):
                self._serve_static(path[len("/static/"):])
            elif path == "/api/state":
                self._json({"status": self.state.status(), "events": self.state.recent(120)})
            elif path == "/stream.mjpg":
                self._stream_video()
            elif path == "/events":
                self._stream_events()
            else:
                self._json({"error": "not found", "path": path}, status=404)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            # The browser closed the tab mid-stream. Entirely normal.
            log.debug("client disconnected from %s", path)
        except Exception:  # noqa: BLE001 - a handler must never take down the server
            log.exception("hub request failed: %s", path)

    # ---------------------------------------------------------------- static

    def _serve_static(self, relative: str) -> None:
        target = (STATIC_DIR / relative).resolve()
        # Path traversal guard: a request for ../../secrets must not escape.
        if not str(target).startswith(str(STATIC_DIR.resolve())) or not target.is_file():
            self._json({"error": "not found"}, status=404)
            return
        content_type, _ = mimetypes.guess_type(target.name)
        self._send(target.read_bytes(), content_type or "application/octet-stream")

    # ----------------------------------------------------------------- video

    def _stream_video(self) -> None:
        """multipart/x-mixed-replace - the oldest trick that still just works.

        No JavaScript, no WebSocket, no codec negotiation: an <img> tag pointed
        at this URL renders a live feed in every browser.
        """
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
        self.end_headers()

        interval = 1.0 / STREAM_FPS
        last_seq = -1
        while True:
            status = self.state.status()
            if not status.get("running", True):
                return
            jpeg, seq = self.state.latest_jpeg()
            if jpeg is None:
                # No camera yet. Idle politely rather than spinning; the page
                # shows its own empty state from /api/state.
                time.sleep(0.25)
                continue
            if seq == last_seq:
                time.sleep(interval / 2)
                continue
            last_seq = seq
            self.wfile.write(b"--" + BOUNDARY.encode() + b"\r\n")
            self.wfile.write(b"Content-Type: image/jpeg\r\n")
            self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
            self.wfile.write(jpeg)
            self.wfile.write(b"\r\n")
            self.wfile.flush()
            time.sleep(interval)

    # ---------------------------------------------------------------- events

    def _stream_events(self) -> None:
        """Server-sent events: the log, plus a status heartbeat.

        SSE rather than WebSockets because the traffic is strictly one-way and
        the browser reconnects on its own if the app restarts.
        """
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        cursor = 0
        last_status = 0.0
        for row in self.state.recent(120):
            cursor = max(cursor, row["seq"])
            self._sse("event", row)

        while True:
            status = self.state.status()
            if not status.get("running", True):
                self._sse("bye", {"reason": "shutting down"})
                return
            rows, cursor = self.state.wait_for_event(cursor, timeout=1.0)
            for row in rows:
                self._sse("event", row)
            now = time.monotonic()
            if now - last_status >= 0.5:
                # Doubles as the SSE keepalive; proxies drop silent streams.
                self._sse("status", self.state.status())
                last_status = now

    def _sse(self, name: str, payload) -> None:
        body = json.dumps(payload)
        self.wfile.write(f"event: {name}\ndata: {body}\n\n".encode("utf-8"))
        self.wfile.flush()


class _Server(ThreadingHTTPServer):
    """ThreadingHTTPServer with address reuse switched off.

    socketserver enables SO_REUSEADDR by default. On Linux that only relaxes
    TIME_WAIT; on Windows it lets a *second* process bind a port another one is
    already listening on, after which requests are delivered to whichever socket
    the OS feels like. Two Airwave instances would then share a URL and neither
    would know. Refusing the bind is what makes the fallback below reachable.
    """

    allow_reuse_address = False
    daemon_threads = True


class HubServer:
    """Owns the socket and the serving thread."""

    def __init__(self, state: HubState, *, port: int = DEFAULT_PORT) -> None:
        self.state = state
        self.requested_port = port
        self.port = port
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{HOST}:{self.port}/"

    def start(self) -> "HubServer":
        handler = partial(HubHandler, state=self.state)
        if self.requested_port and not port_is_free(self.requested_port):
            log.warning("port %s is already serving something, picking a free one", self.requested_port)
            self._httpd = _Server((HOST, 0), handler)
        else:
            try:
                self._httpd = _Server((HOST, self.requested_port), handler)
            except OSError as exc:
                # Fall back rather than refusing to start: a second Airwave
                # instance should be an inconvenience, not a failure.
                log.warning("port %s unavailable (%s), picking a free one", self.requested_port, exc)
                self._httpd = _Server((HOST, 0), handler)
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever,
                                        name="airwave-hub", daemon=True)
        self._thread.start()
        log.info("hub serving at %s", self.url)
        return self

    def stop(self) -> None:
        self.state.wake_all()
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None


def port_is_free(port: int, host: str = HOST) -> bool:
    """True when nothing is listening on the port.

    Probes by *connecting*, not by binding. A bind test inherits the same
    SO_REUSEADDR ambiguity it is meant to detect; if a connection is accepted,
    something is definitively there.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.25)
        return probe.connect_ex((host, port)) != 0
