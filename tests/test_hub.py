"""The web hub: shared state, HTTP surface, and the dispatcher tap.

Runs against a real server on an ephemeral port. No camera and no browser
needed - the frames are numpy arrays and the client is urllib.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from airwave.dispatch.dispatcher import DispatchRecord, Dispatcher
from airwave.dispatch.actions import ActionExecutor
from airwave.dispatch.platform import InputBackend
from airwave.events import Event, EventKind
from airwave.ui.hub.server import HubServer
from airwave.ui.hub.state import HubState


def record(kind=EventKind.GESTURE, value="open_palm", outcome="fired", **kw):
    return DispatchRecord(event=Event(kind=kind, value=value), outcome=outcome, **kw)


def frame(width=64, height=48):
    return np.full((height, width, 3), 40, dtype=np.uint8)


@pytest.fixture
def state():
    return HubState()


@pytest.fixture
def server(state):
    srv = HubServer(state, port=0).start()
    yield srv
    srv.stop()


def get(server, path, timeout=5):
    with urllib.request.urlopen(f"{server.url.rstrip('/')}{path}", timeout=timeout) as response:
        return response.status, response.read(), dict(response.headers)


# -------------------------------------------------------------------- state


def test_pointer_moves_are_kept_out_of_the_log(state):
    """They arrive 30x a second and would bury every real decision."""
    assert state.add_record(record(EventKind.POINTER, "move")) is None
    assert state.add_record(record(EventKind.POINTER, "click")) is not None
    assert len(state.recent()) == 1


def test_records_are_stamped_with_utc_wall_clock(state):
    """DispatchRecord.at is monotonic - meaningless as a time of day."""
    row = state.add_record(record())
    assert row["ts"].endswith("Z")
    assert len(row["time"].split(":")) == 3
    assert row["millis"].isdigit()


def test_row_carries_outcome_binding_and_latency(state):
    row = state.add_record(record(outcome="cooldown", binding="Play/pause",
                                  detail="612ms left", latency_ms=142.4))
    assert row["outcome"] == "cooldown"
    assert row["binding"] == "Play/pause"
    assert row["detail"] == "612ms left"
    assert row["latency_ms"] == 142.4


def test_confidence_is_carried_through_when_present(state):
    event = Event(kind=EventKind.GESTURE, value="one", payload={"confidence": 0.912})
    row = state.add_record(DispatchRecord(event=event, outcome="fired"))
    assert row["confidence"] == 0.91


def test_events_since_acts_as_a_cursor(state):
    for _ in range(3):
        state.add_record(record())
    rows, cursor = state.events_since(0)
    assert len(rows) == 3
    assert state.events_since(cursor)[0] == []


def test_the_ring_buffer_is_bounded(state):
    small = HubState(max_events=5)
    for _ in range(20):
        small.add_record(record())
    assert len(small.recent(100)) == 5


def test_jpeg_encoding_is_cached_per_frame(state):
    """Ten open tabs must not mean ten encodes of the same frame."""
    state.publish_frame(frame())
    first, seq_a = state.latest_jpeg()
    second, seq_b = state.latest_jpeg()
    assert first is not None
    assert first is second, "the second call should hit the cache"
    assert seq_a == seq_b

    state.publish_frame(frame())
    third, seq_c = state.latest_jpeg()
    assert seq_c > seq_a
    assert third is not None


def test_no_frame_means_no_jpeg(state):
    assert state.latest_jpeg()[0] is None
    assert state.has_video is False


def test_wait_for_event_returns_when_one_arrives(state):
    got = []

    def waiter():
        rows, _ = state.wait_for_event(0, timeout=3.0)
        got.extend(rows)

    thread = threading.Thread(target=waiter)
    thread.start()
    state.add_record(record())
    thread.join(timeout=4)
    assert len(got) == 1


def test_wait_for_event_times_out_quietly(state):
    rows, cursor = state.wait_for_event(0, timeout=0.05)
    assert rows == [] and cursor == 0


def test_wake_all_marks_the_hub_stopped(state):
    state.wake_all()
    assert state.status()["running"] is False


# ------------------------------------------------------------------- server


def test_index_and_assets_are_served(server):
    status, body, headers = get(server, "/")
    assert status == 200
    assert b"Airwave" in body
    assert headers["Content-Type"].startswith("text/html")

    for asset, marker in (("/static/style.css", b"--color-midnight-canvas"),
                          ("/static/app.js", b"EventSource")):
        status, body, _ = get(server, asset)
        assert status == 200 and marker in body


def test_api_state_reports_status_and_events(server, state):
    state.add_record(record(binding="Play/pause"))
    status, body, _ = get(server, "/api/state")
    payload = json.loads(body)
    assert status == 200
    assert payload["status"]["running"] is True
    assert payload["events"][0]["binding"] == "Play/pause"
    assert payload["status"]["server_time"].endswith("Z")


def test_unknown_paths_404(server):
    with pytest.raises(urllib.error.HTTPError) as exc:
        get(server, "/nope")
    assert exc.value.code == 404


def test_path_traversal_is_refused(server):
    """/static/ must not become a file browser for the whole disk."""
    for attack in ("/static/../../../../Windows/win.ini", "/static/../server.py"):
        with pytest.raises(urllib.error.HTTPError) as exc:
            get(server, attack)
        assert exc.value.code == 404


def test_server_binds_only_to_loopback(server):
    """This streams a webcam; it must never be reachable from the network."""
    assert server.url.startswith("http://127.0.0.1:")


def test_a_second_server_falls_back_to_a_free_port(state):
    first = HubServer(state, port=0).start()
    try:
        second = HubServer(HubState(), port=first.port).start()
        try:
            assert second.port != first.port
        finally:
            second.stop()
    finally:
        first.stop()


def test_event_stream_sends_backlog_then_stops_cleanly(server, state):
    state.add_record(record(binding="Play/pause"))
    chunks: list[bytes] = []

    def read_stream():
        try:
            with urllib.request.urlopen(f"{server.url}events", timeout=5) as response:
                for _ in range(4):
                    chunks.append(response.readline())
        except Exception:  # noqa: BLE001 - the socket closes underneath us
            pass

    reader = threading.Thread(target=read_stream, daemon=True)
    reader.start()
    reader.join(timeout=4)

    text = b"".join(chunks).decode("utf-8", "replace")
    assert "event: event" in text
    assert "Play/pause" in text


def test_video_stream_serves_multipart_frames(server, state):
    state.publish_frame(frame())
    with urllib.request.urlopen(f"{server.url}stream.mjpg", timeout=5) as response:
        assert "multipart/x-mixed-replace" in response.headers["Content-Type"]
        head = response.read(64)
    assert b"airwaveframe" in head


# --------------------------------------------------------- dispatcher tap


def test_dispatcher_notifies_observers(base_config):
    seen = []
    dispatcher = Dispatcher(base_config, executor=ActionExecutor(InputBackend(dry_run=True)))
    dispatcher.observers.append(seen.append)
    dispatcher.handle(Event(kind=EventKind.GESTURE, value="open_palm"))
    assert [r.outcome for r in seen] == ["fired"]


def test_a_broken_observer_cannot_break_dispatch(base_config):
    """A UI bug must not stop the user's keyboard shortcuts from working."""
    backend = InputBackend(dry_run=True)
    dispatcher = Dispatcher(base_config, executor=ActionExecutor(backend))

    def exploding(_record):
        raise RuntimeError("hub is on fire")

    dispatcher.observers.append(exploding)
    result = dispatcher.handle(Event(kind=EventKind.GESTURE, value="open_palm"))
    assert result.outcome == "fired"
    assert backend.log == ["press playpause x1"]


def test_state_can_subscribe_directly_to_a_dispatcher(base_config):
    """The wiring the app actually uses."""
    hub = HubState()
    dispatcher = Dispatcher(base_config, executor=ActionExecutor(InputBackend(dry_run=True)))
    dispatcher.observers.append(hub.add_record)
    dispatcher.handle(Event(kind=EventKind.GESTURE, value="open_palm"))
    dispatcher.handle(Event(kind=EventKind.GESTURE, value="spock"))

    rows = hub.recent()
    assert [row["outcome"] for row in rows] == ["fired", "unbound"]
    assert rows[0]["value"] == "open_palm"
