"""Dispatcher gates: bindings, arming, cooldown (PRD 4.2, 4.5).

This is where an unintended action would actually escape, so the tests are
written as "what must never happen" rather than "what the code does".
"""

from __future__ import annotations

import pytest
from conftest import FakeClock

from airwave.config import parse_config
from airwave.dispatch.actions import ActionExecutor, Controls
from airwave.dispatch.dispatcher import (
    OUTCOME_ARMED,
    OUTCOME_COOLDOWN,
    OUTCOME_FIRED,
    OUTCOME_PAUSED,
    OUTCOME_UNARMED,
    OUTCOME_UNBOUND,
    Dispatcher,
)
from airwave.dispatch.platform import InputBackend
from airwave.events import Event, EventKind


def build(settings=None, bindings=None, clock=None):
    cfg, issues = parse_config({
        "settings": {"audio": {"enabled": False}, **(settings or {})},
        "bindings": bindings if bindings is not None else [
            {"name": "Play", "trigger": {"type": "gesture", "value": "open_palm"},
             "action": {"type": "key", "keys": ["playpause"]}},
        ],
    })
    assert not issues, issues
    backend = InputBackend(dry_run=True)
    dispatcher = Dispatcher(cfg, executor=ActionExecutor(backend, allow_shell=cfg.settings.allow_shell),
                            clock=clock or FakeClock())
    return dispatcher, backend


def gesture(value="open_palm", at=0.0):
    return Event(kind=EventKind.GESTURE, value=value, captured_at=at, emitted_at=at)


def test_a_bound_gesture_fires_its_action():
    dispatcher, backend = build()
    record = dispatcher.handle(gesture())
    assert record.outcome == OUTCOME_FIRED
    assert backend.log == ["press playpause x1"]


def test_an_unbound_gesture_does_nothing():
    """Most of the vocabulary is unbound; that is normal, not an error."""
    dispatcher, backend = build()
    assert dispatcher.handle(gesture("spock")).outcome == OUTCOME_UNBOUND
    assert backend.log == []


def test_a_disabled_binding_does_not_fire():
    dispatcher, backend = build(bindings=[
        {"name": "Play", "enabled": False, "trigger": {"type": "gesture", "value": "open_palm"},
         "action": {"type": "key", "keys": ["playpause"]}},
    ])
    assert dispatcher.handle(gesture()).outcome == OUTCOME_UNBOUND
    assert backend.log == []


def test_cooldown_suppresses_a_repeat(clock):
    dispatcher, backend = build({"cooldown_ms": 800}, clock=clock)
    assert dispatcher.handle(gesture()).outcome == OUTCOME_FIRED
    clock.advance(0.3)
    assert dispatcher.handle(gesture()).outcome == OUTCOME_COOLDOWN
    assert len(backend.log) == 1


def test_cooldown_expires(clock):
    dispatcher, backend = build({"cooldown_ms": 800}, clock=clock)
    dispatcher.handle(gesture())
    clock.advance(0.9)
    assert dispatcher.handle(gesture()).outcome == OUTCOME_FIRED
    assert len(backend.log) == 2


def test_cooldown_is_per_trigger_not_global(clock):
    """Volume-up must not be blocked by having just pressed play."""
    dispatcher, _ = build({"cooldown_ms": 800}, bindings=[
        {"name": "Play", "trigger": {"type": "gesture", "value": "open_palm"},
         "action": {"type": "key", "keys": ["playpause"]}},
        {"name": "Vol", "trigger": {"type": "gesture", "value": "one"},
         "action": {"type": "key", "keys": ["volumeup"]}},
    ], clock=clock)
    assert dispatcher.handle(gesture("open_palm")).outcome == OUTCOME_FIRED
    assert dispatcher.handle(gesture("one")).outcome == OUTCOME_FIRED


def test_per_binding_cooldown_overrides_the_global_one(clock):
    dispatcher, _ = build({"cooldown_ms": 100}, bindings=[
        {"name": "Play", "cooldown_ms": 2000, "trigger": {"type": "gesture", "value": "open_palm"},
         "action": {"type": "key", "keys": ["playpause"]}},
    ], clock=clock)
    dispatcher.handle(gesture())
    clock.advance(0.5)
    assert dispatcher.handle(gesture()).outcome == OUTCOME_COOLDOWN


def test_cooldown_remaining_counts_down(clock):
    dispatcher, _ = build({"cooldown_ms": 1000}, clock=clock)
    dispatcher.handle(gesture())
    clock.advance(0.4)
    assert dispatcher.cooldown_remaining(("gesture", "open_palm")) == pytest.approx(0.6, abs=1e-6)


# ------------------------------------------------------------------- arming


def test_nothing_fires_before_arming(clock):
    dispatcher, backend = build({"arming": {"enabled": True}}, clock=clock)
    assert dispatcher.handle(gesture()).outcome == OUTCOME_UNARMED
    assert backend.log == []


def test_arming_opens_a_window_that_expires(clock):
    dispatcher, backend = build({"arming": {"enabled": True, "timeout_s": 5}}, clock=clock)
    dispatcher.handle(Event(kind=EventKind.SYSTEM, value="arm"))
    assert dispatcher.handle(gesture()).outcome == OUTCOME_FIRED

    clock.advance(6.0)
    assert dispatcher.handle(gesture("open_palm")).outcome == OUTCOME_UNARMED
    assert len(backend.log) == 1


def test_arm_event_is_recorded_as_such():
    dispatcher, _ = build({"arming": {"enabled": True}})
    assert dispatcher.handle(Event(kind=EventKind.SYSTEM, value="arm")).outcome == OUTCOME_ARMED


def test_disarm_on_fire_gives_one_action_per_arm(clock):
    dispatcher, _ = build({"arming": {"enabled": True, "disarm_on_fire": True}}, bindings=[
        {"name": "Play", "trigger": {"type": "gesture", "value": "open_palm"},
         "action": {"type": "key", "keys": ["playpause"]}},
        {"name": "Vol", "trigger": {"type": "gesture", "value": "one"},
         "action": {"type": "key", "keys": ["volumeup"]}},
    ], clock=clock)
    dispatcher.arm()
    assert dispatcher.handle(gesture("open_palm")).outcome == OUTCOME_FIRED
    assert dispatcher.handle(gesture("one")).outcome == OUTCOME_UNARMED


def test_a_binding_can_opt_out_of_arming(clock):
    """An accessibility user may want one always-live escape hatch."""
    dispatcher, _ = build({"arming": {"enabled": True}}, bindings=[
        {"name": "Panic", "requires_arm": False, "trigger": {"type": "gesture", "value": "open_palm"},
         "action": {"type": "key", "keys": ["escape"]}},
    ], clock=clock)
    assert dispatcher.handle(gesture()).outcome == OUTCOME_FIRED


def test_arming_off_means_everything_is_armed():
    dispatcher, _ = build({"arming": {"enabled": False}})
    assert dispatcher.is_armed() is True


# -------------------------------------------------------------------- speech


def test_speech_matches_a_near_miss_transcript():
    """FR-3.5: ASR hears 'volume app'; the user said 'volume up'."""
    dispatcher, backend = build({"speech": {"enabled": True}}, bindings=[
        {"name": "Vol", "trigger": {"type": "speech", "value": "volume up"},
         "action": {"type": "key", "keys": ["volumeup"]}},
    ])
    record = dispatcher.handle(Event(kind=EventKind.SPEECH, value="volume app"))
    assert record.outcome == OUTCOME_FIRED
    assert backend.log == ["press volumeup x1"]


def test_speech_rejects_an_unrelated_phrase():
    dispatcher, backend = build({"speech": {"enabled": True}}, bindings=[
        {"name": "Vol", "trigger": {"type": "speech", "value": "volume up"},
         "action": {"type": "key", "keys": ["volumeup"]}},
    ])
    assert dispatcher.handle(Event(kind=EventKind.SPEECH, value="what time is it")).outcome == OUTCOME_UNBOUND
    assert backend.log == []


def test_speech_similarity_threshold_is_honoured():
    dispatcher, _ = build({"speech": {"enabled": True, "similarity": 0.99}}, bindings=[
        {"name": "Vol", "trigger": {"type": "speech", "value": "volume up"},
         "action": {"type": "key", "keys": ["volumeup"]}},
    ])
    assert dispatcher.handle(Event(kind=EventKind.SPEECH, value="volume app")).outcome == OUTCOME_UNBOUND


# ------------------------------------------------------------------- pointer


def test_pointer_moves_bypass_bindings_and_cooldown(clock):
    dispatcher, backend = build(clock=clock)
    for x in (0.1, 0.2, 0.3):
        record = dispatcher.handle(Event(kind=EventKind.POINTER, value="move", payload={"x": x, "y": 0.5}))
        assert record.outcome == OUTCOME_FIRED
    assert len(backend.log) == 3


def test_pointer_clicks_keep_their_own_cooldown(clock):
    """One long pinch must not stutter into a double-click."""
    dispatcher, backend = build({"mouse": {"click_cooldown_ms": 400}}, clock=clock)
    assert dispatcher.handle(Event(kind=EventKind.POINTER, value="click")).outcome == OUTCOME_FIRED
    clock.advance(0.1)
    assert dispatcher.handle(Event(kind=EventKind.POINTER, value="click")).outcome == OUTCOME_COOLDOWN
    assert backend.log == ["click left x1"]


# --------------------------------------------------------------------- modes


def test_mode_action_toggles_a_runtime_switch():
    cfg, _ = parse_config({
        "settings": {"audio": {"enabled": False}},
        "bindings": [{"name": "Mouse", "trigger": {"type": "gesture", "value": "rock"},
                      "action": {"type": "mode", "target": "mouse", "verb": "toggle"}}],
    })
    state = {"mouse": False}
    controls = Controls(set_mouse=lambda v: state.__setitem__("mouse", v), get_mouse=lambda: state["mouse"])
    dispatcher = Dispatcher(cfg, executor=ActionExecutor(InputBackend(dry_run=True), controls=controls),
                            clock=FakeClock())
    dispatcher.handle(gesture("rock"))
    assert state["mouse"] is True


def test_pause_blocks_everything_except_the_unpause_binding(clock):
    dispatcher, backend = build(bindings=[
        {"name": "Play", "trigger": {"type": "gesture", "value": "open_palm"},
         "action": {"type": "key", "keys": ["playpause"]}},
        {"name": "Resume", "trigger": {"type": "gesture", "value": "rock"},
         "action": {"type": "mode", "target": "pause", "verb": "toggle"}},
    ], clock=clock)
    dispatcher.paused = True
    assert dispatcher.handle(gesture("open_palm")).outcome == OUTCOME_PAUSED
    assert dispatcher.handle(gesture("rock")).outcome == OUTCOME_FIRED


# ---------------------------------------------------------------- bookkeeping


def test_stats_and_records_track_every_decision(clock):
    dispatcher, _ = build({"cooldown_ms": 5000}, clock=clock)
    dispatcher.handle(gesture())
    dispatcher.handle(gesture())
    dispatcher.handle(gesture("spock"))
    assert dispatcher.stats.fired == 1
    assert dispatcher.stats.cooldown == 1
    assert dispatcher.stats.unbound == 1
    assert [r.outcome for r in dispatcher.records] == [OUTCOME_FIRED, OUTCOME_COOLDOWN, OUTCOME_UNBOUND]


def test_latency_is_measured_from_capture_not_dispatch(clock):
    dispatcher, _ = build(clock=clock)
    captured = clock.now
    clock.advance(0.05)
    record = dispatcher.handle(gesture(at=captured))
    assert record.latency_ms == pytest.approx(50, abs=1)


def test_a_failing_action_is_recorded_not_raised():
    """One bad binding must not kill the dispatcher thread."""
    cfg, _ = parse_config({
        "settings": {"audio": {"enabled": False}},
        "bindings": [{"name": "Bad", "trigger": {"type": "gesture", "value": "one"},
                      "action": {"type": "key", "keys": ["nonexistent_key"]}}],
    })

    class Exploding(InputBackend):
        def press(self, *a, **k):
            raise RuntimeError("boom")

    dispatcher = Dispatcher(cfg, executor=ActionExecutor(Exploding(dry_run=True)), clock=FakeClock())
    record = dispatcher.handle(gesture("one"))
    assert record.outcome == "error"
    assert dispatcher.stats.errors == 1


def test_queue_submit_and_drain_round_trip():
    dispatcher, backend = build()
    dispatcher.submit(gesture())
    records = dispatcher.drain()
    assert [r.outcome for r in records] == [OUTCOME_FIRED]


def test_pointer_events_are_dropped_rather_than_queued_when_full():
    """A stale cursor position is worse than a missing one."""
    dispatcher, _ = build()
    for _ in range(dispatcher.queue.maxsize + 10):
        dispatcher.submit(Event(kind=EventKind.POINTER, value="move", payload={"x": 0.5, "y": 0.5}))
    assert dispatcher.queue.qsize() <= dispatcher.queue.maxsize


def test_hot_reload_keeps_cooldowns(clock):
    """Saving the config must not become a way to double-fire an action."""
    dispatcher, _ = build({"cooldown_ms": 2000}, clock=clock)
    dispatcher.handle(gesture())
    cfg, _ = parse_config({
        "settings": {"cooldown_ms": 2000, "audio": {"enabled": False}},
        "bindings": [{"name": "Play", "trigger": {"type": "gesture", "value": "open_palm"},
                      "action": {"type": "key", "keys": ["playpause"]}}],
    })
    dispatcher.apply_config(cfg)
    clock.advance(0.2)
    assert dispatcher.handle(gesture()).outcome == OUTCOME_COOLDOWN
