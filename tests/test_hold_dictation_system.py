"""Three capabilities added together: held bindings, dictation, and sleep.

They share a theme - each one does something a misread frame must not do. A
held binding refuses to fire until the user proves intent; dictation refuses to
type after it has been switched off; sleep refuses to block the dispatcher for
the length of a night.
"""

from __future__ import annotations

import pytest
from conftest import FakeClock, hand_for

from airwave.config import Action, parse_config
from airwave.dispatch.actions import ActionExecutor, Controls
from airwave.dispatch.dispatcher import (
    OUTCOME_FIRED,
    OUTCOME_HOLDING,
    OUTCOME_PAUSED,
    Dispatcher,
)
from airwave.dispatch.platform import InputBackend
from airwave.events import Event, EventKind
from airwave.vision.landmarks import HandFrame
from airwave.vision.pipeline import VisionPipeline

FPS = 30.0


def build(bindings, settings=None, clock=None, controls=None):
    cfg, issues = parse_config({
        "settings": {"stable_frames": 3, "audio": {"enabled": False}, **(settings or {})},
        "bindings": bindings,
    })
    assert not issues, issues
    backend = InputBackend(dry_run=True)
    executor = ActionExecutor(backend, controls=controls)
    return cfg, Dispatcher(cfg, executor=executor, clock=clock or FakeClock()), backend


SLEEP_BINDING = [{
    "name": "Sleep", "trigger": {"type": "gesture", "value": "rock"},
    "hold_ms": 3000, "action": {"type": "system", "target": "sleep"},
}]


# ---------------------------------------------------------------- hold gate


def test_a_held_binding_does_not_fire_on_the_plain_transition():
    """The whole point: recognizing the pose is not the same as meaning it."""
    _, dispatcher, backend = build(SLEEP_BINDING)
    record = dispatcher.handle(Event(kind=EventKind.GESTURE, value="rock"))
    assert record.outcome == OUTCOME_HOLDING
    assert "3.0s" in record.detail
    assert backend.log == []


def test_a_held_binding_fires_on_the_hold_event():
    _, dispatcher, backend = build(SLEEP_BINDING)
    record = dispatcher.handle(Event(kind=EventKind.GESTURE, value="rock", payload={"hold": True}))
    assert record.outcome == OUTCOME_FIRED
    assert backend.log == ["system sleep"]


def test_bindings_without_hold_are_unaffected():
    _, dispatcher, backend = build([{
        "name": "Play", "trigger": {"type": "gesture", "value": "open_palm"},
        "action": {"type": "key", "keys": ["playpause"]},
    }])
    assert dispatcher.handle(Event(kind=EventKind.GESTURE, value="open_palm")).outcome == OUTCOME_FIRED


def test_pipeline_emits_the_hold_event_only_after_the_full_duration():
    clock = FakeClock()
    cfg, _, _ = build(SLEEP_BINDING, clock=clock)
    pipeline = VisionPipeline(cfg, clock=clock)

    events = []
    for _ in range(20):  # well past stability, well short of 3 seconds
        events.extend(pipeline.process(HandFrame(hand_for("rock"), handedness="Right")).events)
        clock.advance(1 / FPS)
    assert not [e for e in events if e.payload.get("hold")], "fired before the hold elapsed"

    clock.advance(3.0)
    events.extend(pipeline.process(HandFrame(hand_for("rock"), handedness="Right")).events)
    held = [e for e in events if e.payload.get("hold")]
    assert len(held) == 1
    assert held[0].value == "rock"


def test_the_hold_event_fires_once_per_hold_not_every_frame():
    """Holding for ten seconds must not sleep the machine ten times."""
    clock = FakeClock()
    cfg, _, _ = build(SLEEP_BINDING, clock=clock)
    pipeline = VisionPipeline(cfg, clock=clock)

    events = []
    for _ in range(300):
        events.extend(pipeline.process(HandFrame(hand_for("rock"), handedness="Right")).events)
        clock.advance(1 / FPS)
    assert len([e for e in events if e.payload.get("hold")]) == 1


def test_releasing_and_re_holding_arms_a_second_hold():
    clock = FakeClock()
    cfg, _, _ = build(SLEEP_BINDING, clock=clock)
    pipeline = VisionPipeline(cfg, clock=clock)

    def hold(label, seconds):
        out = []
        for _ in range(int(seconds * FPS)):
            out.extend(pipeline.process(HandFrame(hand_for(label), handedness="Right")).events)
            clock.advance(1 / FPS)
        return out

    events = hold("rock", 4) + hold("fist", 1) + hold("rock", 4)
    assert len([e for e in events if e.payload.get("hold")]) == 2


def test_hold_progress_is_reported_for_the_ui():
    clock = FakeClock()
    cfg, _, _ = build(SLEEP_BINDING, clock=clock)
    pipeline = VisionPipeline(cfg, clock=clock)

    result = None
    for _ in range(45):  # 1.5s of a 3s hold
        result = pipeline.process(HandFrame(hand_for("rock"), handedness="Right"))
        clock.advance(1 / FPS)
    assert 0.3 < result.hold_progress < 0.7


def test_end_to_end_hold_then_sleep():
    """Landmarks to suspend, through both halves of the gate."""
    clock = FakeClock()
    cfg, dispatcher, backend = build(SLEEP_BINDING, clock=clock)
    pipeline = VisionPipeline(cfg, clock=clock)

    for i in range(150):  # 5 seconds
        for event in pipeline.process(HandFrame(hand_for("rock"), handedness="Right")).events:
            dispatcher.handle(event)
        clock.advance(1 / FPS)

    assert backend.log == ["system sleep"], "expected exactly one suspend"


def test_a_brief_rock_never_sleeps_the_machine():
    """The failure this whole feature exists to prevent."""
    clock = FakeClock()
    cfg, dispatcher, backend = build(SLEEP_BINDING, clock=clock)
    pipeline = VisionPipeline(cfg, clock=clock)

    for _ in range(20):  # two thirds of a second
        for event in pipeline.process(HandFrame(hand_for("rock"), handedness="Right")).events:
            dispatcher.handle(event)
        clock.advance(1 / FPS)
    assert backend.log == []


# ----------------------------------------------------------------- dictation


class FakeAudio:
    """Stands in for AudioProducer's dictation switch."""

    def __init__(self, *, has_listener=True):
        self.dictation = False
        self.has_listener = has_listener

    def set_dictation(self, active):
        if not self.has_listener:
            return False
        self.dictation = active
        return True


def dictation_setup(active=True):
    state = {"on": active}
    controls = Controls(
        set_dictation=lambda v: state.__setitem__("on", v),
        get_dictation=lambda: state["on"],
    )
    cfg, dispatcher, backend = build(
        [{"name": "Start", "trigger": {"type": "gesture", "value": "three"},
          "action": {"type": "mode", "target": "dictation", "verb": "on"}}],
        settings={"speech": {"enabled": True}},
        controls=controls,
    )
    return dispatcher, backend, state


def dictated(text):
    return Event(kind=EventKind.SPEECH, value=text, payload={"dictation": True})


def test_dictated_speech_is_typed_not_matched():
    dispatcher, backend, _ = dictation_setup(active=True)
    record = dispatcher.handle(dictated("hello there"))
    assert record.outcome == OUTCOME_FIRED
    assert backend.log == ["type 'hello there '"]


def test_a_trailing_space_is_added_between_phrases():
    dispatcher, backend, _ = dictation_setup(active=True)
    dispatcher.handle(dictated("first"))
    dispatcher.handle(dictated("second"))
    assert backend.log == ["type 'first '", "type 'second '"]


def test_the_trailing_space_can_be_turned_off():
    controls = Controls(get_dictation=lambda: True)
    _, dispatcher, backend = build(
        [], settings={"speech": {"enabled": True, "dictation_trailing_space": False}},
        controls=controls,
    )
    dispatcher.handle(dictated("no space"))
    assert backend.log == ["type 'no space'"]


def test_an_utterance_arriving_after_dictation_stopped_is_dropped():
    """The transcription worker can finish after the user said 'stop'. Typing
    it then would be a genuine surprise."""
    dispatcher, backend, _ = dictation_setup(active=False)
    record = dispatcher.handle(dictated("too late"))
    assert record.outcome == OUTCOME_PAUSED
    assert backend.log == []


def test_the_mode_binding_flips_dictation_on():
    dispatcher, backend, state = dictation_setup(active=False)
    dispatcher.handle(Event(kind=EventKind.GESTURE, value="three"))
    assert state["on"] is True


def test_non_dictated_speech_still_goes_through_the_binding_table():
    controls = Controls(get_dictation=lambda: True)
    _, dispatcher, backend = build(
        [{"name": "Switch", "trigger": {"type": "speech", "value": "switch window"},
          "action": {"type": "hotkey", "keys": ["alt", "tab"]}}],
        settings={"speech": {"enabled": True}},
        controls=controls,
    )
    dispatcher.handle(Event(kind=EventKind.SPEECH, value="switch window"))
    assert backend.log == ["hotkey alt+tab"]


def test_listener_in_dictation_mode_bypasses_the_wake_word():
    from airwave.audio.speech import SpeechListener, Utterance

    class Engine:
        name = "fake"

        def transcribe(self, pcm, samplerate):
            return "the quick brown fox"

    typed, matched = [], []
    listener = SpeechListener(
        engine=Engine(),
        on_phrase=lambda t, a: matched.append(t),
        on_dictation=lambda t, a: typed.append(t),
    )
    listener.dictation = True
    listener._handle(Utterance(pcm=None, started_at=0.0, ended_at=1.0))

    assert typed == ["the quick brown fox"]
    assert matched == [], "dictation must not also run phrase matching"


def test_listener_without_dictation_requires_the_wake_word():
    from airwave.audio.speech import SpeechListener, Utterance

    class Engine:
        name = "fake"

        def transcribe(self, pcm, samplerate):
            return "volume up"

    typed, matched = [], []
    listener = SpeechListener(
        engine=Engine(),
        on_phrase=lambda t, a: matched.append(t),
        on_dictation=lambda t, a: typed.append(t),
    )
    listener._handle(Utterance(pcm=None, started_at=0.0, ended_at=1.0))
    assert typed == [] and matched == []


def test_producer_reports_when_dictation_cannot_start():
    """Speech disabled means no listener; the caller has to be able to say so."""
    from airwave.audio.producer import AudioProducer

    cfg, _, _ = build([], settings={"speech": {"enabled": False}})
    producer = AudioProducer(cfg, lambda e: None)
    assert producer.set_dictation(True) is False


def test_producer_starts_dictation_when_a_listener_exists(monkeypatch):
    from airwave.audio import producer as producer_module
    from airwave.audio.producer import AudioProducer
    from airwave.audio.speech import NullEngine

    # Never build a real engine in tests: vosk fetches its model over the
    # network on first construction, and the suite must stay offline.
    monkeypatch.setattr(producer_module, "build_engine", lambda *a, **k: NullEngine())

    cfg, _, _ = build([], settings={"speech": {"enabled": True}})
    producer = AudioProducer(cfg, lambda e: None)
    assert producer.set_dictation(True) is True
    assert producer.listener.dictation is True
    assert producer.set_dictation(False) is True
    assert producer.listener.dictation is False


# -------------------------------------------------------------------- system


def test_sleep_action_does_not_need_allow_shell():
    """A bounded capability is a far smaller grant than arbitrary commands."""
    cfg, issues = parse_config({
        "settings": {"allow_shell": False, "audio": {"enabled": False}},
        "bindings": [{"name": "Sleep", "trigger": {"type": "gesture", "value": "rock"},
                      "hold_ms": 3000, "action": {"type": "system", "target": "sleep"}}],
    })
    assert issues == []


def test_unknown_system_target_is_rejected_with_options():
    _, issues = parse_config({
        "bindings": [{"name": "Boom", "trigger": {"type": "gesture", "value": "rock"},
                      "action": {"type": "system", "target": "shutdown"}}],
    })
    assert any("shutdown" in i.message for i in issues)


def test_sleep_returns_immediately():
    """It must not block the dispatcher until the machine wakes up."""
    backend = InputBackend(dry_run=True)
    result = ActionExecutor(backend).execute(Action("system", target="sleep"))
    assert result.ok
    assert result.duration_ms < 100


def test_hold_ms_is_rejected_on_non_gesture_triggers():
    _, issues = parse_config({
        "bindings": [{"name": "Clap", "trigger": {"type": "sound", "value": "clap"},
                      "hold_ms": 1000, "action": {"type": "noop"}}],
    })
    assert any("only applies to gesture triggers" in i.message for i in issues)


def test_yaml_on_off_are_accepted_as_mode_verbs():
    """YAML 1.1 turns bare on/off into booleans; refusing them would be
    technically correct and completely unhelpful."""
    from airwave.config import load_yaml

    doc = load_yaml("""
bindings:
  - name: Start
    trigger: { type: gesture, value: three }
    action: { type: mode, target: dictation, verb: on }
  - name: Stop
    trigger: { type: gesture, value: four }
    action: { type: mode, target: dictation, verb: off }
""")
    cfg, issues = parse_config(doc)
    assert issues == []
    assert [b.action.verb for b in cfg.bindings] == ["on", "off"]


def test_claps_are_suppressed_while_someone_is_speaking(monkeypatch):
    """Regression, found with real audio: a single dictated sentence produced
    six 'clap' events, because plosives are broadband transients that clear the
    clap threshold. With clap bound to Next track, talking skipped music."""
    import numpy as np

    from airwave.audio import producer as producer_module
    from airwave.audio.producer import AudioProducer
    from airwave.audio.speech import NullEngine

    monkeypatch.setattr(producer_module, "build_engine", lambda *a, **k: NullEngine())
    cfg, _, _ = build(
        [{"name": "Next", "trigger": {"type": "sound", "value": "clap"},
          "action": {"type": "key", "keys": ["nexttrack"]}}],
        settings={"speech": {"enabled": True}, "audio": {"enabled": True}},
    )
    events = []
    producer = AudioProducer(cfg, events.append)

    rng = np.random.default_rng(0)
    quiet = (rng.standard_normal(512) * 0.004).astype(np.float32)
    loud = (rng.standard_normal(512) * 0.6).astype(np.float32)

    now = 0.0
    for i in range(400):          # let the baseline settle on room tone
        now = i * 0.032
        producer.handle_block(quiet, now)

    # A transient with no speech around it: this is a clap and must fire.
    now += 0.032
    producer.handle_block(loud, now)
    assert [e.value for e in events] == ["clap"]

    events.clear()
    # Now the same transient while the VAD has an utterance open.
    for i in range(10):
        now += 0.032
        producer.handle_block(loud, now)
    assert producer.listener.vad.speaking, "fixture did not open an utterance"
    assert events == [], f"speech leaked through as sound events: {events}"


def test_dictation_mode_alone_suppresses_claps(monkeypatch):
    import numpy as np

    from airwave.audio import producer as producer_module
    from airwave.audio.producer import AudioProducer
    from airwave.audio.speech import NullEngine

    monkeypatch.setattr(producer_module, "build_engine", lambda *a, **k: NullEngine())
    cfg, _, _ = build([], settings={"speech": {"enabled": True}, "audio": {"enabled": True}})
    events = []
    producer = AudioProducer(cfg, events.append)
    producer.set_dictation(True)

    rng = np.random.default_rng(1)
    quiet = (rng.standard_normal(512) * 0.004).astype(np.float32)
    for i in range(400):
        producer.handle_block(quiet, i * 0.032)
    producer.handle_block((rng.standard_normal(512) * 0.6).astype(np.float32), 400 * 0.032)
    assert events == []
