# Architecture

Airwave is a small application with one hard problem in the middle of it. This
document is about where that problem lives and why the code is shaped around it.

## The problem

MediaPipe turns a frame into 21 hand landmarks. That part is a function call.
The product is everything after: turning a noisy 30fps stream of poses into
discrete commands that fire exactly once, when the user meant them.

The arithmetic is unforgiving. A classifier that is 95% accurate *per frame* is
wrong roughly 90 times a minute at 30fps. If each wrong frame could fire an
action, the tool is unusable — worse than unusable, because the false actions
are unpredictable. Every architectural decision below follows from driving that
number to zero without making the tool feel sluggish.

## Data flow

```
    ┌───────────────┐
    │ Camera thread │  (the main thread — cv2.imshow requires it on macOS)
    └───────┬───────┘
            │  capture ─ MediaPipe ─ normalize ─ classify ─ stabilize
            │
            ├────────────────────────► Event(kind, value, captured_at)
            │                                        │
    ┌───────┴───────┐                                │
    │ Audio thread  │  (sounddevice's own callback)  │
    └───────┬───────┘                                ▼
            │  RMS onset ─ clap grouping        ┌─────────┐
            │  VAD ─ [worker thread] ASR ─ wake │  Queue  │
            └────────────────────────►          └────┬────┘
                                                     │
                                          ┌──────────▼──────────┐
                                          │  Dispatcher thread  │
                                          │  paused?            │
                                          │  binding lookup     │
                                          │  armed?             │
                                          │  cooldown?          │
                                          │  → Action           │
                                          └─────────────────────┘
```

## Two rules

### Producers never act

The vision and audio paths have exactly one output: pushing a typed `Event` onto
a queue. They cannot press a key. Nothing in `vision/` or `audio/` imports
`pyautogui`.

This buys three things:

- **Testability.** `VisionPipeline.process()` takes landmarks and returns
  events. The entire vision path can be driven from a list of arrays, which is
  why the test suite runs in under a second with no webcam.
- **Substitutability.** Adding a third input modality — a foot pedal, a MIDI
  controller — is purely additive. It emits events; nothing else changes.
- **A migration path.** If profiling ever shows GIL contention, the vision stage
  moves to a `multiprocessing.Process` and the queue becomes a
  `multiprocessing.Queue`. The interface is already shaped for it.

### The dispatcher owns all side effects

Every keypress, mouse move and shell command originates in
`dispatch/dispatcher.py`, which is also the only component that needs to know
about OS differences (delegated to `dispatch/platform.py`).

One owner means one place where an unintended action can escape, and therefore
one place to put the gates.

## The false-positive gauntlet

An event must survive all of these to become an action:

| Gate | Rejects | Cost when wrong |
|---|---|---|
| **Stability window** (N frames agree) | Transient misreads; poses passed through on the way somewhere else | ~165ms of latency at N=5, 30fps |
| **Transition-only emission** | Holding a pose becoming thousands of events | none |
| **Neutral reset** | Ambiguity between consecutive commands | requires returning to `fist` |
| **Binding lookup** | Everything unbound — most of the vocabulary | none |
| **Arming** | Everything outside a deliberate few-second window | one arming gesture per session-ish |
| **Cooldown** | A repeat while the user is still lowering their hand | 800ms per trigger |
| **Hold** (`hold_ms`, per binding) | Anything the user did not mean to hold | the hold duration, on that binding only |
| **Motion gate** | Poses read off a hand that is travelling, not posing | poses ignored above the speed threshold |

Arming is the highest-leverage of these by a wide margin, because it collapses
an always-listening surface to a few seconds per session. It is off by default
only so the first run demos instantly — the README says so in both directions.

## Component map

| Path | Responsibility |
|---|---|
| `events.py` | The `Event` type. The only thing that crosses a thread boundary. |
| `gestures.py` | Vocabulary names. Dependency-free, so the validator can name every legal gesture without importing OpenCV. |
| `capture/camera.py` | `cv2.VideoCapture` wrapper. Backend selection, resolution negotiation, reconnect-with-backoff when a device disappears. |
| `capture/microphone.py` | `sounddevice.InputStream` wrapper. Guarantees the callback never raises into PortAudio. |
| `vision/normalize.py` | Translate → mirror → rotate → scale. The step that decides whether a classifier learns pose or position. |
| `vision/rules.py` | Geometric classifier. Handedness-free tests, per-finger hysteresis. |
| `vision/model.py` | Trained classifier (numpy MLP) behind the same interface, with an abstain threshold. |
| `vision/stabilizer.py` | The debounce state machine. |
| `vision/pointer.py` | One Euro filter, active-region mapping, dwell clicking. |
| `vision/swipe.py` | Dynamic gestures: sliding-window motion detection over the palm centre. |
| `vision/pipeline.py` | Composes the above into `landmarks → events`. |
| `fetch.py` | One-time model downloads, with a curl fallback for machines whose Python cannot verify TLS. |
| `audio/onset.py` | Adaptive-baseline RMS spike detector; clap/double-clap grouping. |
| `audio/speech.py` | VAD → worker-thread ASR → wake-word gate. |
| `audio/matching.py` | Fuzzy phrase matching. Pure functions. |
| `dispatch/dispatcher.py` | The gauntlet above, plus the event log. |
| `dispatch/actions.py` | Executors. Total functions that never raise into the thread. |
| `dispatch/platform.py` | Every OS-specific fact in the project. |
| `config/` | Schema, line-tracking loader, validation, hot reload. |
| `ui/overlay.py` | The debug overlay — the app's answer to "why did nothing happen?" |
| `ui/hub/` | The web dashboard: stdlib HTTP server, MJPEG video, SSE log. A second consumer of the same state the overlay reads. |
| `training/` | Recording mode, dataset format, training script. |

## Decisions worth explaining

**No pydantic, no watchdog, no scikit-learn.** The install budget is one
`pip install -r requirements.txt` on three operating systems. Hand-written
validation is what lets errors carry line numbers and spelling suggestions;
mtime-free polling replaces watchdog in twenty lines; the trained classifier is
a numpy MLP because the dataset is a few hundred samples of a dozen classes and
that is genuinely all it needs.

**Heavy imports are lazy.** `import airwave` pulls in neither OpenCV, MediaPipe,
pyautogui nor sounddevice. Config validation, the stabilizer, the dispatcher and
the whole test suite run on a machine with no camera and no display. `pyautogui`
in particular grabs a display connection and can trigger a macOS permission
prompt at import time, which does not belong in a library import.

**The hot reloader hashes file contents rather than checking mtime and size.**
The most common config edit is changing one digit, which leaves both unchanged.
Hashing a few kilobytes twice a second turns "sometimes reloads" into "always
reloads".

**Landmark datasets store raw landmarks, not feature vectors.** Features are a
function of code, and code changes. Storing raw keeps an improvement to
`normalize.py` one retrain away instead of one re-record away, and re-recording
is the expensive part because it needs a person's hands.

**The stabilizer emits `fist` transitions but the dispatcher usually ignores
them.** Keeping the neutral-pose policy in the binding table rather than the
state machine is what makes FR-1.4's "the config MAY override this" a
configuration question rather than a code change.

**Pointer events go through the queue like everything else**, even at 30/second,
because the alternative is the vision thread moving the cursor — which breaks
the one rule the design rests on. The queue drops pointer moves rather than
blocking when full: a stale cursor position is worse than a missing one.

**The hub is a consumer, not a control path.** It subscribes to
`Dispatcher.observers` — a read-only tap that receives records after the
decision is final — and reads frames the vision loop hands it. It cannot press
a key. This is the same boundary the overlay respects, which is why adding a
whole web UI required no change to the pipeline's semantics: one observer list,
one frame handoff, and the hub starting before the camera so a capture failure
becomes a rendered error rather than a dead process.

## Threading

Three threads, one process:

1. **Vision** — the main thread, because `cv2.imshow` must be on it on macOS.
2. **Audio** — sounddevice's own callback thread. It does RMS and buffering and
   nothing else; the ASR runs on a fourth, short-lived worker so a 400ms
   transcription cannot drop audio frames.
3. **Dispatch** — blocking `queue.get(timeout=…)` so it observes the stop event
   promptly.

Plus a config-watcher thread that only *stages* a new config; the main loop
applies it between frames, because swapping a classifier mid-frame is a race.

The GIL is not the bottleneck: MediaPipe and OpenCV release it during inference,
and the audio path is pure numpy.

## Testing strategy

- **Unit** — normalization invariance, classifier behaviour per pose, the
  stabilizer's exact-once semantics, config error content, cooldown and arming
  arithmetic, filter response.
- **Integration** — landmark sequences through `VisionPipeline` + `Dispatcher`,
  asserting the exact resulting keypresses.
- **Regression** — every fixed false positive becomes a sequence. `replay_video`
  runs the same assertions against a recorded MP4 when one is available.
- **Performance** — p95 of everything Airwave adds on top of MediaPipe, asserted
  under 5ms so the 50ms frame budget stays MediaPipe's to spend.

Hand poses are synthesized as landmark geometry rather than loaded from
recordings, so CI needs no camera, no model download and no fixtures — and a
failure points at the geometry that broke rather than at an opaque `.npz`.

## What v1 does not do

Two-handed gestures, dynamic gestures (swipes need a temporal model over
landmark sequences), continuous sign language, multi-user control, mobile
deployment, cloud sync, and input synthesis under Wayland.

The nearest additions, in the order the design already anticipates them:
richer sound classes (YAMNet) behind the existing `sound` trigger, and a small
temporal model behind the existing classifier interface.
