# Airwave

Control your computer with hand gestures and sound. A webcam watches your hands, a microphone listens for claps and spoken commands, and both feed one event pipeline that maps what you did to keystrokes, mouse movement, or shell commands.

Everything runs locally. No network calls, no cloud inference, no telemetry.

```bash
pip install -r requirements.txt
python -m airwave fetch-model   # one time, ~7.5 MB, only if your MediaPipe needs it
python -m airwave init
python -m airwave run
```

Hold up an open palm to play/pause. Hold up one finger for volume up.

---

## The actual problem

Detecting a hand is one function call — MediaPipe solved that. The hard part is everything after it: turning a noisy 30fps stream of hand poses into discrete commands that fire **exactly once**, when you meant them, and never otherwise.

A per-frame classifier that is 95% accurate is wrong about **ninety times a minute** at 30fps. Making that number zero is the whole product, and it is what most of this codebase is:

| Defense | What it stops | Where |
|---|---|---|
| **Stability window** | A hand passing through "two" on its way to "open palm" | [stabilizer.py](airwave/vision/stabilizer.py) |
| **Transition-only emission** | Holding a pose for a minute becoming 1,800 keypresses | [stabilizer.py](airwave/vision/stabilizer.py) |
| **Neutral pose** | Two different gestures blurring into one ambiguous transition | `fist` is reserved |
| **Cooldown** | A repeat firing while you are still lowering your hand | [dispatcher.py](airwave/dispatch/dispatcher.py) |
| **Arming** | Everything else — nothing fires unless you armed in the last few seconds | [dispatcher.py](airwave/dispatch/dispatcher.py) |
| **Finger hysteresis** | A borderline finger flickering and resetting the window every frame | [rules.py](airwave/vision/rules.py) |

Arming is off by default so the first run feels instant. **Turn it on for daily use** — it is the single highest-leverage setting in the file.

---

## Gestures

| Gesture | How | Default binding |
|---|---|---|
| `fist` | 0 fingers | *neutral — return here between commands* |
| `open_palm` | 5 fingers | Play/pause |
| `one` | index only | Volume up |
| `two` | index + middle | Volume down |
| `three` | 3 fingers | **Start dictation** — types what you say |
| `four` | 4 fingers, thumb tucked | **Stop dictation** |
| `thumbs_up` | thumb up, hand vertical | *arming gesture* |
| `pinch` | thumb touching index | Click, in mouse mode |
| `ok` | pinch + 3 fingers out | *unbound* |
| `rock` | index + pinky | **Sleep the computer** — hold 3s |
| `spock` | 4 fingers split in the middle | *unbound* |

Unbound gestures do nothing — that is normal, not an error. `python -m airwave gestures` prints this list with descriptions.

**Motions** are a separate trigger type, because a swipe is not a held pose:

| Motion | Default binding |
|---|---|
| `swipe_left` | Next slide (Right arrow) |
| `swipe_right` | Previous slide (Left arrow) |
| `swipe_up` / `swipe_down` | *unbound* |

Arrow keys drive Google Slides, PowerPoint and Keynote alike. While the hand is travelling fast enough to be swiping, pose recognition is suppressed — otherwise a swipe with an open palm would also fire Play/pause.

Sound triggers: `clap`, `double_clap`. Speech triggers: any phrase you configure, after a wake word.

---

## Configuration

Everything lives in `airwave.yaml` and **hot-reloads on save** — edit it while the app runs and the change takes effect on the next frame. Validate without starting the camera with `python -m airwave check`.

```yaml
settings:
  stable_frames: 5      # frames that must agree before a gesture counts
  cooldown_ms: 800      # per-trigger dead time after firing
  arming:
    enabled: true       # nothing fires unless armed within timeout_s
    gesture: thumbs_up
    timeout_s: 5

bindings:
  - name: "Play/pause"
    trigger: { type: gesture, value: open_palm }
    action:  { type: key, keys: [playpause] }

  - name: "Next track"
    trigger: { type: sound, value: clap }
    action:  { type: key, keys: [nexttrack] }

  - name: "Switch window"
    trigger: { type: speech, value: "switch window" }
    action:  { type: hotkey, keys: [alt, tab] }
```

**Triggers:** `gesture`, `sound`, `speech`.
**Actions:** `key`, `hotkey`, `mouse_move`, `mouse_click`, `scroll`, `shell`, `mode`, `noop`.

Mistakes are reported with the line number and a suggestion, all of them at once:

```
2 problems in airwave.yaml:
  airwave.yaml:14: bindings[0].trigger.value
      unknown gesture 'open_palmm'
      hint: did you mean 'open_palm'?
  airwave.yaml:19: bindings[1].action.command
      shell actions are disabled
      hint: set 'settings.allow_shell: true' to enable them - a shared config
            that runs arbitrary commands is a real attack surface, so this is opt-in
```

`shell` actions require `allow_shell: true`. A config file that executes commands is a real attack surface if you download someone else's — treat a shared `airwave.yaml` the way you would a shell script.

---

## Commands

| Command | What it does |
|---|---|
| `python -m airwave run` | The app. `--dry-run` logs actions instead of performing them |
| `python -m airwave hub` | The web dashboard: live feed on the left, event log on the right |
| `python -m airwave doctor` | Diagnoses a broken setup — permissions, devices, model, config |
| `python -m airwave check` | Validates the config and lists every binding |
| `python -m airwave devices` | Lists cameras and microphones with their indices |
| `python -m airwave fetch-speech-model` | Downloads the vosk model needed for dictation |
| `python -m airwave calibrate-audio` | Live RMS meter for picking clap thresholds for your room |
| `python -m airwave record <gesture>` | Records samples for a custom gesture |
| `python -m airwave train` | Trains the classifier on what you recorded |
| `python -m airwave replay <video.mp4>` | Runs a clip through the pipeline and prints the events |
| `python -m airwave bench` | Measures fps and p95 frame time against the targets |

While the preview window is focused: `q` quit, `a` arm now, `m` toggle mouse, `p` pause, `l` landmarks, `h` help.

---

## The web hub

```bash
python -m airwave hub --open
```

A local dashboard at `http://127.0.0.1:8760` — the camera feed with landmarks drawn on it filling the left, and every dispatcher decision streaming into a UTC-stamped log on the right. The navbar carries placeholders for Bindings, Gestures, Audio and Settings, which are the natural next screens.

It is served by Python's standard library, so it adds **no dependencies** — `requirements.txt` is unchanged. The video is MJPEG (an `<img>` tag, no player), the log is server-sent events, and JPEG encoding happens on the HTTP thread so an unopened browser costs the pipeline nothing.

**It binds to 127.0.0.1 only, and that is not configurable.** The hub streams your webcam; binding it to `0.0.0.0` would publish a live view of your room to every network you join.

The hub starts *before* the camera, deliberately: if the camera or hand model is missing, the page loads and tells you what to fix instead of the process dying. Audio triggers and the log keep working in that state.

Design: the [AuthKit](https://styles.refero.design/) system — midnight canvas, frosted-glass surfaces, blueprint grid, one violet accent. Its three licensed typefaces are substituted with system stacks rather than loaded from a font CDN, since Airwave makes no network calls; to get them exactly, drop the `.woff2` files into `airwave/ui/hub/static/fonts/` and add `@font-face` rules.

---

## The overlay tells you why nothing happened

Every gate that can swallow a gesture is drawn on the preview: the stability bar, the arming state and its countdown, the cooldown remaining, the live fps, and a log of the last few events with what the dispatcher did with each.

This matters more than it sounds. "Gesture misread" and "gesture correct but still in cooldown" look identical from the outside — both are *nothing happening*. If you can see the stability bar resetting, you know your pose is being classified as something else. If you see it fill and then read `cooldown`, you know to wait.

---

## Mouse control

Off by default. Enable with `settings.mouse.enabled: true`, or press `m`.

Your index fingertip drives the cursor, smoothed by a [One Euro filter](https://gery.casiez.net/1euro/) — heavy smoothing while your hand is nearly still, almost none while it moves fast. Raw landmark positions jitter a few pixels every frame; a cursor that vibrates cannot be aimed.

The middle 50% of the frame maps to the whole screen (`active_region`), so the corners are a wrist movement rather than a full arm extension. Pinch to click, or enable `dwell_click` to click by holding still.

---

## Dictation

`three` starts typing what you say at the cursor; `four` stops. While dictation is on the wake word is bypassed — everything you say is typed — so stop it before you talk to anyone.

```bash
pip install vosk
python -m airwave fetch-speech-model     # ~40MB, one time
```

Then set `settings.speech.enabled: true`. Claps are suppressed while you are speaking: plosives are broadband transients that clear the clap threshold, and without that a dictated sentence skipped six tracks.

## Holding a binding

Anything you would hate to trigger by accident can require the pose to be *held*:

```yaml
  - name: "Sleep the computer"
    trigger: { type: gesture, value: rock }
    hold_ms: 3000
    action:  { type: system, target: sleep }
```

The stability window proves the classifier agrees; `hold_ms` proves **you** meant it. The log shows `holding` with the remaining time while it counts. `system` actions are a bounded capability, so unlike `shell` they need no `allow_shell`.

## Custom gestures

The rule-based classifier handles the built-in vocabulary with zero setup. For a pose geometry cannot express — a sign, a letter, something personal — record it and train:

```bash
python -m airwave record my_sign --samples 60   # space to start, q to save
python -m airwave train
```

Then set `classifier: model` and add `my_sign` to `custom_gestures` in your config.

Training prints per-class counts and a confusion matrix, because "89% accuracy" is not debuggable but "`spock` is read as `four` eleven times" tells you exactly which pose to re-record. Samples are stored as **raw landmarks**, not feature vectors — an improvement to the normalization code is then a retrain away rather than a re-record, and re-recording is the expensive part because it needs your hands.

---

## Setup notes per OS

**macOS** — grant Accessibility permission (System Settings → Privacy & Security → Accessibility) or every keystroke is silently discarded and the app looks broken. `doctor` checks this. For media keys, `pip install pyobjc-framework-Quartz pyobjc-framework-Cocoa`.

**Windows** — works out of the box. Media keys and volume keys are native.

**Linux (X11)** — works. **Linux (Wayland)** — `pyautogui` cannot synthesize input under Wayland, by design. Log into an X11 session, or use `ydotool` via `shell` actions. `doctor` detects this and says so rather than failing silently.

Full details in [docs/SETUP.md](docs/SETUP.md).

---

## Architecture

```
┌──────────────┐
│ Camera thread│ capture → MediaPipe → normalize → classify → stabilize ──┐
└──────────────┘                                                          │
┌──────────────┐                                                          ├──> EventQueue
│ Audio thread │ stream → onset/VAD → clap | wake word + phrase ──────────┘        │
└──────────────┘                                                                   │
┌──────────────────────────────────────────────────────────────────────────────────┘
│
├──> Dispatcher: arming → cooldown → binding lookup → Action
│
└──> Overlay: live preview, gesture, stability, arming, cooldown, event log
```

Two rules hold the design together:

**Producers never act.** The camera and audio threads have exactly one output — pushing typed events onto a queue. They cannot press keys. That is what makes them testable in isolation (feed a recorded clip, assert the event sequence) and what makes a third input modality a purely additive change.

**The dispatcher owns all side effects.** Every keypress, mouse move and shell command originates in one place, which is also the only place that needs to know about OS differences.

More in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

331 tests, under ten seconds, **no camera, no microphone and no model download required**. Hand poses are synthesized as landmark geometry ([tests/conftest.py](tests/conftest.py)), so the whole vision path is exercised on CI hardware that has never seen a webcam.

The suite is organized around the failure modes, not the modules — `test_fist_is_not_read_as_pinch`, `test_a_flicker_shorter_than_the_window_never_fires`, `test_cooldown_is_per_trigger_not_global`. Every false positive that gets fixed leaves a test behind.

---

## Example configurations

Two starting points beyond the default, both validated by the test suite:

- **[examples/media-kitchen.yaml](examples/media-kitchen.yaml)** — media control with wet hands. Strict stability, long cooldowns, arming on with one-action-per-arm, and claps for track skipping since they need no clean hands at all.
- **[examples/accessibility.yaml](examples/accessibility.yaml)** — alternative input for limited fine motor control or RSI. Forgiving thresholds, a generous pinch, heavy cursor smoothing to absorb tremor, dwell clicking, and a long arming window so nothing is rushed.

```bash
python -m airwave run --config examples/accessibility.yaml
```

## Known limitations

- **One hand.** Two-handed gestures and multi-user control are out of scope for v1.
- **Static poses only.** No swipes or dynamic gestures — those need a temporal model.
- **Wayland cannot receive synthetic input.** Documented above; `ydotool` is the workaround.
- **Speech is the slow channel.** Budget ~800ms for a spoken command against ~150ms for a gesture. Assign latency-tolerant actions to voice.
- **Backlighting breaks hand detection.** A window behind you is the single most common reason detection fails; the overlay surfaces MediaPipe's confidence so you can see it happening.

## License

MIT.
