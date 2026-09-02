# Setup

The goal is under five minutes from clone to a working gesture. Most of what
follows exists because of the three or four things that reliably go wrong.

```bash
python -m pip install -r requirements.txt
python -m airwave doctor
```

`doctor` checks imports, camera, microphone, OS permissions, the hand model and
your config, and prints a `[FAIL]` line with the fix for anything broken. Run it
first, and run it again any time behaviour surprises you.

---

## The hand model

MediaPipe ships two generations of Python API and which one you get depends
entirely on the version pip resolved:

- **Older builds** (`mediapipe.solutions.hands`) bundle the model in the wheel.
  Nothing to do.
- **Newer builds** (`mediapipe.tasks.python.vision.HandLandmarker`) load an
  external `hand_landmarker.task` bundle.

Airwave detects and adapts. If your build needs the external model:

```bash
python -m airwave fetch-model
```

That downloads ~7.5 MB to `models/hand_landmarker.task`. It is the only network
call Airwave ever makes, it is explicit, and it happens once. You can also point
at a copy you already have:

```bash
export AIRWAVE_HAND_MODEL=/path/to/hand_landmarker.task
```

---

## macOS

**Accessibility permission is mandatory and fails silently without it.** Every
`pyautogui` keystroke succeeds and does nothing, so the app looks broken rather
than unpermitted. This is the most common first-run failure by a wide margin.

System Settings → Privacy & Security → Accessibility → enable your terminal
(or the Python binary, if you launched it directly). Then restart the terminal —
the permission is read at process start.

```bash
open "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
```

macOS also prompts for **Camera** and **Microphone** on first run. Accept both.

For media keys (`playpause`, `nexttrack`) and for `doctor` to verify the
Accessibility permission rather than guessing:

```bash
pip install pyobjc-framework-Quartz pyobjc-framework-Cocoa
```

Without pyobjc, volume keys still work through AppleScript; play/pause and track
skipping do not, because macOS routes them through a private NSEvent type.

`cv2.imshow` must run on the main thread on macOS. Airwave already does this —
the vision loop *is* the main thread — but it is why the threading model is
shaped the way it is.

---

## Windows

Works with no extra setup. Media and volume keys are native.

If the camera takes several seconds to open, that is the MSMF backend; Airwave
already requests DirectShow, which opens in a few hundred milliseconds.

Windows Defender or a corporate endpoint agent may flag synthetic input from a
Python process. If keystrokes stop arriving with no error, that is where to look.

---

## Linux

### X11

Works. Make sure `DISPLAY` is set — over SSH without X forwarding, input
synthesis has nowhere to go, and `doctor` will say so.

You may need one of:

```bash
sudo apt install python3-tk python3-dev scrot   # pyautogui's X11 dependencies
sudo apt install libportaudio2                  # sounddevice
```

### Wayland

**`pyautogui` cannot synthesize input under Wayland.** This is a deliberate
security property of the display protocol, not a bug in Airwave, and no amount
of configuration works around it from inside the process.

`doctor` and `run` both detect Wayland and print this at startup rather than
letting you conclude the app is broken.

Two options:

1. **Log into an X11/Xorg session.** Most distributions still offer this at the
   login screen. Everything works.
2. **Use `ydotool` through `shell` actions.** `ydotool` talks to the kernel's
   uinput device and bypasses the display server:

   ```yaml
   settings:
     allow_shell: true
   bindings:
     - name: "Play/pause"
       trigger: { type: gesture, value: open_palm }
       action:  { type: shell, command: "ydotool key 164:1 164:0" }
   ```

   This needs `ydotoold` running and your user in the `input` group.

Camera and microphone capture work fine under Wayland — only *output* is blocked.

---

## Choosing devices

```bash
python -m airwave devices
```

```yaml
settings:
  camera_index: 1
  audio:
    device: 3        # index, or a substring of the name
```

If a camera index opens but delivers no frames, another application is holding
it — Zoom, Teams and browser tabs are the usual culprits.

---

## Tuning the clap detector

Room noise varies enormously, so the threshold is per-room:

```bash
python -m airwave calibrate-audio
```

A live meter shows the current level and where the firing threshold sits. Stay
quiet for a few seconds to let the baseline settle, then clap. You want silence
well left of the marker and claps well past it. The command suggests a `floor`
value when you exit.

- Claps missed → lower `multiplier`, or lower `floor`.
- Typing or a door fires it → raise `multiplier`.
- Double-clap not recognized → widen `double_clap_max_ms`.

Note the trade-off: recognizing `double_clap` requires waiting out the
inter-onset window before a single clap can be declared, which costs up to
600ms. Airwave only pays that when a `double_clap` binding actually exists.

---

## Speech (optional)

Off by default, and it needs an engine:

```bash
pip install vosk              # fast, ~50MB model, lower accuracy
# or
pip install faster-whisper    # accurate, ~150MB model, slower on CPU
```

```yaml
settings:
  speech:
    enabled: true
    engine: vosk
    wake_word: "hey airwave"
```

The ASR does not run continuously. An energy VAD detects speech, buffers the
utterance, and only then transcribes — on a worker thread, never in the audio
callback. A wake word gates every command.

Both "hey airwave switch window" and "hey airwave" followed by "switch window"
work. Phrase matching is fuzzy, because ASR output for "volume up" is frequently
"volume app"; tune `similarity` if you get misfires or misses.

---

## Performance

```bash
python -m airwave bench
```

Targets: **≥20fps** and **p95 frame time ≤50ms**. If you are below them:

- Lower `camera_width` / `camera_height` to 480×360.
- Close other applications using the camera.
- Check the fps readout in the overlay — green is ≥20, red is below 12.

MediaPipe inference dominates the frame budget. Everything Airwave adds on top
(normalize, classify, stabilize) runs in well under a millisecond, which the
test suite asserts.

---

## Lighting

The most common reason a gesture "stops working" is not the classifier:

- **Backlighting kills detection.** A bright window behind you turns your hand
  into a silhouette. Face the light instead.
- **Dim rooms** raise camera exposure time, which motion-blurs a moving hand.
- **Keep your whole hand in frame.** A wrist cut off at the edge makes the
  normalization scale wrong, and every ratio with it.

The overlay shows the detection confidence, so you can watch this happen rather
than guess.
