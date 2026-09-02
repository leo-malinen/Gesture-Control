"""The starter config written by ``airwave init``.

Kept as a module constant rather than a data file so that an installed copy
can always regenerate a working config, and so the comments - which are the
real documentation for most users - version alongside the schema.
"""

from __future__ import annotations

TEMPLATE = """\
# Airwave configuration
# Everything here hot-reloads: save the file and the running app picks it up.
# Validate without running the camera:  python -m airwave check

settings:
  # ---- camera ----------------------------------------------------------
  camera_index: 0          # `python -m airwave devices` lists what you have
  camera_width: 640
  camera_height: 480
  mirror: true             # selfie view; move right, the hand moves right

  # ---- the reliability dials -------------------------------------------
  # stable_frames: how many consecutive frames must agree before a gesture
  # counts. Raise it if you get false positives, lower it if gestures feel
  # sluggish. 5 frames at 30fps is about 165ms.
  stable_frames: 5

  # cooldown_ms: after a trigger fires, ignore the same trigger for this long.
  cooldown_ms: 800

  min_detection_confidence: 0.6
  min_tracking_confidence: 0.5
  pinch_threshold: 0.04    # thumb-to-index distance, as a fraction of frame width

  # ---- classifier ------------------------------------------------------
  # 'rules' is the zero-setup geometric classifier. Switch to 'model' after
  # recording your own gestures (airwave record / airwave train).
  classifier: rules
  model_path: data/model.npz

  # ---- arming (the strongest defense against false positives) ----------
  # When enabled, nothing fires unless you armed within the last timeout_s
  # seconds, by holding the arming gesture or saying the wake word.
  # Off by default so the first run feels instant; turn it on for daily use.
  arming:
    enabled: false
    gesture: thumbs_up
    hold_ms: 1000
    timeout_s: 5
    wake_word: true        # the speech wake word also arms
    disarm_on_fire: false  # true = one action per arm

  # ---- mouse control ---------------------------------------------------
  mouse:
    enabled: false
    smoothing: 0.6              # 0 = raw and jittery, 0.95 = very steady, laggy
    active_region: [0.25, 0.25, 0.75, 0.75]   # this part of the frame maps to the whole screen
    pinch_click: true
    dwell_click: false          # click by holding the cursor still
    dwell_ms: 900
    click_cooldown_ms: 400

  # ---- audio -----------------------------------------------------------
  # Tune multiplier and floor with: python -m airwave calibrate-audio
  audio:
    enabled: true
    samplerate: 16000
    blocksize: 512
    multiplier: 8.0        # fire when rms > baseline * multiplier
    floor: 0.06            # ...but never below this, or a quiet room fires on typing
    refractory_ms: 180
    double_clap_min_ms: 150
    double_clap_max_ms: 600

  # ---- speech (optional; needs vosk or faster-whisper installed) --------
  speech:
    enabled: false
    engine: vosk           # vosk = fast, faster-whisper = accurate
    wake_word: "hey airwave"
    listen_window_s: 4.0
    similarity: 0.72       # how close a heard phrase must be to a configured one

  # ---- safety ----------------------------------------------------------
  # Shell actions are off by default: a config file that runs commands is a
  # real attack surface if you download someone else's.
  allow_shell: false

  overlay: true
  show_landmarks: true
  hot_reload: true
  log_level: info

# Custom gestures you have recorded (airwave record my_gesture).
# Listing them here makes them valid trigger values.
custom_gestures: []

bindings:
  - name: "Play/pause"
    trigger: { type: gesture, value: open_palm }
    action:  { type: key, keys: [playpause] }

  - name: "Volume up"
    trigger: { type: gesture, value: one }
    action:  { type: key, keys: [volumeup], repeat: 3 }

  - name: "Volume down"
    trigger: { type: gesture, value: two }
    action:  { type: key, keys: [volumedown], repeat: 3 }

  - name: "Next track"
    trigger: { type: sound, value: clap }
    action:  { type: key, keys: [nexttrack] }
    cooldown_ms: 1200

  # Gestures with no binding simply do nothing - `fist` is the neutral pose
  # you return to between commands, and binding it is not recommended.

  # Speech example. Enable settings.speech.enabled first.
  # - name: "Switch window"
  #   trigger: { type: speech, value: "switch window" }
  #   action:  { type: hotkey, keys: [alt, tab] }

  # Shell example. Requires settings.allow_shell: true.
  # - name: "Open notes"
  #   trigger: { type: gesture, value: l_shape }
  #   action:  { type: shell, command: "open -a Notes" }

  # Toggle mouse control with a gesture instead of the 'm' key.
  # - name: "Toggle mouse"
  #   trigger: { type: gesture, value: rock }
  #   action:  { type: mode, target: mouse, verb: toggle }
"""
