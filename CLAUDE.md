# Airwave — notes for working in this repo

Local gesture + sound control. Python, no build step. See
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the design and the reasoning
behind it; this file is the short version of how to work here.

## Commands

```bash
python -m pytest                      # full suite, <1s, no hardware needed
python -m airwave check               # validate airwave.yaml, list bindings
python -m airwave doctor              # diagnose the environment
python -m airwave run --dry-run       # run without actually pressing keys
python -m airwave bench --frames 200  # fps / p95 against the targets
```

## The invariant everything protects

A per-frame classifier that is 95% accurate is wrong ~90 times a minute at
30fps. Commands must fire **exactly once**, when intended, and never otherwise.
Before changing anything in `vision/stabilizer.py` or `dispatch/dispatcher.py`,
read the tests in `tests/test_stabilizer.py` — they are the specification.

## Rules that are not negotiable

- **Producers never act.** Nothing in `vision/` or `audio/` may press a key,
  move the cursor or run a command. They emit `Event`s onto the queue. All side
  effects live in `dispatch/`.
- **All OS-specific knowledge lives in `dispatch/platform.py`.** No
  `sys.platform` checks scattered elsewhere.
- **Heavy imports stay lazy.** `import airwave` must not pull in OpenCV,
  MediaPipe, pyautogui or sounddevice — import them inside functions. The test
  suite and config validation run on machines with no camera and no display.
- **The audio callback must never block.** RMS and buffering only; anything
  slower goes to a worker thread.

## Conventions

- Comments explain *why*, especially where the obvious implementation is wrong
  (see the beta-units comment in `vision/pointer.py`, the baseline-update
  comment in `audio/onset.py`, the content-hash comment in `config/loader.py`).
  Match that density — it is the house style here.
- Config validation reports **every** problem with a line number and a
  suggestion, never just the first.
- New settings go in `config/schema.py` **and** `config/template.py`, with a
  comment in the template — that file is most users' only documentation.
- Test names state the behaviour being protected
  (`test_fist_is_not_read_as_pinch`), not the function being called.

## Adding things

**A gesture:** add to `gestures.BUILTIN_GESTURES` + `GESTURE_HELP`, add a case
in `rules.RuleClassifier._label`, add a pose to `tests/conftest.py::POSES`. The
parametrized tests in `test_rules.py` will then cover it automatically.

**An action type:** add to `schema.ACTION_TYPES`, validate its fields in
`_parse_action`, add `_do_<type>` to `ActionExecutor`, test it against the
dry-run backend.

**A trigger kind:** add to `EventKind`, emit it from a producer, handle it in
`Dispatcher._resolve`. Nothing else should need to change — if it does, the
producer/dispatcher boundary has been crossed.

## Gotchas

- MediaPipe ships two incompatible Python APIs. `vision/landmarks.py` detects
  which one is installed; newer builds need `python -m airwave fetch-model`.
- `cv2.imshow` must run on the main thread on macOS. The vision loop *is* the
  main thread. Do not move it.
- Hot reload stages a config from the watcher thread; the main loop applies it
  between frames. Never apply one mid-frame.
- The synthetic hands in `tests/conftest.py` are geometry, not real MediaPipe
  output. If you change the extension thresholds in `rules.py`, verify the poses
  still classify: `python -m pytest tests/test_rules.py`.
