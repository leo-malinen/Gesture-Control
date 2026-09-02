"""Command line interface.

Subcommands exist in proportion to how often a first run fails: ``doctor``,
``devices`` and ``check`` are diagnostics, not features. The PRD's "under five
minutes from clone to a working gesture" is mostly a story about making the
three or four things that go wrong say so out loud.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from . import __version__
from .config import ConfigError, DEFAULT_CONFIG_NAME, find_config, load_config
from .config.template import TEMPLATE
from .gestures import BUILTIN_GESTURES, GESTURE_HELP


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)-22s %(message)s",
        datefmt="%H:%M:%S",
    )


def _load(args) -> tuple:
    """Resolve and load the config, exiting with a readable error if invalid."""
    path = find_config(getattr(args, "config", None))
    try:
        return load_config(path), path
    except ConfigError as exc:
        print(exc.render(), file=sys.stderr)
        if "config file not found" in str(exc):
            print("\nWrite a starter config with:  python -m airwave init", file=sys.stderr)
        raise SystemExit(2) from None


# --------------------------------------------------------------------- commands


def cmd_init(args) -> int:
    path = Path(args.path or DEFAULT_CONFIG_NAME)
    if path.exists() and not args.force:
        print(f"{path} already exists (use --force to overwrite)")
        return 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(TEMPLATE, encoding="utf-8")
    print(f"wrote {path}\n\nNext:\n  python -m airwave check\n  python -m airwave run")
    return 0


def cmd_check(args) -> int:
    cfg, path = _load(args)
    print(f"{path}: OK\n")
    print(f"{'trigger':<26} {'action':<34} binding")
    print("-" * 90)
    for binding in cfg.bindings:
        trigger = f"{binding.trigger.type}:{binding.trigger.value}"
        flags = []
        if not binding.enabled:
            flags.append("disabled")
        if binding.cooldown_ms is not None:
            flags.append(f"cooldown {binding.cooldown_ms}ms")
        suffix = f"  ({', '.join(flags)})" if flags else ""
        print(f"{trigger:<26} {binding.action.describe():<34} {binding.name}{suffix}")

    settings = cfg.settings
    print(
        f"\nstable_frames {settings.stable_frames}  cooldown {settings.cooldown_ms}ms  "
        f"classifier {settings.classifier}  arming {'on' if settings.arming.enabled else 'off'}  "
        f"mouse {'on' if settings.mouse.enabled else 'off'}  "
        f"audio {'on' if settings.audio.enabled else 'off'}  "
        f"speech {'on' if settings.speech.enabled else 'off'}"
    )
    if not settings.arming.enabled:
        print(
            "\nnote: arming is off, so any recognized gesture fires immediately.\n"
            "      Turn it on (settings.arming.enabled: true) for everyday use."
        )
    return 0


def cmd_gestures(args) -> int:
    cfg = None
    try:
        cfg, _ = _load(args)
    except SystemExit:
        pass
    print("built-in gestures:\n")
    for name in BUILTIN_GESTURES:
        print(f"  {name:<12} {GESTURE_HELP.get(name, '')}")
    if cfg and cfg.custom_gestures:
        print("\ncustom gestures (from your config):\n")
        for name in cfg.custom_gestures:
            print(f"  {name}")
    print("\n'fist' is the neutral pose - return to it between commands.")
    print("'none' means no hand in frame and cannot be bound.")
    return 0


def cmd_devices(args) -> int:
    from .capture.camera import list_cameras  # noqa: PLC0415
    from .capture.microphone import list_microphones  # noqa: PLC0415

    print("cameras:")
    cameras = list_cameras()
    for camera in cameras:
        print(f"  index {camera['index']}  {camera['width']}x{camera['height']}")
    if not cameras:
        print("  none found")

    print("\nmicrophones:")
    for mic in list_microphones():
        mark = " (default)" if mic["default"] else ""
        print(f"  index {mic['index']}  {mic['name']}  {mic['channels']}ch @ {mic['samplerate']}Hz{mark}")
    return 0


def cmd_doctor(args) -> int:
    """Everything that commonly breaks a first run, checked in one pass."""
    from .dispatch.platform import describe, detect  # noqa: PLC0415
    from .vision.landmarks import resolve_model_path  # noqa: PLC0415

    ok = True
    print(f"airwave {__version__}")
    print(f"platform: {describe()}\n")

    info = detect()
    for warning in info.warnings:
        print(f"[warn] {warning}\n")
    for blocker in info.blocking:
        print(f"[FAIL] {blocker}\n")
        ok = False

    for module in ("cv2", "mediapipe", "numpy", "yaml", "pyautogui", "sounddevice"):
        try:
            __import__(module)
            print(f"[ ok ] import {module}")
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] import {module}: {exc}")
            ok = False

    try:
        import mediapipe as mp  # noqa: PLC0415

        legacy = hasattr(mp, "solutions") and hasattr(mp.solutions, "hands")
        if legacy:
            print("[ ok ] mediapipe legacy solutions API (model bundled)")
        else:
            path = resolve_model_path()
            if path.exists():
                print(f"[ ok ] hand landmarker model at {path}")
            else:
                print(f"[FAIL] mediapipe needs a hand model and none is at {path}\n"
                      f"       fix: python -m airwave fetch-model")
                ok = False
    except Exception:  # noqa: BLE001
        pass

    try:
        from .capture.camera import list_cameras  # noqa: PLC0415

        cameras = list_cameras(limit=3)
        print(f"[ {'ok' if cameras else 'FAIL'} ] cameras found: {[c['index'] for c in cameras] or 'none'}")
        ok = ok and bool(cameras)
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] camera probe: {exc}")
        ok = False

    try:
        from .capture.microphone import list_microphones  # noqa: PLC0415

        mics = list_microphones()
        print(f"[ {'ok' if mics else 'warn'} ] input devices: {len(mics)}")
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] microphone probe: {exc}")

    path = find_config(getattr(args, "config", None))
    if path.exists():
        try:
            cfg = load_config(path)
            print(f"[ ok ] config {path} ({len(cfg.bindings)} bindings)")
            unknown = _unknown_key_names(cfg)
            if unknown:
                print(f"[warn] key names your OS may not recognize: {', '.join(unknown)}")
        except ConfigError as exc:
            print(f"[FAIL] config {path}\n{exc.render()}")
            ok = False
    else:
        print(f"[warn] no config at {path} - run: python -m airwave init")

    print("\n" + ("all good - run: python -m airwave run" if ok else "fix the [FAIL] lines above, then re-run doctor"))
    return 0 if ok else 1


def _unknown_key_names(cfg) -> list[str]:
    from .dispatch.platform import InputBackend  # noqa: PLC0415

    backend = InputBackend()
    keys = [k for b in cfg.bindings if b.action.type in ("key", "hotkey") for k in b.action.keys]
    try:
        return sorted(set(backend.unknown_keys(keys)))
    except Exception:  # noqa: BLE001
        return []


def cmd_fetch_model(args) -> int:
    from .vision.landmarks import MODEL_URL, download_model, resolve_model_path  # noqa: PLC0415

    path = resolve_model_path(args.out)
    if path.exists() and not args.force:
        print(f"already present: {path}")
        return 0
    print(f"downloading {MODEL_URL}\n         -> {path}")
    try:
        download_model(path)
    except Exception as exc:  # noqa: BLE001
        print(f"download failed: {exc}", file=sys.stderr)
        return 1
    print(f"done ({path.stat().st_size / 1e6:.1f} MB)")
    return 0


def cmd_run(args) -> int:
    from .app import AirwaveApp  # noqa: PLC0415

    cfg, path = _load(args)
    if args.arm:
        cfg.settings.arming.enabled = True
    if args.no_arm:
        cfg.settings.arming.enabled = False
    if args.mouse:
        cfg.settings.mouse.enabled = True
    _setup_logging(args.log_level or cfg.settings.log_level)

    app = AirwaveApp(
        cfg,
        config_path=path,
        dry_run=args.dry_run,
        headless=args.headless,
        audio=False if args.no_audio else None,
    )
    print(f"airwave {__version__} - config {path}")
    if args.dry_run:
        print("dry run: actions are logged, not performed")
    print("press q in the preview window (or ctrl-c here) to quit\n")

    stats = app.run(max_frames=args.frames)
    dispatch_stats = app.dispatcher.stats
    print(
        f"\n{stats.frames} frames in {stats.elapsed_s:.1f}s  "
        f"({stats.fps:.1f} fps, p95 frame {stats.p95_frame_ms:.0f}ms)\n"
        f"fired {dispatch_stats.fired}  cooldown-suppressed {dispatch_stats.cooldown}  "
        f"unarmed {dispatch_stats.unarmed}  unbound {dispatch_stats.unbound}  errors {dispatch_stats.errors}"
    )
    if dispatch_stats.fired:
        print(f"mean action latency {dispatch_stats.mean_latency_ms:.0f}ms")
    return 0


def cmd_calibrate_audio(args) -> int:
    """Live RMS readout so thresholds can be picked for this room (FR-3.2)."""
    from .audio.onset import ClapGrouper, OnsetDetector  # noqa: PLC0415
    from .capture.microphone import Microphone  # noqa: PLC0415

    cfg, _ = _load(args)
    settings = cfg.settings.audio
    detector = OnsetDetector(
        baseline_alpha=settings.baseline_alpha,
        multiplier=settings.multiplier,
        floor=settings.floor,
        refractory_ms=settings.refractory_ms,
    )
    grouper = ClapGrouper(min_gap_ms=settings.double_clap_min_ms, max_gap_ms=settings.double_clap_max_ms,
                          detect_double=True)
    hits: list[str] = []

    def handle(block, now):
        onset = detector.push(block, now=now)
        for label, _ in grouper.feed(onset, now=now):
            hits.append(label)

    print(
        "Audio calibration. Stay quiet for a few seconds, then clap.\n"
        "  bar = current level, | = the firing threshold\n"
        "  aim for: silence well left of the marker, claps well past it\n"
        "  ctrl-c to stop\n"
    )
    mic = Microphone(handle, device=settings.device, samplerate=settings.samplerate,
                     blocksize=settings.blocksize, channels=settings.channels).start()
    try:
        while True:
            time.sleep(0.05)
            snap = detector.snapshot()
            width = 46
            scale = max(snap["threshold"] * 2.0, 0.2)
            level = int(min(1.0, snap["rms"] / scale) * width)
            marker = int(min(1.0, snap["threshold"] / scale) * width)
            bar = ["-"] * width
            for i in range(level):
                bar[i] = "#"
            if marker < width:
                bar[marker] = "|"
            flash = ("  <<< " + hits.pop()) if hits else ""
            print(f"\r[{''.join(bar)}] rms {snap['rms']:.4f}  base {snap['baseline']:.4f}  "
                  f"thr {snap['threshold']:.4f}{flash:<18}", end="", flush=True)
    except KeyboardInterrupt:
        print()
    finally:
        mic.stop()

    snap = detector.snapshot()
    suggested_floor = round(max(0.02, snap["baseline"] * 6), 3)
    print(
        f"\nroom baseline settled at {snap['baseline']:.4f}\n"
        f"suggested settings.audio.floor: {suggested_floor}\n"
        f"(raise the multiplier if claps still miss; lower it if typing fires them)"
    )
    return 0


def cmd_record(args) -> int:
    from .training.recorder import record_session  # noqa: PLC0415

    cfg, _ = _load(args)
    _setup_logging(args.log_level or "info")
    record_session(cfg, args.label, samples=args.samples, interval_ms=args.interval_ms)
    return 0


def cmd_train(args) -> int:
    from .training.train import format_report, train_from_dir  # noqa: PLC0415

    _setup_logging(args.log_level or "info")
    try:
        model, metrics = train_from_dir(args.data_dir, args.out, epochs=args.epochs, seed=args.seed)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"{exc}\nrecord at least two different gestures first.", file=sys.stderr)
        return 1
    print(format_report(model, metrics))
    return 0


def cmd_dataset(args) -> int:
    from .training.recorder import dataset_summary  # noqa: PLC0415

    summary = dataset_summary(args.data_dir)
    if not summary:
        print(f"no recordings in {args.data_dir}")
        return 1
    print(f"{'gesture':<16} samples")
    for label, count in sorted(summary.items()):
        print(f"{label:<16} {count}")
    print(f"\ntotal {sum(v for v in summary.values() if v > 0)}")
    return 0


def cmd_replay(args) -> int:
    """Run a recorded clip through the pipeline and print the event sequence."""
    from .app import replay_video  # noqa: PLC0415

    cfg, _ = _load(args)
    events = replay_video(args.video, cfg, max_frames=args.frames)
    for event in events:
        print(f"{event.captured_at:10.3f}  {event.describe():<24} {event.payload}")
    print(f"\n{len(events)} events from {args.video}")
    return 0


def cmd_bench(args) -> int:
    """Measure the frame budget on this machine (PRD success metric)."""
    from .app import AirwaveApp  # noqa: PLC0415

    cfg, path = _load(args)
    cfg.settings.overlay = False
    _setup_logging(args.log_level or "warning")
    app = AirwaveApp(cfg, config_path=path, dry_run=True, headless=True, audio=False)
    print(f"running {args.frames} frames...")
    stats = app.run(max_frames=args.frames)
    p95 = stats.p95_frame_ms
    print(
        f"\n{stats.frames} frames in {stats.elapsed_s:.1f}s\n"
        f"  mean fps        {stats.fps:.1f}   (target >= 20)\n"
        f"  p95 frame time  {p95:.1f}ms  (target <= 50)"
    )
    passed = stats.fps >= 20 and p95 <= 50
    print("\n" + ("PASS" if passed else "below target - try camera_width 480 or close other camera apps"))
    return 0 if passed else 1


# ----------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="airwave",
        description="Local gesture and sound control for your computer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "typical first run:\n"
            "  python -m airwave init\n"
            "  python -m airwave doctor\n"
            "  python -m airwave run\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"airwave {__version__}")
    parser.add_argument("-c", "--config", help=f"path to {DEFAULT_CONFIG_NAME}")
    parser.add_argument("--log-level", choices=("debug", "info", "warning", "error"))
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="run the app (default)")
    run.add_argument("--dry-run", action="store_true", help="log actions instead of performing them")
    run.add_argument("--headless", action="store_true", help="no preview window")
    run.add_argument("--no-audio", action="store_true", help="skip the microphone entirely")
    run.add_argument("--arm", action="store_true", help="force arming mode on")
    run.add_argument("--no-arm", action="store_true", help="force arming mode off")
    run.add_argument("--mouse", action="store_true", help="start with mouse control on")
    run.add_argument("--frames", type=int, help="stop after N frames (for testing)")
    run.set_defaults(func=cmd_run)

    init = sub.add_parser("init", help="write a starter airwave.yaml")
    init.add_argument("path", nargs="?", help=f"where to write it (default {DEFAULT_CONFIG_NAME})")
    init.add_argument("--force", action="store_true")
    init.set_defaults(func=cmd_init)

    check = sub.add_parser("check", help="validate the config and list bindings")
    check.set_defaults(func=cmd_check)

    doctor = sub.add_parser("doctor", help="diagnose a broken setup")
    doctor.set_defaults(func=cmd_doctor)

    devices = sub.add_parser("devices", help="list cameras and microphones")
    devices.set_defaults(func=cmd_devices)

    gestures = sub.add_parser("gestures", help="list the gesture vocabulary")
    gestures.set_defaults(func=cmd_gestures)

    fetch = sub.add_parser("fetch-model", help="download the MediaPipe hand model (one time)")
    fetch.add_argument("--out", help="destination path")
    fetch.add_argument("--force", action="store_true")
    fetch.set_defaults(func=cmd_fetch_model)

    calibrate = sub.add_parser("calibrate-audio", help="live RMS meter for picking clap thresholds")
    calibrate.set_defaults(func=cmd_calibrate_audio)

    record = sub.add_parser("record", help="record samples for a custom gesture")
    record.add_argument("label", help="gesture name, e.g. l_shape")
    record.add_argument("--samples", type=int, default=60)
    record.add_argument("--interval-ms", type=int, default=80)
    record.set_defaults(func=cmd_record)

    train = sub.add_parser("train", help="train the classifier on recorded samples")
    train.add_argument("--data-dir", default="data/gestures")
    train.add_argument("--out", default="data/model.npz")
    train.add_argument("--epochs", type=int, default=400)
    train.add_argument("--seed", type=int, default=0)
    train.set_defaults(func=cmd_train)

    dataset = sub.add_parser("dataset", help="show what has been recorded")
    dataset.add_argument("--data-dir", default="data/gestures")
    dataset.set_defaults(func=cmd_dataset)

    replay = sub.add_parser("replay", help="run a video file through the pipeline")
    replay.add_argument("video")
    replay.add_argument("--frames", type=int)
    replay.set_defaults(func=cmd_replay)

    bench = sub.add_parser("bench", help="measure fps and p95 frame time")
    bench.add_argument("--frames", type=int, default=200)
    bench.set_defaults(func=cmd_bench)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "func", None) is None:
        # Bare `python -m airwave` runs the app, which is what people expect.
        args = parser.parse_args([*(argv or []), "run"])
    _setup_logging(args.log_level or "info")
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        return 130
    except SystemExit as exc:
        return int(exc.code or 0)
    except Exception as exc:  # noqa: BLE001
        logging.getLogger("airwave").debug("unhandled", exc_info=True)
        print(f"\nerror: {exc}\n\nrun 'python -m airwave doctor' to check your setup.", file=sys.stderr)
        return 1
