"""CLI surface: the commands a stuck user is told to run must actually work."""

from __future__ import annotations

import pytest

from airwave.cli import main


def run(argv, capsys) -> tuple[int, str]:
    code = main(argv)
    return code, capsys.readouterr().out


def test_init_writes_a_config_that_validates(tmp_path, capsys):
    path = tmp_path / "airwave.yaml"
    code, out = run(["init", str(path)], capsys)
    assert code == 0 and path.exists()

    code, out = run(["--config", str(path), "check"], capsys)
    assert code == 0
    assert "OK" in out


def test_init_refuses_to_clobber_without_force(tmp_path, capsys):
    path = tmp_path / "airwave.yaml"
    run(["init", str(path)], capsys)
    code, out = run(["init", str(path)], capsys)
    assert code == 1 and "already exists" in out

    assert run(["init", str(path), "--force"], capsys)[0] == 0


def test_check_lists_every_binding(tmp_path, capsys):
    path = tmp_path / "airwave.yaml"
    run(["init", str(path)], capsys)
    _, out = run(["--config", str(path), "check"], capsys)
    for expected in ("gesture:open_palm", "sound:clap", "Play/pause", "Volume up"):
        assert expected in out


def test_check_warns_when_arming_is_off(tmp_path, capsys):
    path = tmp_path / "airwave.yaml"
    run(["init", str(path)], capsys)
    _, out = run(["--config", str(path), "check"], capsys)
    assert "arming is off" in out


def test_check_on_a_broken_config_exits_nonzero_with_line_numbers(tmp_path, capsys):
    path = tmp_path / "airwave.yaml"
    path.write_text(
        "settings:\n  stable_frames: 5\nbindings:\n"
        "  - name: bad\n    trigger: { type: gesture, value: nope }\n"
        "    action: { type: key, keys: [a] }\n",
        encoding="utf-8",
    )
    assert main(["--config", str(path), "check"]) == 2
    err = capsys.readouterr().err
    assert "unknown gesture" in err
    assert f"{path}:5" in err


def test_check_on_a_missing_config_points_at_init(tmp_path, capsys):
    assert main(["--config", str(tmp_path / "nope.yaml"), "check"]) == 2
    assert "airwave init" in capsys.readouterr().err


def test_gestures_lists_the_vocabulary(capsys):
    code, out = run(["gestures"], capsys)
    assert code == 0
    for name in ("fist", "open_palm", "thumbs_up", "spock"):
        assert name in out
    assert "neutral pose" in out


def test_dataset_reports_recorded_samples(tmp_path, capsys):
    from conftest import hand_for

    from airwave.training.recorder import GestureRecorder

    recorder = GestureRecorder("wave", tmp_path)
    recorder.add(hand_for("open_palm"), "Right")
    recorder.save()

    code, out = run(["dataset", "--data-dir", str(tmp_path)], capsys)
    assert code == 0 and "wave" in out


def test_dataset_on_an_empty_dir_is_a_clean_failure(tmp_path, capsys):
    code, out = run(["dataset", "--data-dir", str(tmp_path)], capsys)
    assert code == 1 and "no recordings" in out


def test_train_without_recordings_explains_what_to_do(tmp_path, capsys):
    code = main(["train", "--data-dir", str(tmp_path), "--out", str(tmp_path / "m.npz")])
    assert code == 1
    assert "airwave record" in capsys.readouterr().err


def test_train_produces_a_model_from_recordings(tmp_path, capsys):
    from conftest import hand_for

    from airwave.training.recorder import GestureRecorder

    for label in ("fist", "open_palm"):
        recorder = GestureRecorder(label, tmp_path / "gestures")
        for i in range(20):
            recorder.add(hand_for(label, position=(0.4 + i * 0.01, 0.5), scale=1.0 + i * 0.01), "Right")
        recorder.save()

    out_path = tmp_path / "model.npz"
    code, out = run(["train", "--data-dir", str(tmp_path / "gestures"),
                     "--out", str(out_path), "--epochs", "60"], capsys)
    assert code == 0 and out_path.exists()
    assert "val accuracy" in out
    assert "classifier: model" in out


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "airwave" in capsys.readouterr().out


def test_help_mentions_the_first_run_sequence(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    assert "airwave init" in out and "airwave doctor" in out


def test_unknown_command_is_rejected(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["nonsense"])
    assert exc.value.code != 0
