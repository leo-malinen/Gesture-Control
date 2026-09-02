"""Config loading, validation quality, and hot reload (FR-5.x).

Validation is a user-experience feature here, not a formality: the PRD's
"under five minutes to first success" is mostly a story about a config file
that explains its own mistakes. So these tests assert on the *content* of
error messages - line numbers, suggestions - not just that something raised.
"""

from __future__ import annotations

import textwrap

import pytest

from airwave.config import ConfigError, load_config, load_yaml, parse_config
from airwave.config.loader import ConfigWatcher, find_config
from airwave.config.template import TEMPLATE


def parse(text: str):
    return parse_config(load_yaml(textwrap.dedent(text)), source="test.yaml")


def issues_for(text: str):
    return parse(text)[1]


def test_shipped_template_is_valid():
    """The file `airwave init` writes must pass its own validator."""
    cfg, issues = parse(TEMPLATE)
    assert issues == []
    assert len(cfg.bindings) >= 4


def test_defaults_apply_to_an_empty_config():
    cfg, issues = parse("{}")
    assert issues == []
    assert cfg.settings.stable_frames == 5
    assert cfg.settings.cooldown_ms == 800
    assert cfg.settings.arming.enabled is False
    assert cfg.settings.allow_shell is False


def test_unknown_gesture_names_the_line_and_suggests_a_fix():
    issues = issues_for("""
        bindings:
          - name: oops
            trigger: { type: gesture, value: open_palmm }
            action: { type: key, keys: [space] }
    """)
    assert len(issues) == 1
    assert issues[0].line == 4
    assert "open_palmm" in issues[0].message
    assert "open_palm" in issues[0].hint


def test_every_problem_is_reported_not_just_the_first():
    """Five mistakes should be one edit-run cycle, not five."""
    issues = issues_for("""
        settings:
          stable_frames: "five"
          cooldown_ms: -3
        bindings:
          - name: a
            trigger: { type: gesture, value: nope }
            action: { type: key }
    """)
    paths = {issue.path for issue in issues}
    assert "settings.stable_frames" in paths
    assert "settings.cooldown_ms" in paths
    assert "bindings[0].trigger.value" in paths
    assert "bindings[0].action.keys" in paths


def test_typo_in_a_setting_name_is_an_error_not_a_silent_no_op():
    issues = issues_for("settings: { stable_frame: 5 }")
    assert issues[0].path == "settings.stable_frame"
    assert "stable_frames" in issues[0].hint


def test_unknown_action_type_lists_the_valid_ones():
    issues = issues_for("""
        bindings:
          - name: a
            trigger: { type: gesture, value: one }
            action: { type: keypress, keys: [a] }
    """)
    assert "keypress" in issues[0].message
    assert "key" in issues[0].hint


def test_shell_actions_are_refused_unless_explicitly_allowed():
    """FR-5.5 - a shared config that runs commands is an attack surface."""
    issues = issues_for("""
        bindings:
          - name: notes
            trigger: { type: gesture, value: one }
            action: { type: shell, command: "rm -rf /" }
    """)
    assert any("allow_shell" in (issue.hint or "") for issue in issues)


def test_shell_actions_pass_when_allowed():
    cfg, issues = parse("""
        settings: { allow_shell: true }
        bindings:
          - name: notes
            trigger: { type: gesture, value: one }
            action: { type: shell, command: "open -a Notes" }
    """)
    assert issues == []
    assert cfg.bindings[0].action.command == "open -a Notes"


def test_binding_the_same_trigger_twice_is_an_error():
    """Silently picking one of two bindings is worse than refusing to start."""
    issues = issues_for("""
        bindings:
          - name: first
            trigger: { type: gesture, value: one }
            action: { type: key, keys: [a] }
          - name: second
            trigger: { type: gesture, value: one }
            action: { type: key, keys: [b] }
    """)
    assert any("duplicate trigger" in issue.message for issue in issues)


def test_arming_gesture_cannot_also_be_bound():
    """Arming consumes the gesture, so such a binding could never fire."""
    issues = issues_for("""
        settings:
          arming: { enabled: true, gesture: thumbs_up }
        bindings:
          - name: confirm
            trigger: { type: gesture, value: thumbs_up }
            action: { type: key, keys: [enter] }
    """)
    assert any(issue.path == "settings.arming.gesture" for issue in issues)


def test_speech_bindings_without_speech_enabled_are_flagged():
    issues = issues_for("""
        bindings:
          - name: switch
            trigger: { type: speech, value: "switch window" }
            action: { type: hotkey, keys: [alt, tab] }
    """)
    assert any("speech is disabled" in issue.message for issue in issues)


def test_reserved_none_label_cannot_be_bound():
    issues = issues_for("""
        bindings:
          - name: nope
            trigger: { type: gesture, value: none }
            action: { type: key, keys: [a] }
    """)
    assert any("reserved" in issue.message for issue in issues)


def test_custom_gestures_become_valid_trigger_values():
    cfg, issues = parse("""
        custom_gestures: [my_sign]
        bindings:
          - name: custom
            trigger: { type: gesture, value: my_sign }
            action: { type: key, keys: [a] }
    """)
    assert issues == []
    assert cfg.custom_gestures == ("my_sign",)


def test_active_region_must_be_a_real_box():
    issues = issues_for("settings: { mouse: { active_region: [0.8, 0.2, 0.3, 0.9] } }")
    assert any("active_region" in issue.path for issue in issues)


def test_hotkey_with_one_key_is_rejected_with_a_pointer_to_key():
    issues = issues_for("""
        bindings:
          - name: single
            trigger: { type: gesture, value: one }
            action: { type: hotkey, keys: [enter] }
    """)
    assert any("type: key" in (issue.hint or "") for issue in issues)


def test_double_clap_window_must_be_ordered():
    issues = issues_for("settings: { audio: { double_clap_min_ms: 700, double_clap_max_ms: 300 } }")
    assert any("double_clap_max_ms" in issue.path for issue in issues)


def test_yaml_syntax_error_reports_a_line():
    with pytest.raises(ConfigError) as exc:
        load_yaml("settings:\n  a: 1\n   b: 2\n", "test.yaml")
    assert exc.value.issues[0].line is not None


def test_tab_indentation_gets_a_specific_hint():
    with pytest.raises(ConfigError) as exc:
        load_yaml("settings:\n\tcamera_index: 0\n", "test.yaml")
    assert "tabs" in (exc.value.issues[0].hint or "")


def test_rendered_error_contains_file_line_and_hint():
    _, issues = parse("""
        bindings:
          - name: a
            trigger: { type: gesture, value: opne_palm }
            action: { type: key, keys: [space] }
    """)
    rendered = ConfigError(issues, "airwave.yaml").render()
    assert "airwave.yaml:4" in rendered
    assert "did you mean" in rendered


# ------------------------------------------------------------------ file layer


def test_load_config_roundtrip(tmp_path):
    path = tmp_path / "airwave.yaml"
    path.write_text(TEMPLATE, encoding="utf-8")
    cfg = load_config(path)
    assert cfg.source == str(path)
    assert cfg.bindings_for("gesture", "open_palm")


def test_missing_file_says_how_to_create_one(tmp_path):
    with pytest.raises(ConfigError) as exc:
        load_config(tmp_path / "nope.yaml")
    assert "airwave init" in (exc.value.issues[0].hint or "")


def test_find_config_prefers_explicit_then_local(tmp_path, monkeypatch):
    monkeypatch.delenv("AIRWAVE_CONFIG", raising=False)
    local = tmp_path / "airwave.yaml"
    local.write_text(TEMPLATE, encoding="utf-8")
    assert find_config(None, start=tmp_path) == local
    assert find_config(str(tmp_path / "other.yaml")) == tmp_path / "other.yaml"


def test_env_var_overrides_the_local_file(tmp_path, monkeypatch):
    monkeypatch.setenv("AIRWAVE_CONFIG", str(tmp_path / "from_env.yaml"))
    assert find_config(None, start=tmp_path).name == "from_env.yaml"


# -------------------------------------------------------------------- reload


def test_watcher_reloads_on_change(tmp_path):
    path = tmp_path / "airwave.yaml"
    path.write_text(TEMPLATE, encoding="utf-8")
    seen = []
    watcher = ConfigWatcher(path, seen.append, interval_s=0.01)

    assert watcher.poll() is False, "no change yet"
    path.write_text(TEMPLATE.replace("stable_frames: 5", "stable_frames: 9"), encoding="utf-8")
    assert watcher.poll() is True
    assert seen[-1].settings.stable_frames == 9


def test_watcher_keeps_the_last_good_config_on_a_bad_save(tmp_path):
    """An editor writing a half-finished file must not take down a session."""
    path = tmp_path / "airwave.yaml"
    path.write_text(TEMPLATE, encoding="utf-8")
    applied, errors = [], []
    watcher = ConfigWatcher(path, applied.append, on_error=errors.append, interval_s=0.01)

    path.write_text("settings:\n  stable_frames: [broken\n", encoding="utf-8")
    assert watcher.poll() is False
    assert applied == []
    assert errors and errors[0]


def test_watcher_recovers_after_the_file_is_fixed(tmp_path):
    path = tmp_path / "airwave.yaml"
    path.write_text(TEMPLATE, encoding="utf-8")
    applied = []
    watcher = ConfigWatcher(path, applied.append, on_error=lambda _: None, interval_s=0.01)

    path.write_text("settings: {{{", encoding="utf-8")
    watcher.poll()
    path.write_text(TEMPLATE.replace("cooldown_ms: 800", "cooldown_ms: 400"), encoding="utf-8")
    assert watcher.poll() is True
    assert applied[-1].settings.cooldown_ms == 400


def test_every_shipped_example_config_is_valid():
    """The examples are documentation people copy; broken ones are worse than
    none. This catches them rotting when the schema changes."""
    from pathlib import Path

    examples = sorted((Path(__file__).parent.parent / "examples").glob("*.yaml"))
    assert examples, "expected example configs to ship"
    for path in examples:
        load_config(path)  # raises ConfigError with details if invalid


def test_the_repo_root_config_is_valid():
    from pathlib import Path

    root = Path(__file__).parent.parent / "airwave.yaml"
    if root.exists():
        load_config(root)
