"""Config errors that tell the user where to look and what was legal instead."""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field


@dataclass
class ConfigIssue:
    """One problem with one place in the file.

    ``path`` is a dotted/indexed breadcrumb ("bindings[2].action.keys") so a
    deeply nested mistake is locatable even when the line number is missing.
    """

    path: str
    message: str
    line: int | None = None
    hint: str | None = None

    def render(self, source: str | None = None) -> str:
        where = f"{source}:{self.line}" if source and self.line else (f"line {self.line}" if self.line else "")
        head = f"  {where + ': ' if where else ''}{self.path}"
        out = [f"{head}\n      {self.message}"]
        if self.hint:
            out.append(f"      hint: {self.hint}")
        return "\n".join(out)


class ConfigError(Exception):
    """Raised with every issue found, not just the first one.

    Reporting one error per run turns a five-mistake config into five
    edit-run cycles, which is exactly the friction the PRD's "time to first
    success" metric is trying to kill.
    """

    def __init__(self, issues: list[ConfigIssue], source: str | None = None) -> None:
        self.issues = issues
        self.source = source
        super().__init__(self.render())

    def render(self) -> str:
        n = len(self.issues)
        header = f"{n} problem{'s' if n != 1 else ''} in {self.source or 'config'}:"
        body = "\n".join(issue.render(self.source) for issue in self.issues)
        return f"{header}\n{body}"


def suggest(value: str, candidates, label: str = "valid values") -> str:
    """Turn a typo into a pointer rather than a wall of options."""
    options = sorted(str(c) for c in candidates)
    close = difflib.get_close_matches(str(value), options, n=3, cutoff=0.55)
    if close:
        return "did you mean " + " or ".join(f"'{c}'" for c in close) + "?"
    shown = ", ".join(options[:12]) + (", ..." if len(options) > 12 else "")
    return f"{label}: {shown}"
