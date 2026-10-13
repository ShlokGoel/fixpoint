"""Minimal unified-diff parser: enough to know which files changed and which lines."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_HEADER = re.compile(r"^diff --git a/(\S+) b/(\S+)")


@dataclass
class FileDiff:
    path: str
    old_path: str
    is_new: bool = False
    is_deleted: bool = False
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)


def parse_diff(patch: str) -> list[FileDiff]:
    files: list[FileDiff] = []
    current: FileDiff | None = None
    for line in patch.splitlines():
        m = _HEADER.match(line)
        if m:
            current = FileDiff(path=m.group(2), old_path=m.group(1))
            files.append(current)
            continue
        if current is None:
            continue
        if line.startswith("new file mode"):
            current.is_new = True
        elif line.startswith("deleted file mode"):
            current.is_deleted = True
        elif line.startswith("+++") or line.startswith("---"):
            continue
        elif line.startswith("+"):
            current.added.append(line[1:])
        elif line.startswith("-"):
            current.removed.append(line[1:])
    return files
