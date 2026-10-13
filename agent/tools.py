"""Tool schemas and executors.

The executor is deliberately strict and chatty on failure: when an edit does
not apply, the error message says exactly why and how to fix it, because vague
tool errors are the most common reason agents thrash.
"""

from __future__ import annotations

import posixpath
import shlex
import time
from dataclasses import dataclass, field
from typing import Any

from .config import AgentConfig, ScaffoldConfig
from .sandbox import Sandbox

MAX_READ_LINES = 400
SEARCH_MAX_LINES = 100
FIND_MAX_LINES = 200


@dataclass
class ToolResult:
    content: str
    is_error: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


def truncate_middle(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.6)
    tail = max_chars - head
    cut = len(text) - head - tail
    return f"{text[:head]}\n\n[... {cut} characters truncated ...]\n\n{text[-tail:]}"


def number_lines(text: str, start: int = 1) -> str:
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    width = len(str(start + len(lines)))
    return "\n".join(f"{i:>{width}}\t{line}" for i, line in enumerate(lines, start))


def tool_specs(scaffold: ScaffoldConfig) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {
            "name": "bash",
            "description": (
                "Run a non-interactive shell command from the repository root. Commands time out, "
                "there is no network access, and output is truncated if very long. Use for git, "
                "python scripts, ls, etc. Prefer read_file/search/edit_file for files."
            ),
            "input_schema": {
                "type": "object",
                "properties": {"command": {"type": "string", "description": "The bash command to run."}},
                "required": ["command"],
            },
        },
        {
            "name": "read_file",
            "description": "Read a file with line numbers. Optionally pass start_line/end_line (1-indexed, inclusive).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the repository root."},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
        {
            "name": "edit_file",
            "description": (
                "Replace one exact occurrence of old_str with new_str in a file. old_str must match the "
                "file exactly (including whitespace and indentation) and must be unique in the file; "
                "include surrounding lines if needed to make it unique."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_str": {"type": "string"},
                    "new_str": {"type": "string"},
                },
                "required": ["path", "old_str", "new_str"],
            },
        },
        {
            "name": "create_file",
            "description": "Create a new file with the given content. Fails if the file already exists.",
            "input_schema": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
        },
        {
            "name": "search",
            "description": "Search file contents with an extended regex (grep -E). Returns matching lines with paths and line numbers.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {
                        "type": "string",
                        "description": "Directory or file to search. Defaults to the repo root.",
                    },
                    "include": {"type": "string", "description": "Optional filename glob, e.g. '*.py'."},
                },
                "required": ["pattern"],
            },
        },
        {
            "name": "find_files",
            "description": "Find files whose name matches a glob, e.g. '*.py' or 'test_*.py'.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "name_pattern": {"type": "string"},
                    "path": {"type": "string", "description": "Directory to search. Defaults to the repo root."},
                },
                "required": ["name_pattern"],
            },
        },
        {
            "name": "run_tests",
            "description": (
                "Run a test command, e.g. 'python -m pytest tests/test_x.py -x -q'. Same as bash, but "
                "recorded as a verification step."
            ),
            "input_schema": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    ]
    if scaffold.localization:
        specs.append(
            {
                "name": "report_localization",
                "description": (
                    "Report which files you believe must change to fix the issue, before editing. "
                    "Edits are rejected until this has been called."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "files": {"type": "array", "items": {"type": "string"}},
                        "rationale": {"type": "string"},
                    },
                    "required": ["files", "rationale"],
                },
            }
        )
    specs.append(
        {
            "name": "submit",
            "description": "Finish the task. Call once the fix is in place and verified.",
            "input_schema": {
                "type": "object",
                "properties": {"summary": {"type": "string", "description": "Root cause and fix, in a few sentences."}},
                "required": ["summary"],
            },
        }
    )
    return specs


class ToolExecutor:
    def __init__(self, sandbox: Sandbox, cfg: AgentConfig):
        self.sb = sandbox
        self.cfg = cfg
        self.files_read: set[str] = set()
        self.files_edited: set[str] = set()
        self.localized_files: list[str] | None = None
        self.test_runs = 0
        self.edited_since_last_test = False
        self.submitted_summary: str | None = None

    # ---------------------------------------------------------------- helpers
    def _resolve(self, path: str) -> str:
        """Absolute path inside the repo, or ValueError if it escapes."""
        root = self.sb.workdir.rstrip("/") or "/"
        full = posixpath.normpath(posixpath.join(root, path))
        if full != root and not full.startswith(root + "/"):
            raise ValueError(f"path '{path}' is outside the repository root")
        return full

    def _rel(self, full: str) -> str:
        root = self.sb.workdir.rstrip("/")
        return full[len(root) + 1 :] if full.startswith(root + "/") else full

    def _clip(self, text: str) -> str:
        return truncate_middle(text, self.cfg.tool_output_max_chars)

    # -------------------------------------------------------------- dispatch
    def execute(self, name: str, args: dict[str, Any]) -> ToolResult:
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            return ToolResult(f"Unknown tool '{name}'.", is_error=True)
        start = time.monotonic()
        try:
            result = handler(**args)
        except TypeError as e:
            result = ToolResult(f"Invalid arguments for {name}: {e}", is_error=True)
        except ValueError as e:
            result = ToolResult(str(e), is_error=True)
        except FileNotFoundError as e:
            result = ToolResult(f"File not found: {e}", is_error=True)
        except Exception as e:  # never let a tool crash the episode
            result = ToolResult(f"{type(e).__name__}: {e}", is_error=True)
        result.meta.setdefault("duration_s", round(time.monotonic() - start, 3))
        return result

    # ----------------------------------------------------------------- tools
    def _tool_bash(self, command: str) -> ToolResult:
        res = self.sb.exec(command)
        body = res.output.rstrip() or "(no output)"
        status = "TIMED OUT" if res.timed_out else f"exit code {res.exit_code}"
        return ToolResult(self._clip(f"[{status}]\n{body}"), is_error=False, meta={"exit_code": res.exit_code})

    def _tool_run_tests(self, command: str) -> ToolResult:
        result = self._tool_bash(command)
        self.test_runs += 1
        self.edited_since_last_test = False
        result.meta["test_run"] = True
        return result

    def _tool_read_file(self, path: str, start_line: int | None = None, end_line: int | None = None) -> ToolResult:
        full = self._resolve(path)
        content = self.sb.read_file(full)
        self.files_read.add(self._rel(full))
        lines = content.split("\n")
        total = len(lines) - (1 if lines and lines[-1] == "" else 0)
        start = max(1, start_line or 1)
        end = min(total, end_line or total)
        note = ""
        if start_line is None and end_line is None and total > MAX_READ_LINES:
            end = MAX_READ_LINES
            note = f"\n[showing lines 1-{MAX_READ_LINES} of {total}; pass start_line/end_line to see more]"
        if start > max(total, 1):
            return ToolResult(f"start_line {start} is past the end of the file ({total} lines).", is_error=True)
        chunk = "\n".join(lines[start - 1 : end])
        return ToolResult(self._clip(number_lines(chunk, start) + note))

    def _tool_edit_file(self, path: str, old_str: str, new_str: str) -> ToolResult:
        full = self._resolve(path)
        if old_str == "":
            return ToolResult("old_str must not be empty. Use create_file to make a new file.", is_error=True)
        if old_str == new_str:
            return ToolResult("old_str and new_str are identical; nothing to change.", is_error=True)
        content = self.sb.read_file(full)
        count = content.count(old_str)
        if count == 0:
            return ToolResult(
                f"old_str was not found in {path}. {_near_miss_hint(content, old_str)}"
                "Re-read the file and copy the exact text, including indentation.",
                is_error=True,
            )
        if count > 1:
            starts = _match_lines(content, old_str)
            return ToolResult(
                f"old_str matches {count} times in {path} (starting at lines {starts}). "
                "Include more surrounding context so it matches exactly once.",
                is_error=True,
            )
        updated = content.replace(old_str, new_str, 1)
        self.sb.write_file(full, updated)
        rel = self._rel(full)
        self.files_edited.add(rel)
        self.edited_since_last_test = True

        edit_line = content[: content.index(old_str)].count("\n") + 1
        new_lines = updated.split("\n")
        lo = max(1, edit_line - 4)
        hi = min(len(new_lines), edit_line + new_str.count("\n") + 4)
        snippet = number_lines("\n".join(new_lines[lo - 1 : hi]), lo)
        message = f"Edited {rel}. Lines {lo}-{hi} now read:\n{snippet}"
        if rel.endswith(".py"):
            check = self.sb.exec(
                f"(command -v python >/dev/null && python -m py_compile {shlex.quote(full)}) "
                f"|| python3 -m py_compile {shlex.quote(full)}"
            )
            if check.exit_code != 0:
                message += f"\n\nWARNING: the file no longer compiles:\n{check.output.strip()[-1500:]}"
        return ToolResult(self._clip(message))

    def _tool_create_file(self, path: str, content: str) -> ToolResult:
        full = self._resolve(path)
        if self.sb.file_exists(full):
            return ToolResult(f"{path} already exists. Use edit_file to change it.", is_error=True)
        self.sb.write_file(full, content)
        rel = self._rel(full)
        self.files_edited.add(rel)
        self.edited_since_last_test = True
        return ToolResult(f"Created {rel} ({content.count(chr(10)) + 1} lines).")

    def _tool_search(self, pattern: str, path: str = ".", include: str | None = None) -> ToolResult:
        full = self._resolve(path)
        inc = f" --include={shlex.quote(include)}" if include else ""
        cmd = (
            f"grep -rnE --binary-files=without-match --exclude-dir=.git{inc} "
            f"-e {shlex.quote(pattern)} {shlex.quote(full)} | head -n {SEARCH_MAX_LINES + 1}"
        )
        res = self.sb.exec(cmd)
        lines = [line.replace(self.sb.workdir.rstrip("/") + "/", "", 1) for line in res.stdout.splitlines()]
        if not lines:
            return ToolResult(f"No matches for /{pattern}/.")
        more = len(lines) > SEARCH_MAX_LINES
        body = "\n".join(lines[:SEARCH_MAX_LINES])
        if more:
            body += f"\n[more than {SEARCH_MAX_LINES} matches; narrow the pattern or path]"
        return ToolResult(self._clip(body))

    def _tool_find_files(self, name_pattern: str, path: str = ".") -> ToolResult:
        full = self._resolve(path)
        cmd = (
            f"find {shlex.quote(full)} -path '*/.git' -prune -o -type f -name {shlex.quote(name_pattern)} -print "
            f"| sort | head -n {FIND_MAX_LINES + 1}"
        )
        res = self.sb.exec(cmd)
        files = [self._rel(p) for p in res.stdout.splitlines() if p]
        if not files:
            return ToolResult(f"No files match '{name_pattern}'.")
        body = "\n".join(files[:FIND_MAX_LINES])
        if len(files) > FIND_MAX_LINES:
            body += f"\n[more than {FIND_MAX_LINES} files; narrow the pattern or path]"
        return ToolResult(body)

    def _tool_report_localization(self, files: list[str], rationale: str) -> ToolResult:
        if not files:
            return ToolResult("files must list at least one path.", is_error=True)
        cleaned = [self._rel(self._resolve(f)) for f in files]
        self.localized_files = cleaned
        return ToolResult(f"Localization recorded: {', '.join(cleaned)}. You may now edit files.")

    def _tool_submit(self, summary: str) -> ToolResult:
        self.submitted_summary = summary
        return ToolResult("Submitted.", meta={"submitted": True})


def _match_lines(content: str, needle: str) -> list[int]:
    out, start = [], 0
    while (idx := content.find(needle, start)) != -1:
        out.append(content[:idx].count("\n") + 1)
        start = idx + 1
    return out[:10]


def _near_miss_hint(content: str, old_str: str) -> str:
    """Point at lines that match the first line of old_str once whitespace is ignored."""
    first = next((line.strip() for line in old_str.split("\n") if line.strip()), "")
    if not first:
        return ""
    hits = [i for i, line in enumerate(content.split("\n"), 1) if line.strip() == first]
    if hits:
        return f"A line matching the start of old_str (ignoring whitespace) appears at line(s) {hits[:5]}. "
    return ""
