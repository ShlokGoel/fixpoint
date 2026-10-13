"""Execution sandboxes.

`DockerSandbox` is what experiments use: one container per task, no network,
resource-limited, destroyed afterwards. `LocalSandbox` runs in a host directory
and exists only for unit tests and the toy demo; it is not isolated.
"""

from __future__ import annotations

import posixpath
import shlex
import subprocess
import sys
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from .config import SandboxConfig

# Untracked files with these prefixes at the repo root are treated as scratch
# work (reproduction scripts etc.) and left out of the submitted patch.
SCRATCH_PREFIXES = ("reproduce", "repro_", "scratch", "debug_", "tmp_")
# Build/test byproducts that should never be part of a patch.
JUNK_PARTS = ("__pycache__/", ".pytest_cache/", ".mypy_cache/", ".egg-info/")
JUNK_SUFFIXES = (".pyc", ".pyo", ".so", ".o")


def is_excluded_untracked(path: str) -> bool:
    """New files left out of the submitted patch: scratch scripts and build byproducts."""
    if "/" not in path and path.startswith(SCRATCH_PREFIXES):
        return True
    return path.endswith(JUNK_SUFFIXES) or any(part in path or path.startswith(part) for part in JUNK_PARTS)


@dataclass
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool = False

    @property
    def output(self) -> str:
        if self.stderr and self.stdout:
            return f"{self.stdout}\n[stderr]\n{self.stderr}"
        return self.stdout or self.stderr


class Sandbox(ABC):
    workdir: str

    @abstractmethod
    def exec(self, command: str, timeout: int | None = None) -> ExecResult: ...

    @abstractmethod
    def read_file(self, path: str) -> str: ...

    @abstractmethod
    def write_file(self, path: str, content: str) -> None: ...

    def close(self) -> None:  # noqa: B027 - optional hook, default no-op
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def file_exists(self, path: str) -> bool:
        return self.exec(f"test -f {shlex.quote(path)}").exit_code == 0

    def head_commit(self) -> str:
        return self.exec("git rev-parse HEAD").stdout.strip()

    def git_diff(self) -> str:
        """The patch the agent produced, relative to the base commit.

        Staging happens in a throwaway index file, so calling this mid-episode
        (e.g. for a context-reset summary) never changes what the agent sees in
        `git status` / `git diff`.
        """
        untracked = self.exec("git ls-files --others --exclude-standard").stdout.splitlines()
        excluded = [p for p in untracked if is_excluded_untracked(p)]
        cmd = (
            'gitdir="$(git rev-parse --git-dir)" && idx="$gitdir/agent-diff-index.$$"; '
            "trap 'rm -f \"$idx\"' EXIT; "
            'cp "$gitdir/index" "$idx" 2>/dev/null || true; '
            'export GIT_INDEX_FILE="$idx"; git add -A'
        )
        if excluded:
            cmd += " && git reset -q -- " + " ".join(shlex.quote(p) for p in excluded)
        cmd += " && git diff --cached --no-color --no-ext-diff HEAD"
        result = self.exec(cmd)
        if result.exit_code != 0:
            raise RuntimeError(f"git diff failed: {result.output[:500]}")
        diff = result.stdout
        return diff if diff.endswith("\n") or not diff else diff + "\n"


class DockerSandbox(Sandbox):
    def __init__(self, image: str, cfg: SandboxConfig, name: str | None = None, docker_bin: str = "docker"):
        self.image = image
        self.cfg = cfg
        self.workdir = cfg.workdir
        self.docker = docker_bin
        self.name = name or f"fixpoint-{uuid.uuid4().hex[:10]}"
        self.started = False

    def _run(self, args: list[str], timeout: float, stdin: str | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(
            [self.docker, *args],
            input=stdin,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )

    def start(self) -> DockerSandbox:
        args = [
            "run", "-d",
            "--name", self.name,
            "--network", self.cfg.network,
            "--memory", self.cfg.memory,
            "--cpus", self.cfg.cpus,
            "-w", self.workdir,
            "--entrypoint", "sleep",
            self.image, "infinity",
        ]  # fmt: skip
        proc = self._run(args, timeout=300)
        if proc.returncode != 0:
            raise RuntimeError(f"docker run failed for {self.image}: {proc.stderr.strip()}")
        self.started = True
        return self

    def exec(self, command: str, timeout: int | None = None) -> ExecResult:
        limit = int(timeout or self.cfg.command_timeout_s)
        script = (
            f"{self.cfg.env_prefix}\n"
            f"cd {shlex.quote(self.workdir)} && "
            f"timeout --kill-after=5 {limit} bash -c {shlex.quote(command)}"
        )
        start = time.monotonic()
        try:
            proc = self._run(["exec", self.name, "bash", "-c", script], timeout=limit + 30)
        except subprocess.TimeoutExpired as e:
            return ExecResult(124, _s(e.stdout), _s(e.stderr), time.monotonic() - start, timed_out=True)
        return ExecResult(
            proc.returncode,
            proc.stdout,
            proc.stderr,
            round(time.monotonic() - start, 3),
            timed_out=proc.returncode in (124, 137),
        )

    def read_file(self, path: str) -> str:
        proc = self._run(["exec", "-w", self.workdir, self.name, "cat", "--", path], timeout=60)
        if proc.returncode != 0:
            if "No such file" in proc.stderr or "Is a directory" in proc.stderr:
                raise FileNotFoundError(path)
            raise RuntimeError(proc.stderr.strip())
        return proc.stdout

    def write_file(self, path: str, content: str) -> None:
        target = shlex.quote(path)
        script = f'mkdir -p "$(dirname {target})" && cat > {target}'
        proc = self._run(["exec", "-i", "-w", self.workdir, self.name, "sh", "-c", script], timeout=60, stdin=content)
        if proc.returncode != 0:
            raise RuntimeError(f"write failed for {path}: {proc.stderr.strip()}")

    def close(self) -> None:
        if self.started:
            try:
                for attempt in range(3):
                    try:
                        proc = self._run(["rm", "-f", self.name], timeout=120)
                        if proc.returncode == 0 or "No such container" in proc.stderr:
                            break
                        err = proc.stderr.strip()
                    except subprocess.TimeoutExpired:
                        err = "timeout"
                    time.sleep(2 * (attempt + 1))
                else:
                    print(f"warning: could not remove container {self.name}: {err}", file=sys.stderr)
            finally:
                self.started = False


class LocalSandbox(Sandbox):
    """Runs commands on the host in `root`. Not isolated: tests and demos only."""

    def __init__(self, root: str | Path, command_timeout_s: int = 60):
        self.root = Path(root).resolve()
        self.workdir = str(self.root)
        self.command_timeout_s = command_timeout_s

    def _host_path(self, path: str) -> Path:
        full = Path(posixpath.normpath(posixpath.join(self.workdir, path)))
        if full != self.root and self.root not in full.parents:
            raise PermissionError(f"path escapes sandbox root: {path}")
        return full

    def exec(self, command: str, timeout: int | None = None) -> ExecResult:
        limit = int(timeout or self.command_timeout_s)
        start = time.monotonic()
        try:
            proc = subprocess.run(
                ["bash", "-c", command],
                cwd=self.root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=limit,
            )
        except subprocess.TimeoutExpired as e:
            return ExecResult(124, _s(e.stdout), _s(e.stderr), time.monotonic() - start, timed_out=True)
        return ExecResult(proc.returncode, proc.stdout, proc.stderr, round(time.monotonic() - start, 3))

    def read_file(self, path: str) -> str:
        host = self._host_path(path)
        if not host.is_file():
            raise FileNotFoundError(path)
        return host.read_text(encoding="utf-8", errors="replace")

    def write_file(self, path: str, content: str) -> None:
        host = self._host_path(path)
        host.parent.mkdir(parents=True, exist_ok=True)
        host.write_text(content, encoding="utf-8")


def _s(value: bytes | str | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else value
