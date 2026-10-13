import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.config import AgentConfig  # noqa: E402
from agent.sandbox import LocalSandbox  # noqa: E402
from agent.task import Task  # noqa: E402

GIT = ["git", "-c", "user.email=t@example.com", "-c", "user.name=t"]


@pytest.fixture
def toy_sandbox(tmp_path):
    repo = tmp_path / "repo"
    shutil.copytree(ROOT / "examples" / "toy_repo", repo)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run([*GIT, "add", "-A"], cwd=repo, check=True)
    subprocess.run([*GIT, "commit", "-qm", "base"], cwd=repo, check=True)
    return LocalSandbox(repo, command_timeout_s=30)


@pytest.fixture
def cfg():
    c = AgentConfig()
    c.pricing.input = 2.0
    c.pricing.output = 10.0
    return c


@pytest.fixture
def toy_task():
    return Task(instance_id="toy__1", repo="example/toy", problem_statement="median is wrong for even lists")
