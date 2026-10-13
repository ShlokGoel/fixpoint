"""Integration tests against a real container. Skipped unless an image is given:

docker build -f examples/Dockerfile.toy -t fixpoint-toy examples/
FIXPOINT_TEST_IMAGE=fixpoint-toy python -m pytest tests/test_docker_sandbox.py
"""

import os

import pytest

from agent.config import SandboxConfig
from agent.sandbox import DockerSandbox

IMAGE = os.environ.get("FIXPOINT_TEST_IMAGE")
pytestmark = pytest.mark.skipif(not IMAGE, reason="set FIXPOINT_TEST_IMAGE to run Docker tests")


@pytest.fixture
def sb():
    sandbox = DockerSandbox(IMAGE, SandboxConfig(command_timeout_s=10, memory="1g", cpus="1")).start()
    yield sandbox
    sandbox.close()


def test_exec_runs_in_workdir(sb):
    res = sb.exec("pwd && git rev-parse --is-inside-work-tree")
    assert res.exit_code == 0 and res.stdout.split() == ["/testbed", "true"]


def test_round_trip_preserves_exact_content(sb):
    content = "line with 'quotes' and \"double\" and $HOME and `ticks`\nünïcödé ✓\n\ttab\n"
    sb.write_file("/testbed/nested/dir/file.txt", content)
    assert sb.read_file("/testbed/nested/dir/file.txt") == content
    with pytest.raises(FileNotFoundError):
        sb.read_file("/testbed/missing.txt")


def test_timeout_is_enforced(sb):
    res = sb.exec("sleep 30", timeout=2)
    assert res.timed_out and res.duration_s < 15


def test_no_network(sb):
    res = sb.exec("python -c \"import urllib.request; urllib.request.urlopen('https://example.com', timeout=5)\"")
    assert res.exit_code != 0


def test_git_diff_and_cleanup():
    sandbox = DockerSandbox(IMAGE, SandboxConfig()).start()
    name = sandbox.name
    try:
        sandbox.write_file("/testbed/stats_utils/new_module.py", "X = 1\n")
        assert "stats_utils/new_module.py" in sandbox.git_diff()
    finally:
        sandbox.close()
    import subprocess

    out = subprocess.run(["docker", "ps", "-a", "--filter", f"name={name}", "-q"], capture_output=True, text=True)
    assert out.stdout.strip() == ""
