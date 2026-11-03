import pytest
from sureshot import Store
from sureshot.tasks import run
from test_router import QUESTIONS, FakeClaude, FakeDecider
from sureshot.router import Router


@pytest.fixture
def log(tmp_path):
    path = tmp_path / "sureshot.db"
    router = Router(FakeDecider(), FakeClaude(), log=Store(path))
    for text in ("standup", "interview with Sam", "1:1"):
        router.decide(text, QUESTIONS)
    return str(path)


def test_steps_return_what_the_command_line_prints(log):
    assert run("stats", db=log, sample=0.02)["decisions"] == 3
    assert run("calibrate", db=log, min_samples=1, tier=0) is not None
    assert isinstance(run("prune", db=log, costs={"tier 0": 0, "system2": 0.004}, min_answers=1), dict)


def test_failures_raise_with_the_command_lines_message(log, tmp_path):
    with pytest.raises(RuntimeError, match="no log at"):
        run("stats", db=str(tmp_path / "missing.db"))
    with pytest.raises(RuntimeError, match='"system2" cost per question'):
        run("prune", db=log, costs={"tier 0": 0})
    with pytest.raises(RuntimeError, match="exited with status 2"):
        run("stats", db=log, nope=1)
    with pytest.raises(ValueError, match="no step 'serve'"):
        run("serve")
