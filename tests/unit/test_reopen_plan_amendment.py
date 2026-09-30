"""Isolated checks for the PLAN amendment control-plane transition."""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[2] / ".agent-context/scripts/reopen_plan_amendment.py"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict]:
    root = tmp_path / "project"
    ctx = root / ".agent-context"
    scripts = ctx / "scripts"
    run = ctx / "runs/run-fixture"
    scripts.mkdir(parents=True)
    run.mkdir(parents=True)
    shutil.copyfile(SOURCE, scripts / SOURCE.name)
    (scripts / "verify_run.sh").write_text("#!/bin/sh\nexit 0\n")
    (ctx / "current-run").write_text("run-fixture\n")
    (ctx / ".lock").touch()
    for name in ("plan", "explore", "build", "test"):
        (run / f"{name}.md").write_text(f"original {name}\n")
    (run / "plan-amendment-001.md").write_text(
        "---\namendment_id: PLAN-001\nrun_id: run-fixture\nstatus: proposed\n---\n"
    )
    status = {
        "run_id": "run-fixture", "state": "handed_off", "current_stage": "test",
        "current_role": "test", "current_worker": {
            "runtime": "claude", "provider": "anthropic", "model": "sonnet", "owner_id": None
        }, "lease": None, "blocked": False, "telemetry": {"test_result": "pass"},
        "gates": {
            "plan": {"status": "verified", "artifact_sha256": _hash(run / "plan.md")},
            "explore": {"status": "verified", "artifact_sha256": _hash(run / "explore.md")},
            "build": {"status": "built", "artifact_sha256": _hash(run / "build.md")},
            "test": {"status": "in_progress", "artifact_sha256": None, "heal_attempt": 1},
            "review": {"status": "pending", "artifact_sha256": None},
        },
        "last_verification": {"result": "FAIL"},
    }
    (run / "status.json").write_text(json.dumps(status))
    (ctx / "status.json").write_text(json.dumps(status))
    (run / "events.jsonl").touch()
    return scripts / SOURCE.name, run, status


def _call(script: Path, *, amendment: str = "PLAN-001") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), "--run-id", "run-fixture", "--amendment",
         amendment, "--reopen", "explore", "--owner", "tester", "--reason", "failure feedback"],
        capture_output=True, text=True, check=False,
    )


def test_reopen_preserves_verified_artifacts_and_failed_attempt(tmp_path: Path) -> None:
    script, run, original = _fixture(tmp_path)
    before = {name: _hash(run / f"{name}.md") for name in ("plan", "explore", "build", "test")}
    result = _call(script)
    assert result.returncode == 0, result.stderr
    status = json.loads((run / "status.json").read_text())
    assert status == json.loads((run.parents[1] / "status.json").read_text())
    assert [status["gates"][gate]["status"] for gate in
            ("plan", "explore", "build", "test", "review")] == [
                "verified", "ready", "pending", "pending", "pending"
            ]
    assert status["gates"]["plan"] == original["gates"]["plan"]
    assert status["gates"]["explore"]["artifact_path"] == "explore-amendment-001.md"
    assert status["gates"]["build"]["artifact_path"] == "build-amendment-001.md"
    assert status["gates"]["test"]["artifact_path"] == "test-attempt-002.md"
    assert status["gates"]["test"]["attempt_number"] == 2
    assert status["gates"]["test"]["heal_attempt"] == 1
    assert status["telemetry"]["test_result"] == "fail"
    assert status["active_plan_amendment"]["sha256"] == _hash(run / "plan-amendment-001.md")
    assert {entry["gate"]: entry["artifact_sha256"] for entry in status["gate_history"]} == {
        "explore": before["explore"], "build": before["build"],
        "test": before["test"], "review": None,
    }
    assert before == {name: _hash(run / f"{name}.md") for name in before}
    messages = (run.parents[1] / "intercom.jsonl").read_text().splitlines()
    assert len(messages) == 1
    assert json.loads(messages[0])["type"] == "broadcast"
    assert _call(script).returncode != 0  # replay cannot rewind a second time


@pytest.mark.parametrize("change", ["lease", "review", "amendment_run", "explore"])
def test_reopen_rejects_unsafe_state(tmp_path: Path, change: str) -> None:
    script, run, status = _fixture(tmp_path)
    if change == "lease":
        status["lease"] = {"role": "test", "owner": "other"}
    elif change == "review":
        status["gates"]["review"]["status"] = "verified"
    elif change == "explore":
        status["gates"]["explore"]["status"] = "ready"
    else:
        (run / "plan-amendment-001.md").write_text(
            "---\namendment_id: PLAN-001\nrun_id: run-other\n---\n"
        )
    (run / "status.json").write_text(json.dumps(status))
    (run.parents[1] / "status.json").write_text(json.dumps(status))
    before = copy.deepcopy(status)
    result = _call(script)
    assert result.returncode != 0
    assert json.loads((run / "status.json").read_text()) == before
