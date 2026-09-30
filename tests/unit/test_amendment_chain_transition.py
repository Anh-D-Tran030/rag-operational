"""Chained PLAN amendments: EXPLORE blocker records and unexplored supersedes."""
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
BLOCKED = "msg-explore-blocked"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _amendment(run: Path, number: str) -> dict:
    path = run / f"plan-amendment-{number}.md"
    path.write_text(f"---\namendment_id: PLAN-{number}\nrun_id: run-fixture\n---\n")
    return {"id": f"PLAN-{number}", "path": path.name, "sha256": _hash(path)}


def _fixture(tmp_path: Path, *, resolved: bool = True) -> tuple[Path, Path, dict]:
    ctx = tmp_path / "project/.agent-context"
    run = ctx / "runs/run-fixture"
    (ctx / "scripts").mkdir(parents=True)
    run.mkdir(parents=True)
    shutil.copyfile(SOURCE, ctx / "scripts" / SOURCE.name)
    (ctx / "scripts/verify_run.sh").write_text("#!/bin/sh\nexit 0\n")
    (ctx / "current-run").write_text("run-fixture\n")
    (ctx / ".lock").touch()
    (run / "events.jsonl").touch()
    (run / "plan.md").write_text("plan\n")
    (run / "explore.md").write_text("original explore\n")
    (run / "explore-amendment-001.md").write_text("---\namendment: PLAN-001\n---\nblockers\n")
    first = _amendment(run, "001")
    _amendment(run, "002")
    _amendment(run, "003")
    messages = [{"id": BLOCKED, "from": "explore", "type": "blocked_request",
                 "run_id": "run-fixture", "resolves": None}]
    if resolved:
        messages.append({"id": "msg-plan", "from": "plan", "type": "resolution",
                         "run_id": "run-fixture", "resolves": BLOCKED})
    (ctx / "intercom.jsonl").write_text("".join(json.dumps(m) + "\n" for m in messages))
    status = {
        "run_id": "run-fixture", "state": "handed_off", "lease": None, "blocked": False,
        "current_worker": None, "telemetry": {"test_result": "fail"},
        "plan_amendments": [first], "active_plan_amendment": first,
        "gates": {
            "plan": {"status": "verified", "artifact_sha256": _hash(run / "plan.md")},
            "explore": {"status": "ready", "artifact_sha256": None,
                        "artifact_path": "explore-amendment-001.md"},
            "build": {"status": "pending", "artifact_sha256": None,
                      "artifact_path": "build-amendment-001.md"},
            "test": {"status": "pending", "artifact_sha256": None,
                     "artifact_path": "test-attempt-002.md", "attempt_number": 2},
            "review": {"status": "pending", "artifact_sha256": None,
                       "artifact_path": "review-amendment-001.md"},
        },
    }
    for path in (run / "status.json", ctx / "status.json"):
        path.write_text(json.dumps(status))
    return ctx / "scripts" / SOURCE.name, run, status


def _call(script: Path, amendment: str, *mode: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), "--run-id", "run-fixture", "--amendment", amendment,
         "--reopen", "explore", "--owner", "tester", "--reason", "chain", *mode],
        capture_output=True, text=True, check=False,
    )


def test_blocker_record_then_supersede_builds_auditable_chain(tmp_path: Path) -> None:
    script, run, _ = _fixture(tmp_path)
    blocker_hash = _hash(run / "explore-amendment-001.md")
    result = _call(script, "PLAN-002", "--accept-explore-blockers", BLOCKED)
    assert result.returncode == 0, result.stderr
    status = json.loads((run / "status.json").read_text())
    record = status["gate_history"][0]
    assert record["gate_record"]["status"] == "verified"
    assert record["gate_record"]["disposition"] == "blockers_recorded"
    assert record["gate_record"]["blockers_ref"] == BLOCKED
    assert record["artifact_sha256"] == blocker_hash
    assert status["gates"]["explore"] == {
        "status": "ready", "artifact_sha256": None, "locked_at": None,
        "artifact_path": "explore-amendment-002.md",
    }
    assert status["gates"]["test"]["artifact_path"] == "test-attempt-002.md"
    assert status["gates"]["build"]["status"] == "pending"

    result = _call(script, "PLAN-003", "--supersede-unexplored")
    assert result.returncode == 0, result.stderr
    status = json.loads((run / "status.json").read_text())
    superseded = status["gate_history"][-4]
    assert superseded["gate"] == "explore" and superseded["artifact_sha256"] is None
    assert superseded["gate_record"]["disposition"] == "superseded_unexplored"
    assert superseded["gate_record"]["superseded_by"] == "PLAN-003"
    assert [a["id"] for a in status["plan_amendments"]] == ["PLAN-001", "PLAN-002", "PLAN-003"]
    assert status["next_action"].endswith("PLAN + PLAN-001 + PLAN-002 + PLAN-003")
    assert status["gates"]["explore"]["artifact_path"] == "explore-amendment-003.md"
    assert status["gates"]["test"]["attempt_number"] == 2
    assert _hash(run / "explore-amendment-001.md") == blocker_hash


def test_blocker_record_annotates_already_locked_explore(tmp_path: Path) -> None:
    script, run, status = _fixture(tmp_path)
    blocker_hash = _hash(run / "explore-amendment-001.md")
    status["gates"]["explore"].update(
        {"status": "verified", "artifact_sha256": blocker_hash, "locked_at": "t0"}
    )
    status["gates"]["build"]["status"] = "ready"  # lock_gate.sh promoted BUILD
    (run / "status.json").write_text(json.dumps(status))
    result = _call(script, "PLAN-002", "--accept-explore-blockers", BLOCKED)
    assert result.returncode == 0, result.stderr
    new = json.loads((run / "status.json").read_text())
    record = new["gate_history"][0]["gate_record"]
    assert (record["locked_at"], record["disposition"]) == ("t0", "blockers_recorded")
    assert new["gates"]["build"]["status"] == "pending"
    assert new["gates"]["explore"]["status"] == "ready"


@pytest.mark.parametrize("change", ["unresolved", "wrong_amendment", "no_mode", "supersede"])
def test_blocker_record_rejects_unsafe_state(tmp_path: Path, change: str) -> None:
    script, run, status = _fixture(tmp_path, resolved=change != "unresolved")
    mode: tuple[str, ...] = ("--accept-explore-blockers", BLOCKED)
    if change == "wrong_amendment":
        (run / "explore-amendment-001.md").write_text("---\namendment: PLAN-009\n---\nx\n")
    elif change == "no_mode":
        mode = ()
    elif change == "supersede":
        mode = ("--supersede-unexplored",)  # EXPLORE artifact already written
    before = copy.deepcopy(status)
    assert _call(script, "PLAN-002", *mode).returncode != 0
    assert json.loads((run / "status.json").read_text()) == before


def test_supersede_refuses_started_downstream_gate(tmp_path: Path) -> None:
    script, run, status = _fixture(tmp_path)
    (run / "explore-amendment-001.md").unlink()
    status["gates"]["build"]["status"] = "ready"
    (run / "status.json").write_text(json.dumps(status))
    assert _call(script, "PLAN-002", "--supersede-unexplored").returncode != 0
    assert json.loads((run / "status.json").read_text()) == status
