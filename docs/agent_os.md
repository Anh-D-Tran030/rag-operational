# Provider-independent Agent OS

This document is the operating guide for the repository-owned agent harness.
Its central invariant is: **no task depends on a previous model's chat
history**. Every continuation-critical fact for substantial work is stored in
the active run directory.

## Architecture

The system has four deliberately separate layers:

1. `AGENTS.md` is the constitution: permanent safety, permissions, lanes, gate
   semantics, startup, checkpoint, and handoff rules. It contains no task state.
2. `.agent-context/agents/` contains seven provider-neutral roles. PLAN,
   EXPLORE, BUILD, TEST, and REVIEW are gates; compressor and docs-writer are
   supporting roles.
3. `.agent-context/runs/<run-id>/` is operational state. `current-run` points
   to the authoritative run. Root `status.json` is an atomic compatibility
   mirror while legacy consumers remain.
4. `.agent-context/memory/` is durable learning. Per-run failures go to
   `attempts.jsonl`; reusable facts continue through the existing episodic,
   reflector, pending-consolidation, and human-review path.

Each full run contains:

```text
task.md              objective                 status.json       machine state
handoff.md           60-second continuation    changed-files.json Git-derived delta
plan.md              PLAN artifact             decisions.jsonl   deliberate choices
explore.md           EXPLORE artifact          attempts.jsonl    meaningful attempts
build.md             BUILD record              events.jsonl      lifecycle events
test.md              TEST/compressed failures  commands.log      selected verification
review.md            REVIEW artifact           docs.md           support-role output
```

Verified artifact hashes in `status.json` make gate output immutable. If one
is wrong, create a numbered amendment and explicitly reopen/revalidate affected
gates; do not overwrite history.

## Bootstrapping another repository

Run the repository-owned bootstrap script from this project:

```bash
bash .agent-context/scripts/bootstrap_new_project.sh /path/to/target
```

The script copies only provider-neutral Agent OS infrastructure. It then runs
`ground_target.sh` inside the target to generate the target's canonical
`AGENTS.md`, `project.md`, `conventions.md`, `domain.md`, and representative
examples. Unsupported facts are recorded as unknown. It does not copy this
repository's domain rules, thresholds, source examples, or application
instructions. `AGENTS.md` is the only canonical instruction file; a target
does not need Claude Code or a Claude installation.

## Data flow

```text
AGENTS.md + provider-neutral role
                  │
current-run → validate_context.sh → context_for_role.sh
                  │                         │
                  │                    worker claims lease
                  │                         │
                  └──── run artifacts/state ◄──── work
                                              │
                          decisions / attempts / commands / Git manifest
                                              │
                           checkpoint or compressor → concise handoff.md
                                              │
                           next runtime validates and resumes
                                              │
                         TEST sensor → REVIEW → Codex acceptance
```

## Startup and shutdown

A fresh worker reads `AGENTS.md`, its full role file, then runs:

```bash
bash .agent-context/scripts/context_for_role.sh build
bash .agent-context/scripts/claim_role.sh build \
  --owner codex-build-01 --runtime codex --provider openai --model gpt-5.6-sol
```

The context command resolves the active run, validates ordering, hashes, Git
HEAD, required files, and lease legality, then prints bounded task/handoff and
relevant upstream artifacts. `HALT` is terminal until the state is corrected.

At a meaningful stop, record only useful facts, then checkpoint:

```bash
bash .agent-context/scripts/record_decision.sh build \
  "Keep run status authoritative" "Avoid two sources of truth" "root authority" "status.json"
bash .agent-context/scripts/record_attempt.sh build \
  "local worker health" "minimal Ollama invocation" failed "timed out"
bash .agent-context/scripts/record_command.sh PASS \
  "bash .agent-context/tests/pipeline/run_tests.sh" "44/44 cases passed"
bash .agent-context/scripts/update_changed_files.sh
AGENT_ROLE=build CHECKPOINT_REASON=runtime_switch \
  NEXT_ACTION="OpenCode runs the next BUILD step" \
  NEXT_RUNTIME=opencode NEXT_PROVIDER=deepseek NEXT_MODEL=deepseek-v4-pro \
  bash .agent-context/hooks/checkpoint.sh
```

The same handoff contract is used for role/model switches, interruption,
blocked state, escalation, compaction, supported shutdown events, and recovery.
It is concise and excludes secrets.

## Routing and escalation

`providers/models.yaml` is an observed capability registry;
`providers/routing.yaml` is deterministic policy. Capability tiers and worker
selection are repository-specific configuration, not facts copied from the
bootstrap source project. Selection remains manual and unavailable entries
fall through the role's ordered candidates:

```bash
bash .agent-context/scripts/switch_model.sh build opencode 2
bash .agent-context/scripts/switch_model.sh test opencode 0
bash .agent-context/scripts/switch_model.sh review codex 3
```

Escalation is capability-based rather than provider-based. The target project
must establish its own model availability and risk policy from evidence. No
quota signal is inferred and telemetry does not route work automatically.

Bootstrap does not claim that any provider, model, credential, or local service
is available. Those facts are discovered and recorded per target repository.

## Ownership and recovery

One worker owns one role through a time-bounded lease. A second claim is
rejected. An active lease is never stolen; a stale one requires explicit
recovery:

```bash
bash .agent-context/scripts/resume_run.sh
bash .agent-context/scripts/claim_role.sh build \
  --owner codex-recovery-01 --runtime codex --provider openai \
  --model gpt-5.6-sol --recover-stale
```

`resume_run.sh` is read-only. It reports stage, prior worker, lease, last
verification, exact next action, and recommended worker. Unexpected Git HEAD
changes or modified verified artifacts cause validation to halt; recovery never
resets, checks out, cleans, or discards files.

## Runtime trust and enforcement

Codex is the initial orchestrator, gate-advancement authority, high-risk
decision maker, and final acceptance authority. Its hooks archive/checkpoint on
compaction and checkpoint on session end; its Stop hook runs the real sensor and
blocks a red turn.

OpenCode is a provider-neutral worker runtime. Repository agent wrappers point
to the same role definitions. The installed plugin claims leases, validates and
preloads run context, refreshes changed files/events after tools, checkpoints on
idle/error/delete/compaction, and sends critic failures back into an idle
session. OpenCode has no identical hard Stop enforcement, so it does not
self-certify final completion. Codex performs acceptance.

Runtime permissions remain least-privilege by role. Never use handoffs, logs,
telemetry, attempts, decisions, events, or memory for credentials. Production
deployment, destructive database/cloud/Git work, credential rotation, security
policy changes, disabling checks, or gate-invariant changes require explicit
human approval.

## Telemetry

Each run status contains nullable/counter telemetry for runtime, provider,
model, role, measured duration, handoff count, heal iterations, first-pass test
success, test/review result, and review rejection count. Input/output tokens and
cost stay `null` unless a runtime supplies reliable values. Telemetry is an
observation scaffold only; no automatic cost or quota routing exists.

## Daily use

Fast-lane work needs no run:

```text
Codex prompt: Fix <small issue> using the fast lane. Read AGENTS.md, keep the
change scoped, run relevant verification, and review the diff.
```

Start and inspect a full run:

```bash
bash .agent-context/scripts/new_run.sh --task "Implement <objective>"
bash .agent-context/scripts/resume_run.sh
bash .agent-context/scripts/context_for_role.sh plan
```

Run PLAN with Codex:

```bash
bash .agent-context/scripts/claim_role.sh plan --owner codex-plan-01 \
  --runtime codex --provider openai --model gpt-5.6-sol
codex "Run PLAN for the active repository run. Use only repository-owned state."
```

Hand BUILD to configured OpenCode/DeepSeek after PLAN and EXPLORE are verified:

```bash
bash .agent-context/scripts/switch_model.sh build opencode deepseek-v4-pro
opencode run --agent build --model deepseek/deepseek-v4-pro \
  "Run BUILD for the active repository run. Start with context_for_role.sh."
```

Return to Codex:

```bash
bash .agent-context/scripts/switch_model.sh build codex 3
codex "Resume the active BUILD from repository handoff state; do not rely on chat history."
```

Use a local worker only when registry health is `available: true`:

```bash
bash .agent-context/scripts/switch_model.sh test opencode 0
opencode run --agent tester --model ollama/qwen2.5-coder:7b-16k \
  "Run TEST for the active repository run from repository state."
```

After closing the editor, run `resume_run.sh`, then its printed context and
claim commands. Inspect structured status with:

```bash
RUN_ID=$(cat .agent-context/current-run)
jq . ".agent-context/runs/$RUN_ID/status.json"
```

Finish only after gate-specific verification and Codex review:

```bash
bash .agent-context/scripts/verify_run.sh "$(cat .agent-context/current-run)" --complete
codex review
```

The fast lane remains intentionally lightweight: no run directory, lease, or
gate transitions for a small fix, but the same permissions, relevant tests,
critic sensor, and diff review still apply.
