import { readFileSync } from "node:fs";
import { join } from "node:path";
import { spawnSync } from "node:child_process";

const coreRoles = new Set(["plan", "explore", "build", "test", "review"]);

export const AgentOSPlugin = async ({ client, directory }) => {
  const sessions = new Map();
  const run = (script, args = [], env = {}) => spawnSync(
    "bash",
    [join(directory, ".agent-context", script), ...args],
    { cwd: directory, encoding: "utf8", env: { ...process.env, ...env } },
  );
  const active = () => {
    try {
      const runID = readFileSync(join(directory, ".agent-context", "current-run"), "utf8").trim();
      const status = JSON.parse(readFileSync(join(directory, ".agent-context", "runs", runID, "status.json"), "utf8"));
      return { runID, status };
    } catch {
      return null;
    }
  };
  const checkpoint = (reason, sessionID) => {
    const current = active();
    if (!current?.status.lease || !sessionID) return;
    const owner = `opencode:${sessionID}`;
    if (current.status.lease.owner !== owner) return;
    run("hooks/checkpoint.sh", [], {
      RUN_ID: current.runID,
      AGENT_ROLE: current.status.current_role,
      CHECKPOINT_REASON: reason,
      NEXT_ACTION: current.status.next_action || "resume recorded work",
    });
  };
  const record = (type, detail = "") => run("hooks/log_event.sh", [type, detail]);

  return {
    "chat.message": async (input) => {
      sessions.set(input.sessionID, { agent: input.agent, model: input.model });
      const current = active();
      if (!current || !coreRoles.has(current.status.current_role)) return;
      const owner = `opencode:${input.sessionID}`;
      if (current.status.lease?.owner && current.status.lease.owner !== owner) {
        throw new Error(`Agent OS role is leased by ${current.status.lease.owner}`);
      }
      if (!current.status.lease) {
        const model = input.model || { providerID: "unknown", modelID: "unknown" };
        const claim = run("scripts/claim_role.sh", [
          current.status.current_role, "--owner", owner, "--runtime", "opencode",
          "--provider", model.providerID, "--model", model.modelID,
        ]);
        if (claim.status !== 0) throw new Error(claim.stderr.trim() || "Agent OS role claim failed");
      }
      const checked = run("scripts/validate_context.sh", [current.status.current_role, current.runID, "--lease-optional"]);
      if (checked.status !== 0) throw new Error(checked.stderr.trim() || "Agent OS context validation failed");
    },
    "experimental.chat.system.transform": async (_input, output) => {
      const current = active();
      if (!current || !coreRoles.has(current.status.current_role)) return;
      const context = run("scripts/context_for_role.sh", [current.status.current_role, current.runID]);
      if (context.status === 0) output.system.push(context.stdout.slice(0, 30000));
      else output.system.push(`HALT: repository context invalid\n${context.stderr.slice(0, 2000)}`);
    },
    "tool.execute.after": async (input) => {
      record("opencode_tool_complete", input.tool);
      run("scripts/update_changed_files.sh");
    },
    event: async ({ event }) => {
      if (event.type === "file.edited") {
        run("scripts/update_changed_files.sh", [event.properties.file]);
        return;
      }
      if (event.type === "session.compacted") {
        checkpoint("opencode_compaction", event.properties.sessionID);
        record("opencode_compacted", event.properties.sessionID);
        return;
      }
      if (event.type === "session.error") {
        checkpoint("opencode_session_error", event.properties.sessionID);
        record("opencode_session_error", event.properties.sessionID || "unknown");
        return;
      }
      if (event.type === "session.deleted") {
        const sessionID = event.properties.sessionID || event.properties.info?.id;
        checkpoint("opencode_session_deleted", sessionID);
        return;
      }
      if (event.type !== "session.idle") return;
      checkpoint("opencode_session_idle", event.properties.sessionID);
      const current = active();
      if (!current) return;
      const critic = run("hooks/critic_gate.sh", [], { RUN_ID: current.runID });
      if (critic.status !== 2) return;
      record("opencode_stop_blocked", critic.stderr.slice(-1000));
      const meta = sessions.get(event.properties.sessionID) || {};
      await client.session.promptAsync({
        path: { id: event.properties.sessionID },
        body: {
          agent: meta.agent,
          model: meta.model ? { providerID: meta.model.providerID, modelID: meta.model.modelID } : undefined,
          parts: [{ type: "text", text: critic.stderr.slice(-4000) }],
        },
      });
    },
  };
};

export default AgentOSPlugin;
