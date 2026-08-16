#!/bin/bash
# end-session.sh — run before closing Claude Code or switching to OpenCode.
# Writes a compact context_snapshot.md (max 10 lines) to .agent-context/.

TOOL=${1:-"claude-code"}
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Auto-detect last artifact written (most recently modified file in src/ or tests/)
LAST_ARTIFACT=$(find src tests .agent-context -type f -newer .agent-context/explore.md 2>/dev/null \
  | grep -v '__pycache__' | head -1)
LAST_ARTIFACT=${LAST_ARTIFACT:-"none"}

cat > .agent-context/context_snapshot.md << EOF
tool:          $TOOL
switched_at:   $TIMESTAMP
completed:     ${2:-"none"}
interrupted:   ${3:-"none"}
last_artifact: $LAST_ARTIFACT
next_action:   ${4:-"[fill in: one sentence]"}
blockers:      none
EOF

echo "Snapshot written ($(wc -l < .agent-context/context_snapshot.md) lines):"
cat .agent-context/context_snapshot.md
