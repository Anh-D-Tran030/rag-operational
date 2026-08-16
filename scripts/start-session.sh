#!/bin/bash
# start-session.sh
# Run this when starting a session on either tool.
# Reads context_snapshot.md if it exists.

if [ -f .agent-context/context_snapshot.md ]; then
  echo "=== Resuming from previous session ==="
  cat .agent-context/context_snapshot.md
  echo "======================================="
  echo "Read .agent-context/context_snapshot.md and start from next_action."
else
  echo "No previous context found. Starting fresh."
  echo "Run explore agent first to generate .agent-context/explore.md"
fi
