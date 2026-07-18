#!/usr/bin/env bash
# Terminal 2 of the Level-2 demo: the INFERENCE node. Opens a distributed iex
# named inference@127.0.0.1 with the shared cookie, and auto-loads the real
# smolvla_base checkpoint into a named InferenceServer on boot (via the sibling
# inference-node.iex.exs, whose bindings stay live in the REPL) -- so you land
# in a live REPL with the model already loading, nothing to type.
#
#   ./scripts/demo/inference-node.sh
#
# The checkpoint snapshot path is resolved for you. Loading ~1.1GB takes a few
# seconds; watch for "[inference node ready ...]". Leave this running.
# Override identity via DEMO_INFERENCE_NODE / DEMO_COOKIE.

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

cd "$DEMO_REPO_ROOT"

DEMO_CHECKPOINT="$(demo_resolve_checkpoint)"
export DEMO_CHECKPOINT
echo "inference node: $DEMO_INFERENCE_NODE (cookie: $DEMO_COOKIE)"
echo "loading checkpoint: $DEMO_CHECKPOINT"

# --dot-iex loads inference-node.iex.exs INTO the shell session (bindings
# survive, unlike --eval); it reads DEMO_CHECKPOINT and starts the server.
exec iex \
  --name "$DEMO_INFERENCE_NODE" \
  --cookie "$DEMO_COOKIE" \
  --dot-iex "$DEMO_REPO_ROOT/scripts/demo/inference-node.iex.exs" \
  -S mix
