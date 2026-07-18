#!/usr/bin/env bash
# Terminal 3 of the Level-2 demo: the SIM node. Opens a distributed iex named
# sim@127.0.0.1 with the shared cookie, connects to the inference node's
# cluster, starts the sim env adapter + production ControlLoop wired to the
# running sim server, and binds `loop` in the REPL (via the sibling
# sim-node.iex.exs) -- so all you type is:
#
#   iex> Demo.SimNode.run_loop(loop, 200, 50)   # 200 ticks, 50ms apart
#
# and the arm moves in the sim-server viewer window.
#
#   ./scripts/demo/sim-node.sh
#
# Prereqs: sim-server-viewer.sh (Terminal 1) and inference-node.sh (Terminal 2)
# are already running. Override identity/address via DEMO_* env vars (they are
# passed through to the .iex.exs).

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

cd "$DEMO_REPO_ROOT"

# The .iex.exs reads these from the environment.
export DEMO_INFERENCE_NODE DEMO_COOKIE DEMO_SIM_ADDRESS

echo "sim node: $DEMO_SIM_NODE (cookie: $DEMO_COOKIE)"
echo "  -> inference node: $DEMO_INFERENCE_NODE"
echo "  -> sim server:     $DEMO_SIM_ADDRESS"

# --dot-iex loads sim-node.iex.exs INTO the shell session so `loop` stays bound.
exec iex \
  --name "$DEMO_SIM_NODE" \
  --cookie "$DEMO_COOKIE" \
  --dot-iex "$DEMO_REPO_ROOT/scripts/demo/sim-node.iex.exs" \
  -S mix
