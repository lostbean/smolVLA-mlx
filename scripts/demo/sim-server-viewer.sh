#!/usr/bin/env bash
# Terminal 1 of the Level-2 demo: the Python sim server WITH the live 3D
# MuJoCo window (ADR-0013). Leave this running; the arm moves in the window
# once the sim node drives the loop (scripts/demo/sim-node.sh).
#
# macOS needs mjpython (the main-thread launcher) for the live window; this
# script uses it. On Linux plain python works, but this script prefers
# mjpython when present and falls back to python otherwise.
#
#   ./scripts/demo/sim-server-viewer.sh
#
# Override the bind address with DEMO_SIM_ADDRESS=tcp://127.0.0.1:5556.
# Pass --headless as the first arg to run WITHOUT the window (plain server).

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

cd "$DEMO_REPO_ROOT"

viewer_flag="--viewer"
runner="$(demo_mjpython)"
if [ "${1:-}" = "--headless" ]; then
  viewer_flag=""
  runner="$(demo_python)"
  shift
elif [ ! -x "$runner" ]; then
  echo "note: mjpython not found at $runner; falling back to python (no live window on macOS)." >&2
  runner="$(demo_python)"
fi

echo "starting sim server on $DEMO_SIM_ADDRESS ${viewer_flag:+(with live viewer)}"
echo "  (close the window to shut the server down cleanly)"
exec "$runner" -m sim_server --address "$DEMO_SIM_ADDRESS" $viewer_flag "$@"
