#!/usr/bin/env bash
# Shared helpers for the Level-2 closed-loop demo launchers (scripts/demo/*).
# Sourced by sim-server-viewer.sh, inference-node.sh, sim-node.sh.

set -euo pipefail

# Repo root, regardless of where the script is invoked from.
DEMO_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Shared cluster identity -- both BEAM nodes must agree. Overridable via env.
: "${DEMO_COOKIE:=demo}"
: "${DEMO_INFERENCE_NODE:=inference@127.0.0.1}"
: "${DEMO_SIM_NODE:=sim@127.0.0.1}"
: "${DEMO_SIM_ADDRESS:=tcp://127.0.0.1:5556}"

# Resolve the local smolvla_base checkpoint snapshot directory (the hash-named
# dir under the HF cache). Prints the path, or exits with a clear message if
# the checkpoint has not been downloaded.
demo_resolve_checkpoint() {
  local hub="${HF_HOME:-$HOME/.cache/huggingface}/hub"
  local snaps="$hub/models--lerobot--smolvla_base/snapshots"
  if [ ! -d "$snaps" ]; then
    echo "error: smolvla_base checkpoint not found under $snaps" >&2
    echo "       download it first (it is ~1.1GB), e.g. via the HF hub, then re-run." >&2
    return 1
  fi
  # Newest snapshot dir (there is normally exactly one).
  local snap
  snap="$(ls -dt "$snaps"/*/ 2>/dev/null | head -1)"
  if [ -z "$snap" ]; then
    echo "error: no snapshot directory under $snaps" >&2
    return 1
  fi
  # Strip the trailing slash.
  echo "${snap%/}"
}

# The venv python / mjpython the sim server runs under.
demo_python() { echo "$DEMO_REPO_ROOT/.venv/bin/python"; }
demo_mjpython() { echo "$DEMO_REPO_ROOT/.venv/bin/mjpython"; }
