# demo — glossary

_Register_: simulation and demo-rig vocabulary — sim env, sim node, sim server.
This context's terms name the demo scaffold, never the production system it
exercises; where a concept belongs to
[control-loop](../control-loop/CONTEXT.md) or
[model-runtime](../model-runtime/CONTEXT.md), this context links the owner and
never redefines.

### Demo rig {#term-demo-rig}

The whole runnable demonstration: two BEAM nodes on one machine — a
[sim node](#term-sim-node) running a simulated SO-101 arm and the production
control loop, joined over a BEAM cluster to a Mac
[inference node](#term-inference-node) running SmolVLA inference. A scaffold
that exercises the production [control-loop](../control-loop/design.md) and
[model-runtime](../model-runtime/design.md) contexts as a closed
perception→action loop; it owns no model or queue logic of its own. _Avoid_:
"the app" (too generic to name the specific two-node assembly).

### Sim node {#term-sim-node}

The BEAM/OTP node that hosts the [sim env adapter](#term-sim-env-adapter), the
production [ControlLoop](../control-loop/design.md), and the connection to the
[sim server](#term-sim-server). One of the two nodes in the
[demo rig](#term-demo-rig)'s cluster; the other is the
[inference node](#term-inference-node). Runs on the Mac alongside the inference
node for the local demo — its identity is its role (driving the simulated loop),
not its hardware. _Avoid_: "Pi node" (there is no Raspberry Pi — the simulation
replaces hardware entirely, per
[ADR-0011](../../adr/0011-demo-is-a-simulated-closed-loop.md#adr-0011)).

### Inference node {#term-inference-node}

The Mac as a single BEAM/OTP node hosting the
[inference server](../model-runtime/design.md) — the process holding the loaded
emily-native SmolVLA model that answers
[infer_action](../model-runtime/CONTEXT.md#term-infer-action-port) calls from
the [sim node](#term-sim-node) across the cluster. Named for its role in the
demo; the server process it hosts is owned by
[model-runtime](../model-runtime/design.md), not this context.

### Sim env adapter {#term-sim-env-adapter}

The [sim node](#term-sim-node) unit that owns the whole simulation seam: it
drives one LeRobot/MuJoCo gym environment — `reset`, `step(action)`, `render` —
and exposes both interfaces the production
[ControlLoop](../control-loop/design.md) needs. As the loop's
[observation source](../control-loop/CONTEXT.md#term-observation-source) it
returns the current [observation](../model-runtime/CONTEXT.md#term-observation)
(the env's rendered frame plus the arm's state and the fixed instruction); as
the loop's actuator sink it applies each popped
[action](../model-runtime/CONTEXT.md#term-action-chunk) by calling `step`. One
unit, because in a gym env producing the next observation and consuming the
action are two halves of a single step cycle — they cannot be separated. It
reaches the Python environment through the [sim server](#term-sim-server).
_Avoid_: "camera capture" and "virtual bot" (the retired open-loop split — the
sim couples obs-out and action-in into one entity, so two units mis-carve it).

### Sim server {#term-sim-server}

The long-lived Python process that wraps the LeRobot/MuJoCo gym environment and
answers `reset` / `step` / `render` requests over a ZeroMQ REQ/REP socket with
MessagePack framing, mirroring [model_runtime_server](../control-loop/design.md)'s
own transport. It is the single seam between the Elixir
[sim node](#term-sim-node) and the Python physics engine, the demo's counterpart
to the inference path's Python fallback server. Runs headless by default;
an optional [sim viewer](#term-sim-viewer) mode opens a live window onto the
same env. See
[ADR-0012](../../adr/0012-sim-env-bridged-via-python-sim-server-over-zeromq.md#adr-0012).
_Avoid_: "simulator" alone (names the physics engine inside it, not the
request-answering server process the sim node talks to).

### Sim viewer {#term-sim-viewer}

The optional live 3D window onto the running simulation — a `mujoco.viewer`
passive viewer attached to the exact env instance the
[sim server](#term-sim-server) steps, so it shows the SO-101 arm moving under
real inference in real time. A dev-time aid, off by default; enabling it makes
the server serve ZeroMQ on a background thread and hold the main thread for the
window (launched via `mjpython` on macOS). Presentation only — it reads the
simulation, never drives it. See
[ADR-0013](../../adr/0013-live-mujoco-viewer-mode-on-the-sim-server.md#adr-0013).
_Avoid_: "GUI" (too generic — names the toolkit category, not the live window
onto this specific simulation); "dashboard" (implies aggregated metrics, not a
direct 3D view).
