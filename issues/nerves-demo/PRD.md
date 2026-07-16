Status: ready-for-agent
Category: enhancement

# nerves-demo — runnable closed-loop simulated vision→action demo

Source: the [demo context](../../docs/design/demo/design.md) redesign landed
2026-07-16 — the demo became a **closed** perception→action loop over a
LeRobot/MuJoCo simulation, superseding its earlier open-loop scaffold. See the
[sim env adapter](../../docs/design/demo/CONTEXT.md#term-sim-env-adapter)
(component 01.1), the [sim node](../../docs/design/demo/CONTEXT.md#term-sim-node)
wiring (01.2), the control-loop
[observation source](../../docs/design/control-loop/CONTEXT.md#term-observation-source)
seam, the [InferenceServer](../../docs/design/model-runtime/design.md)
(model-runtime 01.5), and ADRs
[0010](../../docs/adr/0010-beam-distribution-orthogonal-to-infer-action-port.md#adr-0010),
[0011](../../docs/adr/0011-demo-is-a-simulated-closed-loop.md#adr-0011),
[0012](../../docs/adr/0012-sim-env-bridged-via-python-sim-server-over-zeromq.md#adr-0012).
This is that design delta's pending-ledger `build` entries sliced into a
dependency-ordered sequence of tracer-bullet issues.

The demo: a LeRobot/MuJoCo simulation of the SO-101 arm (the robot the
`lerobot/svla_so101_pickplace` checkpoint was trained on) doing pick-and-place,
driven as a **closed loop** — `env.step(action)` moves the simulated arm AND
returns the next rendered frame, so SmolVLA's inference on each frame drives the
next actions and the arm's motion is observable. It runs as two BEAM nodes on
one Mac: a sim node (sim env adapter + production `ControlLoop`) clustered to an
inference node (the `InferenceServer` running the real emily-native forward
pass), the cross-node `infer_action` call being native BEAM distribution
(ADR-0010), and the sim itself bridged to Elixir through a Python sim server
over ZeroMQ (ADR-0012).

## What changed from the earlier open-loop demo

The first design of this feature was an **open loop**: a real webcam fed images
to SmolVLA and a virtual bot merely rendered the returned actions — nothing the
bot did affected what the camera next saw. That proved the plumbing but never
showed the policy attempting a task. This redesign (ADR-0011) replaces it with a
simulated **closed** loop: simulated MuJoCo dynamics are now in scope, while real
actuator/kinematics/safety on physical hardware stays a hard no-goal, and no
Raspberry Pi, camera, or NERVES firmware is built. The earlier open-loop issues
(virtual bot, camera capture, Pi hardware) are retired; the InferenceServer and
the two-node loop survive in reshaped form.

## Sequencing rationale

Three independent leaves come first, parallelizable: the
[observation source](../../docs/design/control-loop/CONTEXT.md#term-observation-source)
seam on `ControlLoop` (the small control-loop change that makes a real closed
loop possible), the InferenceServer (the cross-node `infer_action` de-risk of
ADR-0010, provable with two local nodes and no simulation), and the Python sim
server (the biggest unknown — standing up a working MuJoCo SO-101 env — kept as
one slice, demoable standalone with canned actions). The Elixir sim env adapter
then drives the sim server and exposes both loop seams, demoable against the
server alone. The closed two-node loop integrates everything: the sim node's
`ControlLoop` (with its observation source and actuator sink both bound to the
sim env adapter) calling the inference node's `InferenceServer` across the
cluster, closing the loop.

The production `ControlLoop` + `ActionQueue` are reused unchanged except for the
one designed addition — the observation-source seam (its own slice). The
emily-native forward pass and `ControlLoop`/`ActionQueue` are already built and
tracked in `smolvla-mlx-build`.

## Sequence

1. [01-observation-source-seam](issues/01-observation-source-seam.md) — the
   injectable observation source on `ControlLoop`; no blockers
2. [02-inference-server-distributed](issues/02-inference-server-distributed.md)
   — the InferenceServer answering `infer_action` from a second BEAM node; no
   blockers
3. [03-python-sim-server](issues/03-python-sim-server.md) — a Python sim server
   wrapping a LeRobot/MuJoCo SO-101 env over ZeroMQ; no blockers
4. [04-sim-env-adapter](issues/04-sim-env-adapter.md) — the Elixir adapter
   driving the sim server, exposing observe/actuate; blocked by 03
5. [05-closed-two-node-loop](issues/05-closed-two-node-loop.md) — the full
   closed loop across two nodes on one machine; blocked by 01, 02, 04

## Out of scope (whole feature)

Reinforcement learning / online learning / acting-while-learning (root "Not
reinforcement learning, yet" no-goal — the simulation is the world the policy
acts in, not a reward signal); real actuator/kinematics/safety logic on physical
hardware (standing no-goal — simulated dynamics only); real hardware of any kind
— Raspberry Pi, camera, NERVES firmware (dropped, per ADR-0011); the
ZeroMQ/Python fallback inference path (stays the permanent fallback, unused by
this demo).
