Status: ready-for-agent
Category: enhancement

## Parent

[demo design](../../../docs/design/demo/design.md), component 01.2 ([sim node](../../../docs/design/demo/CONTEXT.md#term-sim-node) wiring) and its end-to-end walkthrough. Pending build entry, [ADR-0010](../../../docs/adr/0010-beam-distribution-orthogonal-to-infer-action-port.md#adr-0010), [ADR-0011](../../../docs/adr/0011-demo-is-a-simulated-closed-loop.md#adr-0011), and [ADR-0012](../../../docs/adr/0012-sim-env-bridged-via-python-sim-server-over-zeromq.md#adr-0012).

## What to build

The whole demo loop running end to end across **two BEAM nodes on one machine**. One node plays the [sim node](../../../docs/design/demo/CONTEXT.md#term-sim-node) role: it hosts the [sim env adapter](../../../docs/design/demo/CONTEXT.md#term-sim-env-adapter) and the production `ControlLoop` + `ActionQueue`, with the loop's [observation source](../../../docs/design/control-loop/CONTEXT.md#term-observation-source) and actuator sink both wired to the sim env adapter. The other node plays the [inference node](../../../docs/design/demo/CONTEXT.md#term-inference-node) role, hosting the `InferenceServer`.

Wire them into the design's walkthrough as a **closed loop**: each tick, `ControlLoop` pops an action and hands it to the sim env adapter, which calls `env.step` — the simulated arm moves; when the [action queue](../../../docs/design/control-loop/CONTEXT.md#term-action-queue) drops below its [low-water threshold](../../../docs/design/control-loop/CONTEXT.md#term-low-water-threshold), `ControlLoop` asks the sim env adapter for the current [observation](../../../docs/design/model-runtime/CONTEXT.md#term-observation) (the freshly rendered frame) and fires an async `infer_action` across the cluster to the `InferenceServer`, aggregating the returned [action chunk](../../../docs/design/model-runtime/CONTEXT.md#term-action-chunk) back into the queue. The production loop is reused **unchanged** — this slice adds only the demo wiring (the two node roles, the cluster setup, and binding the sim env adapter to the loop's two seams), owning no model or queue logic itself.

This is the full closed perception→action demo: the arm moves in simulation, the movement changes the next frame, and SmolVLA's inference on that frame drives the next actions — all over BEAM distribution, on one machine.

## Acceptance criteria

- [ ] Two BEAM nodes on one machine form a cluster; the sim/loop node calls the inference node's `InferenceServer` across distribution as the [infer_action port](../../../docs/design/model-runtime/CONTEXT.md#term-infer-action-port).
- [ ] A sustained run shows the full closed loop: sim frame → [observation](../../../docs/design/model-runtime/CONTEXT.md#term-observation) → async cross-node `infer_action` → [action chunk](../../../docs/design/model-runtime/CONTEXT.md#term-action-chunk) aggregated into the queue → actions popped one per tick to the sim env adapter, which steps the sim, and the arm's motion is observable.
- [ ] The loop is genuinely closed: an action applied via `env.step` changes the state the next observation is drawn from — the frame SmolVLA sees reflects the arm's prior movement.
- [ ] `ControlLoop` fires `infer_action` only when the queue crosses its [low-water threshold](../../../docs/design/control-loop/CONTEXT.md#term-low-water-threshold), and the async call never blocks the tick loop (ticks keep draining what is already queued while a call is in flight).
- [ ] The demo node introduces no model or forward-pass logic and no duplicated queue logic — every inference and queue operation routes to the production context that owns it; `ControlLoop` is reused unchanged beyond wiring its observation-source and actuator-sink options.
- [ ] A lost connection to the inference node degrades exactly as the control loop already specifies (the cross-node call errors/times out, the loop keeps draining its existing queue) — no new demo-specific failure mode.

## Out of scope

Real hardware (no Pi, no camera, no NERVES — the simulation replaces it, per [ADR-0011](../../../docs/adr/0011-demo-is-a-simulated-closed-loop.md#adr-0011)); the ZeroMQ/Python fallback inference path; RL or online learning (root no-goal — the sim is the world, not a reward signal).

## Blocked by

01-observation-source-seam, 02-inference-server-distributed, 04-sim-env-adapter
