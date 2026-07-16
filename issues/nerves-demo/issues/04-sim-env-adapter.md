Status: ready-for-agent
Category: enhancement

## Parent

[demo design](../../../docs/design/demo/design.md), component 01.1 ([sim env adapter](../../../docs/design/demo/CONTEXT.md#term-sim-env-adapter)). Pending build entry, [ADR-0011](../../../docs/adr/0011-demo-is-a-simulated-closed-loop.md#adr-0011) and [ADR-0012](../../../docs/adr/0012-sim-env-bridged-via-python-sim-server-over-zeromq.md#adr-0012).

## What to build

The Elixir unit that owns the whole simulation seam: it drives one gym env through the Python [sim server](../../../docs/design/demo/CONTEXT.md#term-sim-server) (over ZeroMQ + MessagePack) and exposes both interfaces the production `ControlLoop` needs. As the loop's [observation source](../../../docs/design/control-loop/CONTEXT.md#term-observation-source) it returns the current [observation](../../../docs/design/model-runtime/CONTEXT.md#term-observation) — the env's rendered frame, the arm's state, and the fixed demo instruction; as the loop's actuator sink it applies each popped [action](../../../docs/design/model-runtime/CONTEXT.md#term-action-chunk) by calling `step`.

It is **one unit, not two**, because in a gym env producing the next observation and consuming the action are two halves of a single `step` cycle — they cannot be separated ([ADR-0011](../../../docs/adr/0011-demo-is-a-simulated-closed-loop.md#adr-0011)). This slice is demoable against the sim server alone: drive `observe`/`actuate` directly (no control loop) and watch the simulated arm move and the returned observation change.

## Acceptance criteria

- [ ] `observe/1` returns a well-formed observation (`%{image, image_shape, state, instruction}`) assembled from the sim server's current frame and state, with the fixed demo instruction and the state within the checkpoint's `max_state_dim`.
- [ ] `actuate/2` applies one action by issuing a `step` to the sim server, advancing the simulation; the next `observe/1` reflects the movement.
- [ ] The adapter exposes exactly the shapes `ControlLoop` expects at both seams: a zero-arity observation source (closing over the adapter) and an actuator sink `(action -> :ok)` — so `ControlLoop` drives it with no demo-specific change beyond wiring these two options.
- [ ] Demoable standalone: a canned sequence of `actuate` calls interleaved with `observe` drives the sim and the arm's motion is observable, with no `ControlLoop` present.
- [ ] A sim-server failure (dead process, lost socket, an env that raised) surfaces loud and local from `observe/1` or `actuate/2` — an explicit error or raise, never a fabricated blank frame or a silently dropped action.

## Out of scope

The full closed loop wiring `ControlLoop` + `InferenceServer` across two nodes (next slice); the Python sim server itself (previous slice); the observation_source seam on `ControlLoop` (its own slice); real hardware.

## Blocked by

03-python-sim-server
