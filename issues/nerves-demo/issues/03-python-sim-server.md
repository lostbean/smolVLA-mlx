Status: ready-for-human
Category: enhancement

## Parent

[demo design](../../../docs/design/demo/design.md), the [sim server](../../../docs/design/demo/CONTEXT.md#term-sim-server) (component 01.1 drives it). Pending build entry, [ADR-0012](../../../docs/adr/0012-sim-env-bridged-via-python-sim-server-over-zeromq.md#adr-0012).

## What to build

A long-lived Python process that wraps one LeRobot/MuJoCo gym environment of the SO-101 arm doing pick-and-place — the robot the `lerobot/svla_so101_pickplace` checkpoint was trained on — and answers `reset` / `step(action)` / `render` requests over a ZeroMQ REQ/REP socket with MessagePack framing. It is the single seam between the Elixir [sim node](../../../docs/design/demo/CONTEXT.md#term-sim-node) and the Python physics engine, mirroring the existing model-runtime ZeroMQ server's transport and wire conventions ([ADR-0012](../../../docs/adr/0012-sim-env-bridged-via-python-sim-server-over-zeromq.md#adr-0012), reusing the MessagePack-over-ZeroMQ framing of ADR-0007/ADR-0008).

`step(action)` advances the simulation with the given action and returns the resulting rendered frame plus the arm's proprioceptive state; `render` returns the current frame; `reset` starts a fresh episode. The observation payload it returns must be shaped so the Elixir side can assemble the standard observation (image + state) the [infer_action port](../../../docs/design/model-runtime/CONTEXT.md#term-infer-action-port) accepts, with the arm's state within the checkpoint's `max_state_dim`. This slice is Python-only and demoable standalone: a canned sequence of actions drives the env and the simulated arm visibly moves, no Elixir involved yet.

## Acceptance criteria

- [ ] A running server, driven by a canned sequence of `step` requests, advances a real MuJoCo SO-101 pick-and-place simulation and the arm's motion is observable (rendered frames or saved video).
- [ ] `reset`, `step(action)`, and `render` each answer over a ZeroMQ REQ/REP socket with MessagePack framing, consistent with the existing model-runtime server's transport.
- [ ] `step` returns both the resulting rendered frame and the arm's proprioceptive state; the state's dimensionality is within the checkpoint's `max_state_dim`.
- [ ] A malformed request (unknown op, wrong action shape) is rejected with an explicit error over the wire, never a silent no-op or a fabricated frame.
- [ ] The env dependency is declared in the Python project so the server runs under the repo's environment; a missing simulator dependency fails loud at startup, not mid-episode.

## Out of scope

The Elixir sim env adapter that will drive this server (later slice); wiring into `ControlLoop` or the cluster (later slice); real hardware; any fine-tuning or reward/RL signal (the sim is the world the policy acts in, not a training loop).

## Blocked by

none
