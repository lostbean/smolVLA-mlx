Status: ready-for-agent
Category: enhancement

## Parent

[model-runtime design](../../../docs/design/model-runtime/design.md), component 01.5 (`InferenceServer`). Pending build entry, [ADR-0010](../../../docs/adr/0010-beam-distribution-orthogonal-to-infer-action-port.md#adr-0010).

## What to build

A named GenServer that loads one emily-native `SmolVLA` model ([infer_action port](../../../docs/design/model-runtime/CONTEXT.md#term-infer-action-port) supplier, model-runtime component 01.2) once at start and answers `infer_action` calls against it — reachable both in-process and from a **second BEAM node** across distribution, by a plain `GenServer.call` whose target may be `{name, remote_node}`.

The whole point of this slice is to prove the distribution seam of [ADR-0010](../../../docs/adr/0010-beam-distribution-orthogonal-to-infer-action-port.md#adr-0010): that the emily-native adapter, reached across a BEAM cluster, is the same one port contract with no foreign serialization — the [observation](../../../docs/design/model-runtime/CONTEXT.md#term-observation) goes out and the [action chunk](../../../docs/design/model-runtime/CONTEXT.md#term-action-chunk) comes back as native BEAM terms, no MessagePack, no ZeroMQ. This is provable entirely with two nodes on one machine — no simulation, no camera. The [demo](../../../docs/design/demo/design.md)'s [sim node](../../../docs/design/demo/CONTEXT.md#term-sim-node) is the first remote caller.

The port contract is unchanged by where the caller sits: the same `max_state_dim` fail-loud bound the model-runtime forward pass already holds is enforced identically for a local and a remote call.

## Acceptance criteria

- [ ] Starting the server loads the emily-native model once; a bad or missing checkpoint fails loud at start, never a lazily half-loaded server.
- [ ] A caller in the same process gets a well-formed [action chunk](../../../docs/design/model-runtime/CONTEXT.md#term-action-chunk) from a real [observation](../../../docs/design/model-runtime/CONTEXT.md#term-observation).
- [ ] A caller on a **second, separate BEAM node** (two `iex` sessions on one machine, clustered by name + cookie) gets the identical result via `GenServer.call` — demonstrating the port answered across distribution with no serialization-format change at the call site.
- [ ] An observation whose state vector exceeds the checkpoint's `max_state_dim` is rejected before the forward pass, identically for a local and a remote caller.
- [ ] A remote caller that loses the cluster connection sees a standard distributed call timeout/error; the server never blocks waiting on a dead caller.

## Out of scope

Any simulation, camera, or real hardware (other slices); the `ControlLoop` that will drive this call (existing, reused unchanged); the ZeroMQ/Python fallback path (untouched — stays the permanent fallback); rebuilding the emily-native forward pass itself (already built, this slice only wraps it as a process).

## Blocked by

none
