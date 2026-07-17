Status: ready-for-human
Category: enhancement

## Parent

[control-loop design](../../../docs/design/control-loop/design.md), component 01.1 (`ControlLoop`), the [observation source](../../../docs/design/control-loop/CONTEXT.md#term-observation-source) seam. Pending build entry, [ADR-0011](../../../docs/adr/0011-demo-is-a-simulated-closed-loop.md#adr-0011).

## What to build

Add an injectable [observation source](../../../docs/design/control-loop/CONTEXT.md#term-observation-source) to `ControlLoop` — a zero-arity function the loop calls to obtain the current observation whenever it fires an `infer_action`, symmetric with the existing actuator sink (the output seam). Today `ControlLoop` sources its observation from a private, hardcoded placeholder; this makes that source a `start_link` option so a caller can supply where observations come from, while the loop stays agnostic to the provider.

The default must remain today's fixed placeholder observation, so every existing caller and test behaves identically when no source is supplied. `ControlLoop`'s own queue, timing, and adapter-dispatch logic must not change — this adds one injected input, nothing else. This is the seam the [demo](../../../docs/design/demo/design.md)'s sim env adapter will plug into as its first customer, making a real closed loop possible.

## Acceptance criteria

- [ ] `ControlLoop.start_link` accepts an `observation_source` option: a zero-arity function returning an observation (the `%{image, image_shape, state, instruction}` shape the adapters already consume).
- [ ] When `observation_source` is supplied, the observation passed to `infer_action` comes from calling that function — verified by injecting a source that returns a distinctive observation and asserting the adapter receives it.
- [ ] When `observation_source` is omitted, behavior is byte-for-byte identical to today: the same fixed placeholder observation is used, and all existing `ControlLoop` tests pass unchanged.
- [ ] The observation source is called once per triggered `infer_action` (when the queue crosses its low-water threshold), not on every tick, and never blocks the tick loop.
- [ ] `ControlLoop`'s queue/timing/state-machine logic is unchanged — only the observation input is now injectable.

## Out of scope

The sim env adapter that will supply a real source (later slice); the actuator sink (already exists); any change to the adapter-dispatch, queue, or tick-timing logic; the observation's own validation beyond what already exists.

## Blocked by

none
