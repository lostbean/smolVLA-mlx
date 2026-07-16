# control-loop — glossary

_Register_: BEAM/OTP vocabulary — process, supervision, tick. Reject
generic distributed-systems idiom where a plainer BEAM word exists (no
"worker pool" where "process" suffices).

### Control loop {#term-control-loop}

The Elixir side's per-tick cycle: pop a queued action, send it to the bb bot,
and top up the [action queue](#term-action-queue) when it runs low. Owned
end-to-end by the [ControlLoop](design.md) process (component 01.1).

### Action queue {#term-action-queue}

The ordered, currently-executing-plus-queued sequence of actions
[ControlLoop](design.md) (component 01.1) holds between
[infer_action](../model-runtime/CONTEXT.md#term-infer-action-port) calls. An
entity — its identity persists across ticks even as its contents mutate.

### Low-water threshold {#term-low-water-threshold}

The [action queue](#term-action-queue) depth below which `ControlLoop`
triggers the next `infer_action` call, mirroring SmolVLA's own reference
queueing policy (a new prediction request once the remaining queue drops
below a fraction of the chunk size). _Avoid_: "backpressure" (queueing-theory
idiom describing the wrong direction here — this system pushes an inference
request outward when the queue runs low, it does not push back on an
upstream producer).

### Observation source {#term-observation-source}

The injected function `ControlLoop` calls to obtain the current
[observation](../model-runtime/CONTEXT.md#term-observation) when it fires an
`infer_action` — the input seam symmetric with the actuator sink's output
seam, so the loop is agnostic to where observations come from. A real sensor
rig, a recorded dataset, or the [demo](../demo/design.md)'s
[sim env adapter](../demo/CONTEXT.md#term-sim-env-adapter) each supply one; the
default is a fixed placeholder. _Avoid_: "sensor" (names one possible provider,
not the injectable seam itself).
