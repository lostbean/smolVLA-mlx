Status: ready-for-agent
Category: enhancement

## Parent

[model-runtime design](../../../docs/design/model-runtime/design.md), component 01.2 (`SmolVLAModel (Elixir-native)`). Pending build entry, [ADR-0003](../../../docs/adr/0003-emily-native-primary-zeromq-fallback.md#adr-0003).

## What to build

Port SmolVLA's real forward pass — the actual SmolVLM2-shaped backbone and
flow-matching action expert, at full parameter scale, loading the actual
trained checkpoint's weights — to `Nx.Defn` running through `emily`'s
`Nx.Backend`, replacing the scaled-down mechanism prototype with the real
model.

A prior `/prototype` run already de-risked the mechanism itself (cross-attention
into a frozen intermediate layer, the iterative flow-matching structure, and
`emily`'s op coverage) against a small stand-in architecture with random
weights — see the **"De-risked by prototype"** note on component 01.2. This
slice is the full-scale port that was explicitly left open by that
prototype: real weights, real backbone size, and the real conformance check
against the Python implementation (slice 02) rather than a NumPy oracle.

Weights are the only artifact crossing from the Python side — no code is
shared between the two implementations (the **"Weights are the only
cross-runtime artifact"** invariant).

## Acceptance criteria

- [ ] The Elixir-native `infer_action` produces action chunks numerically
      equivalent (within an agreed tolerance) to the Python implementation
      (slice 02), given the same loaded checkpoint and the same observation
      — a real conformance check, not the prototype's synthetic one.
- [ ] `ControlLoop` (slice 04) runs unmodified against this adapter — only
      its configured adapter changes (`:emily_native` instead of
      `:zeromq_fallback`); no code in `control-loop` needs to know which
      adapter is active.
- [ ] Latency at full scale is measured and reported (the prototype's 5ms
      p50 was on a scaled-down stand-in; this slice reports the real
      number) against the 100ms budget named in the design.
- [ ] A shape or dimensionality mismatch raises before dispatching to
      `emily`, never silently reshapes or truncates (matching the Python
      adapter's own failure behavior).

## Out of scope

Fine-tuning (Phase D, slices 06-08); any change to `ControlLoop`'s own
queue/timing logic; the ZeroMQ path (stays as the permanent fallback,
untouched by this slice).

## Blocked by

01-smolvla-model-class, 04-control-loop-zeromq
