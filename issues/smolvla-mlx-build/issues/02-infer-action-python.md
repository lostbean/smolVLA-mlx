Status: ready-for-agent
Category: enhancement

## Parent

[model-runtime design](../../../docs/design/model-runtime/design.md), component 01.1 (`SmolVLAModel (Python)`). Pending build entry, [ADR-0001](../../../docs/adr/0001-parallel-action-entry-point.md#adr-0001).

## What to build

Implement the `{#term-infer-action-port}` contract on the loaded SmolVLA
model from the previous slice: one call taking an
`{#term-observation}` (camera image(s), robot proprioceptive state, a
language instruction) and returning one `{#term-action-chunk}` — a
continuous-valued action sequence, never a text token sequence (per the
**"The action expert's output is never tokenized"** invariant).

This runs SmolVLA's real forward pass: vision/language encoding through the
frozen SmolVLM2 backbone (reusing mlx-vlm's existing vision/language
plumbing, never reimplementing it — the **"Reuse mlx-vlm's plumbing, never
its assumptions"** principle), then the flow-matching action expert
producing the action chunk. This is implemented alongside mlx-vlm's standard
contract but never through `generate()` — SmolVLA has no token-sampling
output path.

## Acceptance criteria

- [ ] Calling `infer_action` with a real image, a real or synthetic robot
      state vector, and a language instruction returns one action chunk of
      the shape declared by the loaded checkpoint's config (chunk size ×
      action dimensionality, from the previous slice).
- [ ] `generate()` is not implemented for this model type — calling it
      raises or is simply absent, never silently falls back to a token
      interface.
- [ ] An `infer_action` call with a wrong action-space dimensionality
      (mismatched against the loaded config) raises before running the
      forward pass, never silently reshapes or truncates.
- [ ] Robot state is compressed to exactly one token before entering the
      action expert, matching SmolVLA's own architecture.

## Out of scope

Fine-tuning; the Elixir-native adapter; any network-facing server; any
robot-side control logic (kinematics, safety, interpolation).

## Blocked by

01-smolvla-model-class
