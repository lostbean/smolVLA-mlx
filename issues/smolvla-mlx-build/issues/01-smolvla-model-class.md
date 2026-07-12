Status: ready-for-agent
Category: enhancement

## Parent

[model-runtime design](../../../docs/design/model-runtime/design.md), component 01.1 (`SmolVLAModel (Python)`). Pending build entry, [ADR-0001](../../../docs/adr/0001-parallel-action-entry-point.md#adr-0001).

## What to build

Add SmolVLA as a registered model in the mlx-vlm fork: a config type and a
model type that load a real SmolVLA checkpoint (safetensors weights plus its
`config.json`) and expose the loaded model's architecture parameters (chunk
size, action dimensionality, vision/action-expert layer counts).

Registration follows mlx-vlm's own existing convention: the model registers
by `model_type` exactly as its checkpoint's `config.json` declares (the
`{#term-infer-action-port}` contract this model will expose in the next
slice depends on this loading correctly) — see the **"A new model registers
by model_type"** invariant in the model-runtime design's foundation.

This slice does NOT yet implement `infer_action()` itself (that's the next
slice) — it stops at "the checkpoint loads and its config is introspectable."

## Acceptance criteria

- [ ] A real, publicly available SmolVLA checkpoint loads successfully from
      its safetensors weights and `config.json`.
- [ ] The loaded model's config is introspectable: chunk size, action
      dimensionality, and vision/action-expert layer counts are all readable
      from the loaded object.
- [ ] A malformed or missing checkpoint raises loud and local at load time —
      never a silent zero-initialized fallback (per the "Fails" behavior
      named in component 01.1).
- [ ] The model registers under mlx-vlm's own dynamic `model_type` import
      convention (existing tests for other models in the same convention
      continue to pass unmodified).

## Out of scope

`infer_action()` itself (next slice); fine-tuning; the Elixir-native
adapter; any network-facing server.

## Blocked by

none — first slice.
