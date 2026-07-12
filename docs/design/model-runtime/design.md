---
eyebrow: Context · model-runtime · [root](../design.md)
lede: SmolVLA's forward pass, its weights, and the two adapters that expose it — a Python fork of mlx-vlm and an Elixir-native Nx.Defn port — behind one infer_action port.
footer: This document owns the model-runtime components; CONTEXT owns the terms; ADRs own the rationale; the root indexes both contexts.
---

# model-runtime context

This context owns everything about SmolVLA as a model: its architecture, its
weights, how it's fine-tuned, and the two independent adapters (Python,
Elixir-native) that expose the same `infer_action` port. The
[control-loop](../control-loop/design.md) context is a sibling and this
context's only customer; references to it are explicit pointers.

## 00 Foundation

:::goal
**A parallel action entry point, not a forced generate()**

`infer_action(observation) -> action_chunk` is SmolVLA's entry point,
implemented alongside mlx-vlm's standard `Config`/weight-loading contract but
never through `generate()`. See
[ADR-0001](../../adr/0001-parallel-action-entry-point.md#adr-0001).
:::

:::goal
**Fine-tune the action expert locally**

Fine-tune SmolVLA's action expert (VLM backbone frozen, matching the paper's
reference training path) against
[episodes](CONTEXT.md#term-episode) on this Mac.
:::

:::no-goal
**Not a second training framework**

Reuse LeRobot's dataset format and training conventions where possible; this
context does not invent its own data format or training loop from scratch.
:::

:::invariant {enforcement=convention}
**The action expert's output is never tokenized**

No path in this context encodes a continuous action value as a text token,
quantized or otherwise. See
[ADR-0001](../../adr/0001-parallel-action-entry-point.md#adr-0001).
:::

:::invariant {enforcement=partial script=test_models.py}
**A new model registers by model_type, mlx-vlm's own convention**

SmolVLA's directory name under `mlx_vlm/models/` matches its `model_type` in
`config.json` exactly, per mlx-vlm's existing dynamic-import convention —
partial because `test_models.py` checks load/shape correctness, not the
convention-following itself.
:::

:::principle {id=P1 lens=composition}
**Reuse mlx-vlm's plumbing, never its assumptions**

Vision/language encoding, weight loading, and registration are reused wholly
from mlx-vlm's existing contract. What is never reused is the assumption that
every model's output is a token sequence.
:::

## Pending updates

:::pending {kind=build since=2026-07-12}
The mlx-vlm fork itself (config class, model class, `infer_action()`,
LoRA/full fine-tuning of the action expert) is designed, not built. See
[ADR-0001](../../adr/0001-parallel-action-entry-point.md#adr-0001).
:::

:::pending {kind=build since=2026-07-12}
The Elixir-native `Nx.Defn` adapter (SmolVLA's forward pass reimplemented
against `emily`'s `Nx.Backend`) is designed, not built. A `/prototype` run on
2026-07-12 de-risked the core mechanism (see 01.2); the full-scale port
against real SmolVLA weights remains to be built. See
[ADR-0003](../../adr/0003-emily-native-primary-zeromq-fallback.md#adr-0003).
:::

## 01 Components

:::cards {cols=2}

### SmolVLAModel (Python) `lens:depth`

**Own weight loading and the Python-side forward pass.** A `Config` dataclass
plus a `Model` class following mlx-vlm's registration contract — vision/state
encoding reused from the standard mlx-vlm plumbing, `infer_action()` as the
only output entry point. See 01.1.

### SmolVLAModel (Elixir-native) `lens:composition`

**Own the same forward pass, reimplemented against `emily`.** `Nx.Defn`
functions calling `emily`'s `Nx.Backend`, loading the same safetensors
weights, exposed to `control-loop` as the in-process adapter. Designed; the
core mechanism is prototype-verified, the full-scale port is not yet built.
See 01.2.

### FineTuneJob `lens:state`

**Own one fine-tuning run.** Takes a set of [episodes](CONTEXT.md#term-episode)
and the frozen VLM backbone, produces updated action-expert weights,
checkpointed for resumability. See 01.3.
:::

### 01.1 SmolVLAModel (Python) — responsibility, interface, invariants

**Responsible for:** loading a SmolVLA checkpoint (safetensors +
`config.json`), encoding one [observation](CONTEXT.md#term-observation)
(image(s), robot state, instruction) through the frozen SmolVLM2 backbone, and
running the flow-matching action expert to produce one
[action chunk](CONTEXT.md#term-action-chunk).

**Interface:**
```
SmolVLAConfig  # model_type="smolvla", chunk_size, action_dim, vision/action-expert layer counts
SmolVLAModel.from_pretrained(checkpoint_path) -> SmolVLAModel
SmolVLAModel.infer_action(image, state, instruction) -> ActionChunk
```

**Interacts with:** mlx-vlm's existing weight-loading and vision/language
encoding plumbing (reused, not reimplemented); `control-loop`'s ZeroMQ client,
which is the only caller of `infer_action()` at runtime, through the fallback
adapter.

**Invariants held:** never implements `generate()`; state is compressed to
exactly one token per SmolVLA's own architecture, matching the reference
implementation's shape.

**Fails:** a malformed or missing checkpoint raises at `from_pretrained()` —
loud and local, never a silent zero-initialized fallback; an
`infer_action()` call with a wrong action-space dimensionality (mismatched
against the loaded config) raises before running the forward pass.

### 01.2 SmolVLAModel (Elixir-native) — responsibility, interface, invariants

**Responsible for:** the identical `Observation -> ActionChunk` transformation
as 01.1, expressed as `Nx.Defn` numerical functions executing in-process
through `emily`'s `Nx.Backend` — no Python process in this path.

**Interface:**
```elixir
SmolVLA.load(checkpoint_path) :: SmolVLA.t()
SmolVLA.infer_action(model, image, state, instruction) :: ActionChunk.t()
```

**Interacts with:** `emily`'s `Nx.Backend` for every tensor op (no other
tensor runtime); loads the same safetensors weights 01.1 produces or consumes
— the only artifact shared between the two adapters
([ADR-0004](../../adr/0004-weights-only-cross-runtime-sharing.md#adr-0004)).

**Invariants held:** behaviorally equivalent to 01.1 on the same weights and
inputs — enforced by convention today (a conformance check comparing outputs
on fixed inputs is the intended mechanism once built, not yet a script).

**Fails:** same loud/local failure shape as 01.1 — a shape or dimensionality
mismatch raises before dispatching to `emily`, never silently reshapes or
truncates.

**De-risked by prototype (2026-07-12):** a scaled-down but architecturally
faithful stand-in — a multi-layer self-attention backbone plus a flow-matching
action expert doing self-attention, cross-attention into a frozen intermediate
backbone layer, and multi-step Euler integration — was implemented twice from
identical fixed random weights: once in NumPy (the oracle), once in `Nx.Defn`
against `emily`'s `Nx.Backend`. Result: numerical parity to 2.96×10⁻⁹ max
absolute difference (float32 rounding noise) against a 1×10⁻³ tolerance bar,
and 5.26ms p50 latency (N=50) against a 100ms budget — roughly 19× headroom
even against the strict 33ms/30Hz-every-tick bar. This confirms the mechanism
— cross-attention into a frozen intermediate layer, the iterative
flow-matching structure, and `emily`'s op coverage — is expressible and fast
enough. **Not yet proven:** the full-scale port against SmolVLA's real
~450M-parameter backbone and ~100M-parameter action expert, and against real
trained weights rather than random ones — that remains the open work in the
[pending ledger](../design.md).

### 01.3 FineTuneJob — responsibility, interface, invariants

**Responsible for:** taking a set of [episodes](CONTEXT.md#term-episode) and
producing updated action-expert weights, VLM backbone frozen (matching the
paper's own reference training path); checkpointing so a run is resumable.

**Interface:**
```
FineTuneJob.run(checkpoint_path, episodes, output_path) -> FineTuneJob  # identity persists across the run
FineTuneJob.resume(checkpoint_path) -> FineTuneJob
```

**Interacts with:** LeRobotDataset-format episode data as input; produces
safetensors weights consumed by both 01.1 and (once built) 01.2.

**Invariants held:** the VLM backbone stays frozen for the default training
path (matching the paper); training only the action expert is the default,
not full fine-tuning — a config flag switches this, it is never silently
inconsistent between a run and its checkpoint.

**Fails:** a job interrupted mid-run resumes from its last checkpoint, never
silently restarts from scratch nor silently continues from a corrupt
checkpoint (checksum or shape-validated on resume).
