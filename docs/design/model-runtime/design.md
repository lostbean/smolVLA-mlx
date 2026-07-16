---
eyebrow: Context · model-runtime · [root](../design.md)
lede: SmolVLA's forward pass, its weights, and the adapter pairs that expose it — Python and Elixir-native, behind the infer_action port for inference and the FineTuneJob port for training.
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
**Fine-tune locally, from real or simulated episodes alike**

Fine-tune SmolVLA's action expert (VLM backbone frozen, matching the paper's
reference training path) against [episodes](CONTEXT.md#term-episode) on this
Mac, regardless of whether an episode came from real robot usage or a
simulation environment — the fine-tuning contract never changes based on
source.
:::

:::goal
**Elixir-native fine-tuning, intended, conditional cutover**

Fine-tuning follows the same ports-and-adapters shape as inference: one
`FineTuneJob` contract, an Elixir-native (`Nx.Defn` autodiff plus `Polaris`
for optimization) adapter as the intended target, and the Python
(LeRobot-based) adapter as the reference and the permanent fallback if a
task-performance-parity check never clears. See
[ADR-0005](../../adr/0005-elixir-native-finetuning-conditional-retirement.md#adr-0005).
:::

:::no-goal
**Not a second training framework — for the Python adapter**

The Python `FineTuneJob` adapter reuses LeRobot's dataset format and training
conventions rather than inventing its own; the Elixir-native adapter
necessarily reimplements the training loop (no shared code crosses the
boundary, same as inference), but not the data format or recipe design.
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
convention-following itself. A real SmolVLA checkpoint's `config.json` is
[LeRobot-native](CONTEXT.md#term-lerobot-native-checkpoint), not mlx-vlm's
usual HF-nested shape — `SmolVLAModel.from_pretrained()` reads it directly
rather than routing through mlx-vlm's shared dynamic-dispatch machinery. See
[ADR-0006](../../adr/0006-config-adapter-for-lerobot-checkpoint-shape.md#adr-0006).
:::

:::principle {id=P1 lens=composition}
**Reuse mlx-vlm's plumbing, never its assumptions**

Vision/language encoding, weight loading, and registration are reused wholly
from mlx-vlm's existing contract. What is never reused is the assumption that
every model's output is a token sequence.
:::

## Pending updates

:::pending {kind=build since=2026-07-13}
The mlx-vlm fork (config class, model class, `infer_action()`, and Python
fine-tuning via LeRobot's own `lerobot-train`, component 01.3) is built. A
real fine-tuning run against real episodes reloads through `infer_action()`
producing a finite, correctly-shaped action chunk. LoRA (as opposed to the
current full-parameter action-expert training) remains unbuilt — a smaller,
separate follow-up, not blocking anything else in this ledger. See
[ADR-0001](../../adr/0001-parallel-action-entry-point.md#adr-0001).
:::

:::pending {kind=build since=2026-07-13}
Both Elixir-native adapters are built: the forward pass (component 01.2,
inference) conformance-checked at 0.65% mean relative error against the
Python implementation — its warm latency (~186ms, freshly re-measured
2026-07-16) now beats the Python reference's own (~321ms same-session) and
has ~27× headroom against the actual async-tick-triggered deadline (~5s at
this system's own 5Hz-class target and a 25-action low-water threshold —
see 01.2's own latency note for the derivation and the full four-step
optimization history that closed the earlier per-layer-dispatch gap) — and
fine-tuning (component 01.4, `Nx.Defn.value_and_grad` plus `Polaris`) — a
real training run reloads through both inference adapters with structurally
identical safetensors output to the Python trainer's. **Judged
(2026-07-13):** the task-performance-parity gate between the two trainers
(component 01.4's own cutover gate) has run — both trainers fine-tuned 20
real steps against an identical 44-episode training subset of
`lerobot/svla_so101_pickplace` and the identical `lerobot/smolvla_base`
starting checkpoint, evaluated identically against a 6-episode held-out set
(30 frames) via the [action-accuracy
proxy](CONTEXT.md#term-action-accuracy-proxy) (never a live task-success
rate, per ADR-0009): Elixir's accuracy proxy (38.80 mean absolute error) was
actually lower (better) than Python's (46.35) on this run — a −16.3%
"regression" against a documented ≤20% threshold, i.e. comfortably within
it in the favorable direction — and Elixir's own-adapter throughput (0.82
actions/sec) was 27.1% of Python's against a documented ≥10% floor — both
sub-gates pass. (Small sample — 30 held-out frames, 20 training steps each
— so this should be read as "not meaningfully worse," per ADR-0005's own
bar, rather than as proof Elixir is definitively more accurate.)
**01.4 is promoted to the production trainer**; 01.3 remains the reference
implementation and permanent fallback per ADR-0005. See
[ADR-0003](../../adr/0003-emily-native-primary-zeromq-fallback.md#adr-0003),
[ADR-0005](../../adr/0005-elixir-native-finetuning-conditional-retirement.md#adr-0005),
and
[ADR-0009](../../adr/0009-offline-action-accuracy-as-task-success-proxy.md#adr-0009).
The full run configuration and numbers are recorded in
`finetune_job/parity_gate/parity_gate_report.json`.
:::

:::pending {kind=build since=2026-07-16}
The `InferenceServer` (component 01.5) — the named GenServer wrapping the
emily-native adapter (01.2) so it answers `infer_action` in-process or across a
BEAM cluster — is designed, not built. First remote caller is the
[demo](../demo/design.md) context's [sim node](../demo/CONTEXT.md#term-sim-node).
See
[ADR-0010](../../adr/0010-beam-distribution-orthogonal-to-infer-action-port.md#adr-0010).
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

### FineTuneJob (Python) `lens:state`

**Own one fine-tuning run, LeRobot-based.** Takes a set of
[episodes](CONTEXT.md#term-episode) and the frozen VLM backbone, produces
updated action-expert weights, checkpointed for resumability. The reference
implementation and the permanent fallback. See 01.3.

### FineTuneJob (Elixir-native) `lens:state`

**Own the same fine-tuning contract, via `Nx.Defn` autodiff and `Polaris`.**
Identical episodes-in/weights-out contract as the Python adapter, training
loop reimplemented against `Nx.Defn.value_and_grad` directly on 01.2's
hand-written forward pass, with `Polaris` for the optimizer. The intended
target, conditional on task-performance parity. See 01.4.

### InferenceServer (Elixir-native) `lens:composition`

**Own the emily-native adapter as a long-lived, cluster-addressable
process.** A named GenServer holding one loaded 01.2 `SmolVLA` model, answering
[infer_action](CONTEXT.md#term-infer-action-port) calls whether the caller is
in-process or on a remote BEAM node. Adds no forward-pass logic — it is the
process wrapper that makes 01.2 reachable across a cluster. See 01.5.
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

**Interacts with:** mlx-vlm's existing vision/language encoding plumbing
(reused, not reimplemented) and its safetensors-reading primitives; weight
*dispatch* is not reused, since a real checkpoint's
[LeRobot-native config](CONTEXT.md#term-lerobot-native-checkpoint) does not
carry mlx-vlm's `model_type`/nested-config shape — `from_pretrained()` reads
it directly instead of routing through mlx-vlm's generic
`get_model_and_args()` dispatch, keeping that shared path unmodified for
every other model (see [ADR-0006](../../adr/0006-config-adapter-for-lerobot-checkpoint-shape.md#adr-0006)).
`control-loop`'s ZeroMQ client is the only caller of `infer_action()` at
runtime, through the fallback adapter.

**Invariants held:** never implements `generate()`; state is compressed to
exactly one token per SmolVLA's own architecture, matching the reference
implementation's shape.

**Fails:** a malformed or missing checkpoint raises at `from_pretrained()` —
loud and local, never a silent zero-initialized fallback; an
`infer_action()` call whose state vector *exceeds* the loaded config's
`max_state_dim` raises before running the forward pass — a shorter state is
valid and zero-padded to the checkpoint's full width, never a silent
truncation of an oversized one.

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
enough.

**Proven at full scale (2026-07-13):** the real port against SmolVLA's actual
~450M-parameter backbone and ~100M-parameter action expert, real trained
`lerobot/smolvla_base` weights, and a real conformance check against the
Python implementation (component 01.1) rather than a NumPy oracle —
0.65% mean relative error / 0.008 max absolute difference on a real
observation, well inside the 2% budget the errors' own bf16-drift character
justifies (see this component's test suite for the full reasoning). This
confirms the mechanism holds at real scale, not just the prototype's stand-in.
**Latency:** measured warm latency is ~186ms median (2026-07-16, 9 samples,
184.3–189.3ms) after the latency-gap work landed — now *below* the Python
adapter's own ~327ms (see the cross-runtime table below), down from the
~634ms recorded after the first dispatch-tax fix and the ~1.2s before it.
The real deadline this must clear is not a flat 100ms per call —
`ControlLoop` (01.1) fires `infer_action` asynchronously via a `Task`,
never blocking the tick loop, so the actual constraint is *time to complete
before the
[queue](../control-loop/CONTEXT.md#term-action-queue) drains from the
[low-water threshold](../control-loop/CONTEXT.md#term-low-water-threshold)
to empty* — at this system's own 5Hz-class target tick rate and a
25-action threshold (half the 50-action chunk size), that budget is ~5s,
not 100ms. Measured against that real bar, ~186ms has ~27× headroom.

The gap to Python closed across four measured optimizations, each in its
own commit and each parity-neutral (end-to-end MRE held at 0.646% / 0.0081
max abs diff throughout — byte-identical to the pre-optimization figure):

1. **Resize sample geometry as tensor arithmetic.** `prepare_images`'s
   bilinear interpolation built four 262,144-element Elixir lists per call
   (the outer-product gather grid) and paid a host→tensor conversion for
   each; replaced with per-axis floor/ceil index + weight tensors computed
   once per `{in, out}` size pair (memoized, built in f64 on
   `Nx.BinaryBackend` so the floor/clamp/weight arithmetic stays
   bit-identical to the old float64 path) plus a separable
   `Nx.take`-on-axis-0-then-1 gather. The `[0,255]` range heuristic's host
   round-trip (`Nx.to_number(Nx.reduce_max(...))`) became an on-device
   `Nx.select`. ~137ms.
2. **Skip the SigLIP tower for zero-masked fake cameras.** The checkpoint
   declares 3 camera slots; slots 2–3 get zero images with `pad_mask=false`
   yet each ran the full 12-layer tower to produce tokens the pad mask
   excludes from every attention row. Emit a zeros embedding of the exact
   shape/dtype instead. ~106ms.
3. **Prefill/step KV cache across Euler steps.** The mask guarantees the
   backbone branch never attends the suffix, so its 16-layer trajectory —
   and the per-layer key/value tensors the suffix attends — are identical
   across all 10 Euler steps. `SmolVLA.Expert` splits into a `prefill` pass
   (run once: backbone trajectory + per-layer cached k/v) and a `step` pass
   (run 10×: suffix-only queries attending the cached keys), mirroring
   lerobot's `fill_kv_cache`. `Expert.forward/6` is unchanged for the
   training path. ~168ms.
4. **Compile the vision tower.** `SmolVLA.Vision` ran its 12 encoder layers
   eagerly (op-by-op `def`/`defp` dispatch); converted to the same
   `deftransform` shim → `defn` entry → `deftransformp` stack pattern
   `Expert.forward` uses, verified to lower fully native. ~23ms.

An earlier `Emily.Compiler` `fuse: true` experiment gave ~8% against the
original large per-step graph but was reverted: once the KV-cache split
(3) and the vision compile (4) shrank the per-step work, it became
neutral-to-slightly-slower and it perturbs f32 parity, so the shipped
config omits it. See the [pending ledger](../design.md).

**Cross-runtime comparison (updated 2026-07-16, post-latency-gap-work):**
the Elixir-native adapter (01.2) now does the identical `infer_action` work
*faster* than the Python reference (01.1), after the four optimizations
above closed and then overtook the earlier gap. Both figures below are
freshly re-measured this session on the same machine, same checkpoint,
same fixture observation, same methodology (median of 9 warm runs after one
warm-up); the Python side via `bench/warm_latency_python.py`
(`SmolVLAModel.infer_action`, which calls `mx.eval` before returning, so
end-to-end wall-clock is honest), the Elixir side via
`bench/warm_latency.exs`:

| | Python (01.1, direct MLX) | Elixir-native (01.2, emily/`Nx.Defn`) | Gap |
| --- | --- | --- | --- |
| Warm latency, one `infer_action` call | ~321ms median (9 samples, 319.1–345.3ms) | ~186ms median (9 samples, 184.3–189.3ms) | Elixir ~1.7× faster |
| Inference rate (calls = images analyzed = chunks produced / sec) | ~3.1 /sec | ~5.4 /sec | Elixir ~1.7× Python's rate |
| Against the real ~5s deadline (derived above, not the stale 100ms figure) | ~16× headroom | ~27× headroom | both clear it; Elixir by more |

Both adapters clear the real deadline derived above; the Elixir side is now
the faster of the two, having cut its warm latency from ~1.2s (pre-fix) to
~634ms (dispatch-tax fix) to ~186ms (the latency-gap work above), parity
unchanged at 0.646% mean relative error / 0.0081 max abs diff throughout.

**What the inference rate means for a robot (it is not the tick rate).**
One `infer_action` call consumes ONE image and produces ONE chunk of
`chunk_size` (50) actions via the 10-step denoising loop — so ~5.4 /sec is
*images analyzed / action-chunks produced* per second, NOT individual robot
moves per second (that would be ~5.4 × 50, a meaningless figure since the
bot never consumes actions that fast). Two rates are deliberately
decoupled: `ControlLoop` (01.1) pops one action per **tick** (the ~5Hz-class
actuation rate) and only fires a fresh `infer_action` when the queue drains
below its [low-water
threshold](../control-loop/CONTEXT.md#term-low-water-threshold) (~25
actions, half a chunk). So today the *environmental-re-evaluation cadence* —
how often a new camera frame is analyzed and the plan refreshed — is bounded
by that queue policy (~every ~5s at a 25-action threshold and 5Hz ticks),
NOT by inference speed. The ~186ms latency means inference is no longer the
constraint on that cadence: re-evaluating every camera frame more eagerly
(a higher low-water threshold, or re-inferring every N ticks) is affordable
up to a ceiling of ~5.4 fresh evaluations/sec (~5.4Hz) before inference
itself becomes the bottleneck — a control-loop policy choice
([control-loop](../control-loop/design.md)), not a model-runtime limit.
The freshly-measured Python median (~321ms) confirms the ~327–331ms figure
this table previously carried was real, not stale.

:::info {title="Dispatch-tax fix: unblocked and landed (2026-07-15)"}
The first step of closing the gap was fusing the per-op
`Emily.Fast`/`Nx.Defn` dispatch into one traced `defn` graph so
`Emily.Compiler`'s single-NIF whole-graph replay applies instead of eager
per-op dispatch from Elixir orchestration. This was **resolved**, not just
attempted. The 16-layer Expert stack is traced as one `defn` graph per
Euler step (`SmolVLA.Expert.forward/6` converted from `def` to a
`deftransform` shim over a `defn` entry point), and the `emily` dependency
is pinned to the fork
[lostbean/emily](https://github.com/lostbean/emily) branch
`fix/205-fast-defn-composition` (commit `5ade402`) carrying the fix for
[ausimian/emily#205](https://github.com/ausimian/emily/issues/205).

The original block was a real `emily` limitation: every `Emily.Fast`
kernel's public function was plain `def`, and `defn`'s call-dispatch
requires the *callee itself* to be `defn`-defined — so the kernels could not
be composed inside a whole-graph `defn`. The fork branch resolves this,
letting the whole Expert stack trace as a single graph that
`Emily.Compiler` replays as one NIF call rather than per-op eager dispatch.
Measured effect at the time: warm `infer_action/4` dropped from ~1.2s to
~634ms median. The subsequent latency-gap work (the four optimizations
above) then took it to ~186ms — below the Python adapter — with numerical
parity unchanged at 0.646% mean relative error / 0.0081 max abs diff
against the Python reference, inside the 2% budget.
:::

### 01.3 FineTuneJob (Python) — responsibility, interface, invariants

**Responsible for:** taking a set of [episodes](CONTEXT.md#term-episode) —
real-robot or simulation-sourced, indistinguishable to this contract — and
producing updated action-expert weights, VLM backbone frozen (matching the
paper's own reference training path); checkpointing so a run is resumable.

**Interface:**
```
FineTuneJob.run(checkpoint_path, episodes, output_path) -> FineTuneJob  # identity persists across the run
FineTuneJob.resume(checkpoint_path) -> FineTuneJob
```

**Interacts with:** LeRobotDataset-format episode data as input, regardless of
whether the episodes originated on the real robot or a simulator; produces
safetensors weights consumed by 01.1, 01.2, and (for the parity check) 01.4.

**Invariants held:** the VLM backbone stays frozen for the default training
path (matching the paper); training only the action expert is the default,
not full fine-tuning — a config flag switches this, it is never silently
inconsistent between a run and its checkpoint.

**Fails:** a job interrupted mid-run resumes from its last checkpoint, never
silently restarts from scratch nor silently continues from a corrupt
checkpoint (checksum or shape-validated on resume).

**Role:** the reference implementation, and the permanent fallback if 01.4
never clears its cutover gate — see
[ADR-0005](../../adr/0005-elixir-native-finetuning-conditional-retirement.md#adr-0005).

### 01.4 FineTuneJob (Elixir-native) — responsibility, interface, invariants

**Responsible for:** the identical `episodes -> updated weights` contract as
01.3, training loop expressed via `Nx.Defn.value_and_grad` directly against
01.2's already-hand-written forward pass — not `Axon`'s graph-building DSL,
which does not fit a model built as plain `Nx.Defn` functions — with
`Polaris` (Axon's optimizer implementations, usable standalone) providing
the Adam update step. Action-expert-only gradient updates with the VLM
backbone frozen (via differentiating only the trainable parameter subset),
matching 01.3's default training path.

**Interface:**
```elixir
FineTuneJob.run(checkpoint_path, episodes, output_path) :: FineTuneJob.t()
FineTuneJob.resume(checkpoint_path) :: FineTuneJob.t()
```

**Interacts with:** the same LeRobotDataset-format episodes as 01.3 (real or
simulated, same non-distinction), decoded via a shell-out to the system's
`ffmpeg` binary for raw video-frame extraction — a generic media-format
reader, not model or training logic, so it sits outside
[ADR-0004](../../adr/0004-weights-only-cross-runtime-sharing.md#adr-0004)'s
"no code crosses the boundary" scope the same way `Safetensors`/`Explorer`
already do for their own formats. Produces safetensors weights consumed by
01.1 and 01.2, identically to 01.3's output.

**Invariants held:** same frozen-backbone default as 01.3; a training run's
identity persists across resumption exactly as 01.3's does.

**Fails:** same loud/local checkpoint-corruption handling as 01.3 — never a
silent restart from scratch, never a silent continuation from a corrupt
checkpoint.

**Cutover gate:** promoted from "designed, not built" to the production
default only once a task-performance-parity check — fine-tuning both 01.3 and
01.4 on identical episodes, then comparing the resulting policies' [action-
accuracy proxy](CONTEXT.md#term-action-accuracy-proxy) on held-out evaluation
episodes, never their loss curves — shows 01.4 is not meaningfully worse.
**This gate has run (2026-07-13) and passed**: 01.4 is the production
trainer; 01.3 remains the reference implementation and permanent fallback
per ADR-0005's own fallback path. See this context's own "Pending updates"
section above for the real run's numbers, and
[ADR-0005](../../adr/0005-elixir-native-finetuning-conditional-retirement.md#adr-0005)
and
[ADR-0009](../../adr/0009-offline-action-accuracy-as-task-success-proxy.md#adr-0009).

### 01.5 InferenceServer (Elixir-native) — responsibility, interface, invariants

**Responsible for:** loading one 01.2 `SmolVLA` model once and holding it as
process state, then answering each
[infer_action](CONTEXT.md#term-infer-action-port) call against it — a named
GenServer so a caller on any node in the BEAM cluster reaches it by name. This
is the process that makes the in-process adapter (01.2) also a
cluster-addressable service, without changing the adapter itself.

**Interface:**
```elixir
InferenceServer.start_link(checkpoint_path) :: {:ok, pid()}   # loads the 01.2 model once
InferenceServer.infer_action(server, observation) :: {:ok, ActionChunk.t()} | {:error, reason}
# server may be a local name or {name, remote_node} — a plain GenServer.call target
```

**Interacts with:** the 01.2 `SmolVLA.infer_action/4` it wraps (its only model
dependency); callers reaching it through the
[infer_action port](CONTEXT.md#term-infer-action-port) — in-process on the same
node, or across BEAM distribution from a remote node (the
[demo](../demo/design.md)'s [sim node](../demo/CONTEXT.md#term-sim-node) is the
first such remote caller). The
cross-node case passes the [observation](CONTEXT.md#term-observation) and
returns the [action chunk](CONTEXT.md#term-action-chunk) as native BEAM terms;
no MessagePack or ZeroMQ is involved — that path stays exclusively the Python
fallback's. See
[ADR-0010](../../adr/0010-beam-distribution-orthogonal-to-infer-action-port.md#adr-0010).

**Invariants held:** the port contract is unchanged by where the caller sits —
one [observation](CONTEXT.md#term-observation) in, one
[action chunk](CONTEXT.md#term-action-chunk) out, the same `max_state_dim`
fail-loud bound 01.1 and 01.2 hold, honored identically for a local and a
remote call. Distribution is a topology axis orthogonal to the port, not a
third adapter (ADR-0010).

**Fails:** the model loads once at `start_link` — a bad checkpoint fails there,
loud and local, exactly as 01.2's `load` does, never a lazily-half-loaded
server. A malformed observation fails the same `max_state_dim` check before the
forward pass. A remote caller that loses the cluster connection sees a standard
distributed `GenServer.call` timeout/error — the server itself never blocks
waiting on a dead caller, and the calling
[ControlLoop](../control-loop/design.md) owns what a failed call means for the
tick (component 01.1 there).
