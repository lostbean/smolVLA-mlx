# model-runtime — glossary

_Register_: robotics/ML systems vocabulary — checkpoint, weights, forward
pass, episode. Reject generic web/business idiom (no "endpoint" for a model
call, no "pipeline" as a catch-all).

### Observation {#term-observation}

One tick's input to the model: camera image(s), robot proprioceptive state
(compressed to a single token per SmolVLA's own architecture), and a language
instruction. A value object — no identity across time.

### Action chunk {#term-action-chunk}

The continuous-valued sequence of actions produced by one
[infer_action port](#term-infer-action-port) call — SmolVLA's own default
length is 50. A value object; a new call always produces a new chunk, never
mutates a prior one.

### infer_action port {#term-infer-action-port}

The one contract both [model-runtime](../model-runtime/design.md) adapters
implement: [observation](#term-observation) in, [action
chunk](#term-action-chunk) out, synchronous, no opinion about queueing or
timing. _Avoid_: "inference endpoint" (network idiom implying HTTP; this port
has two adapters, only one of which is networked), "the API" (too generic to
name the specific contract).

### Action expert {#term-action-expert}

The flow-matching transformer component of SmolVLA that consumes the frozen
VLM backbone's encoding and produces an [action chunk](#term-action-chunk).
The only component fine-tuning normally updates.

### Episode {#term-episode}

One recorded demonstration — an observation/action sequence — in
LeRobotDataset format, used as [FineTuneJob](design.md) input (components
01.3, 01.4). A value object; its provenance (real robot usage or a simulation
environment) never changes its shape or how a `FineTuneJob` adapter consumes
it.
