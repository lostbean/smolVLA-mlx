Status: ready-for-agent
Category: enhancement

## Parent

[model-runtime design](../../../docs/design/model-runtime/design.md), component 01.4 (`FineTuneJob (Elixir-native)`). Pending build entry, [ADR-0005](../../../docs/adr/0005-elixir-native-finetuning-conditional-retirement.md#adr-0005).

## What to build

Implement the identical `FineTuneJob` contract — episodes in, updated
action-expert weights out — via `Nx`'s autodiff and `Axon`'s training API,
mirroring the Python trainer's default behavior (frozen VLM backbone,
action-expert-only gradient updates) but as an independent implementation
sharing no code with the Python trainer (the **"Weights are the only
cross-runtime artifact"** invariant, applied to training exactly as it
already applies to inference).

This is the intended target trainer, conditional on the parity gate in the
next slice — until that gate clears, this stays an evaluated candidate, not
the production trainer (see the component's own **"Cutover gate"** note and
[ADR-0005](../../../docs/adr/0005-elixir-native-finetuning-conditional-retirement.md#adr-0005)).

## Acceptance criteria

- [ ] A fine-tuning run against the same episode set used in slice 06
      produces updated action-expert weights, loadable through either
      `infer_action` adapter (Python from slice 02, or emily-native from
      slice 05).
- [ ] The VLM backbone stays frozen by default, matching the Python
      trainer's behavior; a training run's identity persists across
      resumption exactly as the Python trainer's does.
- [ ] An interrupted run resumes from its last checkpoint; a corrupt
      checkpoint is detected rather than silently continued from — same
      loud/local failure shape as the Python trainer.
- [ ] The resulting weights are structurally compatible with both
      inference adapters (same safetensors shape as the Python trainer's
      output) — no format divergence between the two trainers' outputs.

## Out of scope

The task-performance-parity gate itself (next slice — this slice only
produces a trained candidate, it does not judge it); retiring the Python
trainer (that decision belongs to the next slice's outcome, per ADR-0005);
reinforcement learning (standing no-goal).

## Blocked by

06-python-finetune, 05-emily-native-full-scale
