Status: ready-for-agent
Category: enhancement

## Parent

[model-runtime design](../../../docs/design/model-runtime/design.md), component 01.3 (`FineTuneJob (Python)`). Pending build entry, [ADR-0001](../../../docs/adr/0001-parallel-action-entry-point.md#adr-0001).

## What to build

Implement the Python `FineTuneJob`: takes a set of `{#term-episode}`s and the
frozen VLM backbone, produces updated action-expert weights, checkpointed so
an interrupted run resumes rather than restarting. Reuses LeRobot's dataset
format and training conventions rather than inventing a new one (the **"Not
a second training framework"** no-goal).

An episode's provenance — real robot usage or a simulation environment —
never changes this contract; both are the same `{#term-episode}` shape and
are indistinguishable to `FineTuneJob`. This is the reference
implementation and the permanent fallback trainer (see
[ADR-0005](../../../docs/adr/0005-elixir-native-finetuning-conditional-retirement.md#adr-0005)).

The default training path freezes the VLM backbone and updates only the
action expert, matching SmolVLA's own paper. A config flag may switch to
full fine-tuning, but a run and its checkpoint are never silently
inconsistent about which mode produced them.

## Acceptance criteria

- [ ] A fine-tuning run against a small set of real episode data produces
      updated action-expert weights, loadable back through the Python
      `infer_action` (slice 02).
- [ ] The same run against simulation-sourced episodes (same
      `{#term-episode}` shape, different origin) succeeds identically — no
      code path branches on episode provenance.
- [ ] An interrupted run resumes from its last checkpoint rather than
      restarting from scratch, and a corrupt checkpoint is detected
      (checksum or shape-validated) rather than silently continued from.
- [ ] The VLM backbone stays frozen in the default training path; the
      config flag for full fine-tuning is tested and does not silently
      desync a run from its checkpoint's recorded mode.

## Out of scope

The Elixir-native trainer (next slice); the task-performance-parity gate
between the two trainers (slice 08); reinforcement learning of any kind
(standing no-goal — this is supervised fine-tuning only).

## Blocked by

01-smolvla-model-class
