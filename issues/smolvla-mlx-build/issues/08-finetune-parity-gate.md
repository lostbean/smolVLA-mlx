Status: ready-for-agent
Category: enhancement

## Parent

[model-runtime design](../../../docs/design/model-runtime/design.md), component 01.4's "Cutover gate". [ADR-0005](../../../docs/adr/0005-elixir-native-finetuning-conditional-retirement.md#adr-0005).

## What to build

The task-performance-parity check that decides whether Elixir-native
fine-tuning (slice 07) becomes the production trainer or Python fine-tuning
(slice 06) remains the permanent trainer with weights migrated over, per
ADR-0005.

Fine-tune with both trainers on an identical episode set (same real-or-
simulated episodes, same starting checkpoint). Compare the two resulting
policies' **task success rate on held-out evaluation episodes** — not their
training loss curves, which the design explicitly rejects as the wrong bar
(two correct trainers can diverge in loss trajectory while converging to
equally good policies).

This slice is a decision point, not a blocking dependency for anything else:
its outcome determines which trainer is "production" going forward, but
both slices 06 and 07 already exist and work independently of this gate's
result.

## Acceptance criteria

- [ ] Both trainers run against the identical episode set and starting
      checkpoint, producing two independently loadable weight sets.
- [ ] A held-out evaluation episode set (not used in either training run)
      has a defined task-success metric, applied identically to both
      resulting policies.
- [ ] The comparison report states, in absolute terms, how close the two
      task-success numbers are — not just a pass/fail verdict — so the
      "not meaningfully worse" judgment in ADR-0005 is auditable rather than
      hidden inside the gate's own code.
- [ ] The design layer's pending ledger entry for Elixir-native fine-tuning
      is updated to reflect this gate's outcome (cleared to "production" or
      left open with the documented gap, per ADR-0005's fallback path).

## Out of scope

Any change to either trainer's implementation based on this gate's result
(a failed gate does not imply "go fix the Elixir trainer" as part of this
slice — that would be new, separately-scoped work); reinforcement learning.

## Blocked by

06-python-finetune, 07-elixir-native-finetune
