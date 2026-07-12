Status: ready-for-human
Category: enhancement

# smolVLA-mlx — build sequence

Source: the design layer at [docs/design/design.md](../../docs/design/design.md)
and its two contexts,
[model-runtime](../../docs/design/model-runtime/design.md) and
[control-loop](../../docs/design/control-loop/design.md). This is not a new
plan — it's the pending-ledger build entries (5 of them, as of this writing)
sliced into a dependency-ordered sequence of tracer-bullet issues.

## Sequencing rationale

Get one real, in-process Python inference call working first (issues 01-02)
— the cheapest path to something real and demoable, and nothing else in the
system can run without a loadable checkpoint. Then prove the whole
cross-language loop over the lower-risk ZeroMQ/Python fallback path
(issues 03-04) before swapping in the higher-payoff but higher-risk
emily-native path (issue 05) — if the full-scale port surprises us, the
system still works end to end on the fallback while that's debugged.
Fine-tuning (issues 06-08) can start as soon as issue 01 lands (it needs the
same checkpoint machinery) and proceeds in parallel with issues 03-05; the
Elixir-native trainer and its parity gate come last because they need
something to compare against.

## Sequence

1. [01-smolvla-model-class](issues/01-smolvla-model-class.md)
2. [02-infer-action-python](issues/02-infer-action-python.md) — blocked by 01
3. [03-zeromq-server](issues/03-zeromq-server.md) — blocked by 02
4. [04-control-loop-zeromq](issues/04-control-loop-zeromq.md) — blocked by 03
   — **first full end-to-end demo of the whole system**
5. [05-emily-native-full-scale](issues/05-emily-native-full-scale.md) —
   blocked by 01, 04
6. [06-python-finetune](issues/06-python-finetune.md) — blocked by 01 (can
   run in parallel with 03-05)
7. [07-elixir-native-finetune](issues/07-elixir-native-finetune.md) —
   blocked by 06, 05
8. [08-finetune-parity-gate](issues/08-finetune-parity-gate.md) — blocked by
   06, 07 — decision point, not a blocker for anything downstream
