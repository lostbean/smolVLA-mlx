"""The cutover-gate comparison (issue 08, the final chunk of this build):
fine-tunes both the Python (``finetune_job.job.FineTuneJob``, component
01.3) and Elixir-native (``FineTuneJob``, component 01.4) trainers against
an IDENTICAL held-in episode subset of the real ``lerobot/svla_so101_pickplace``
dataset, evaluates both resulting policies' action-accuracy proxy and
throughput against an IDENTICAL held-out episode subset (never seen by
either training run), and records the resulting cutover judgment.

This is a ONE-SHOT comparison run, not a permanent service -- see
``docs/design/model-runtime/design.md`` component 01.4's "Cutover gate" and
``docs/adr/0005-elixir-native-finetuning-conditional-retirement.md``
(ADR-0005) for why the gate exists, and
``docs/adr/0009-offline-action-accuracy-as-task-success-proxy.md``
(ADR-0009) for why the metric is an offline proxy rather than a live
task-success rollout (this repo has no simulator or connected robot).

Modules:

``split``
    Splits a real LeRobotDataset directory into a training-episode subset
    and a held-out evaluation subset, materialized as two independent
    on-disk LeRobotDataset v3.0 directories (not just an index list) --
    so a training run pointed at the training directory has no way to see
    a held-out episode's data, structurally, not just by convention.

``metrics``
    The action-accuracy proxy (predicted action chunk vs. a held-out
    episode's real recorded actions) and throughput computation, plus the
    cutover-judgment threshold logic. Pure functions, tested against
    synthetic data first (this module's own test suite).

``run_gate``
    The orchestrating script: builds the split, invokes both trainers,
    evaluates both resulting policies, and writes the comparison report.
    Real, slow, wall-clock-heavy -- not part of the fast test gate, mirrors
    this repo's own established ``RUN_*_INTEGRATION_CHECK=1`` opt-in
    convention for real-checkpoint runs.
"""
