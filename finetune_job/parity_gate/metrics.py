"""The action-accuracy proxy, throughput, and cutover-judgment threshold
logic for the parity gate (issue 08).

Per ADR-0009 and the CONTEXT term "Action-accuracy proxy"
(docs/design/model-runtime/CONTEXT.md): this is an OFFLINE proxy for task
success, not a live-rollout result -- a fine-tuned policy's predicted
action chunk compared against a held-out episode's real recorded
(ground-truth) action, never task-completion observed by actually running
the action. Every docstring/report string this module produces says so
explicitly (never "task success rate" unqualified, per the CONTEXT term's
own _Avoid_ note).

Per ADR-0005's explicit rejection: this module never compares training
loss values -- only the accuracy proxy (distance between predicted and
real actions) and throughput (a pure compute-performance dimension,
orthogonal to accuracy).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Sequence


@dataclass(frozen=True)
class AccuracyProxyResult:
    """One policy's action-accuracy proxy over a held-out episode set:
    mean/max absolute error between each predicted action chunk and the
    matching held-out frame's real recorded action, averaged per-frame
    across every evaluated frame (not per-episode -- a longer held-out
    episode contributes proportionally more evaluated frames, matching how
    the raw prediction errors are actually distributed). NEVER a
    "task success rate" -- an offline proxy only (ADR-0009)."""

    mean_absolute_error: float
    max_absolute_error: float
    per_episode_mean_absolute_error: dict[int, float]
    n_frames_evaluated: int


@dataclass(frozen=True)
class ThroughputResult:
    """Actions (equivalently, images/observations -- one `infer_action`
    call produces one action chunk from one observation) processed per
    second, over the same evaluated frame set. A pure compute-performance
    dimension, orthogonal to the accuracy proxy (ADR-0005's own explicit
    framing) -- a fast, inaccurate policy and a slow, accurate one are
    both real, separately-reportable outcomes."""

    actions_per_second: float
    seconds_per_action: float
    n_calls: int
    total_seconds: float


def action_chunk_absolute_error(predicted_chunk, ground_truth_action) -> float:
    """Mean absolute error between one predicted action CHUNK (shape
    `(chunk_size, action_dim)`, the raw `infer_action` output) and one
    real ground-truth action (shape `(action_dim,)`, one dataset frame's
    own `action` column) -- ground truth is broadcast across the chunk's
    time axis since a single held-out frame carries one recorded action,
    not a multi-step recorded chunk (LeRobotDataset's own per-frame
    action column; see `lib/finetune_job.ex`'s own `sample_batch/4`
    comment on this same real-data shape, which this evaluation mirrors
    for consistency with what both trainers were actually trained
    against).

    `predicted_chunk`/`ground_truth_action` accept anything exposing
    `len()`/nested indexing (a plain nested list, a numpy array, an
    `mx.array` converted via `.tolist()`, an `Nx.Tensor` converted via
    `Nx.to_flat_list/1` reshaped by the caller) -- kept a pure-Python
    numeric function, no MLX/Nx import, so it runs identically regardless
    of which language produced the prediction.
    """
    chunk = _to_nested_list(predicted_chunk)
    action = _to_flat_list(ground_truth_action)

    if not chunk:
        raise ValueError("predicted_chunk must have at least one timestep")

    action_dim = len(action)
    errors = []
    for step in chunk:
        step = list(step)[:action_dim]
        if len(step) != action_dim:
            raise ValueError(
                f"predicted_chunk step has {len(step)} dims, ground_truth_action has "
                f"{action_dim} -- cannot compare mismatched action dimensionality"
            )
        errors.extend(abs(p - g) for p, g in zip(step, action))

    return sum(errors) / len(errors)


def compute_accuracy_proxy(
    episode_predictions: dict[int, list[tuple[object, object]]],
) -> AccuracyProxyResult:
    """Aggregates `action_chunk_absolute_error` over every evaluated
    (predicted_chunk, ground_truth_action) pair in every held-out episode.

    `episode_predictions`: `{episode_index: [(predicted_chunk,
    ground_truth_action), ...], ...}` -- one entry per evaluated frame
    within that episode (a caller may subsample frames per episode rather
    than evaluating every single one; this function does not care, it
    only aggregates whatever pairs it is given).

    Raises `ValueError` if given no episodes or an episode with no pairs
    (an empty accuracy-proxy result would silently look like "zero
    error", which is misleading -- never computed silently over nothing).
    """
    if not episode_predictions:
        raise ValueError(
            "compute_accuracy_proxy needs at least one episode's predictions"
        )

    per_episode_mean: dict[int, float] = {}
    all_errors: list[float] = []

    for episode_index, pairs in episode_predictions.items():
        if not pairs:
            raise ValueError(
                f"episode {episode_index} has zero (prediction, ground_truth) pairs"
            )
        episode_errors = [action_chunk_absolute_error(pred, gt) for pred, gt in pairs]
        per_episode_mean[episode_index] = sum(episode_errors) / len(episode_errors)
        all_errors.extend(episode_errors)

    return AccuracyProxyResult(
        mean_absolute_error=sum(all_errors) / len(all_errors),
        max_absolute_error=max(all_errors),
        per_episode_mean_absolute_error=per_episode_mean,
        n_frames_evaluated=len(all_errors),
    )


def measure_throughput(
    infer_fn: Callable[[], object], *, n_calls: int, warmup_calls: int = 1
) -> ThroughputResult:
    """Times `n_calls` real invocations of `infer_fn` (a zero-arg closure
    that runs one real `infer_action` call and returns its result,
    unused here beyond forcing evaluation), after `warmup_calls` untimed
    warmup calls (excludes one-time model/JIT/compilation warmup from the
    steady-state throughput number -- both this repo's own design doc
    (component 01.2's "warm latency") and its real accepted benchmarks
    already draw this same warm/cold distinction).

    Real wall-clock timing (`time.perf_counter`), never estimated or
    modeled.
    """
    if n_calls <= 0:
        raise ValueError(f"n_calls must be positive, got {n_calls}")

    for _ in range(warmup_calls):
        infer_fn()

    start = time.perf_counter()
    for _ in range(n_calls):
        infer_fn()
    elapsed = time.perf_counter() - start

    return ThroughputResult(
        actions_per_second=n_calls / elapsed if elapsed > 0 else float("inf"),
        seconds_per_action=elapsed / n_calls,
        n_calls=n_calls,
        total_seconds=elapsed,
    )


@dataclass(frozen=True)
class CutoverThreshold:
    """The concrete, documented "not meaningfully worse" threshold
    (ADR-0005 names the concept but pins no number -- this chunk's own
    work order explicitly calls for one, "pick something defensible and
    STATE your reasoning").

    `max_accuracy_regression_fraction=0.20`: the Elixir-native policy's
    mean absolute error may be at most 20% WORSE (higher) than the Python
    policy's, i.e. `elixir_mae <= python_mae * 1.20`. Reasoning: the
    accuracy proxy is itself a noisy estimate over a small (6-episode)
    held-out set and a short (laptop-scale) training run for BOTH
    trainers -- a stringent bar (e.g. 5%) would be dominated by that
    noise floor rather than by genuine trainer quality differences, while
    a very loose bar (e.g. 2x) would let a materially worse trainer
    through. 20% is a common tolerance band for this kind of small-sample
    imitation-learning proxy comparison (enough headroom to absorb
    run-to-run variance from batch sampling/random noise/seed, tight
    enough to catch a trainer that is really, structurally worse, not
    just unlucky). This is a documented judgment call, not a value ADR-0005
    itself specifies -- ADR-0005 names the CONCEPT ("not meaningfully
    worse") but deliberately leaves the number to the gate that actually
    runs, which is this chunk.

    `min_throughput_fraction=0.10`: the Elixir-native policy's throughput
    (actions/sec via its OWN inference adapter) must be at least 10% of
    the Python policy's. Reasoning: throughput is explicitly "orthogonal"
    to accuracy per ADR-0005/ADR-0009 -- a real, already-documented,
    already-accepted finding (component 01.2's own design text) is that
    the Elixir-native inference adapter's warm latency is ~12x OVER the
    100ms control-loop budget for a structural reason (per-op dispatch
    instead of one traced graph) that is explicitly flagged as separate,
    already-known, un-closed follow-up work -- NOT something this gate
    should re-discover as a surprise or treat as disqualifying on its
    own, since the design document already scopes closing it as future
    work outside this chunk. A 10% floor exists so throughput still
    counts for something in the judgment (a policy that is 1000x slower
    would be a real, reportable problem) without making the ALREADY-KNOWN,
    ALREADY-DOCUMENTED latency gap alone veto the cutover -- the design
    layer's own text treats it as an open item to close later, not a
    gate-failing defect today.
    """

    max_accuracy_regression_fraction: float = 0.20
    min_throughput_fraction: float = 0.10


@dataclass(frozen=True)
class CutoverJudgment:
    """The gate's recorded decision, with the real numbers that produced
    it -- never just a bare PASS/FAIL (this chunk's own explicit
    acceptance criterion: "auditable rather than hidden inside the gate's
    own code")."""

    python_accuracy: AccuracyProxyResult
    elixir_accuracy: AccuracyProxyResult
    python_throughput: ThroughputResult
    elixir_throughput: ThroughputResult
    threshold: CutoverThreshold
    accuracy_regression_fraction: float
    throughput_fraction: float
    accuracy_passes: bool
    throughput_passes: bool

    @property
    def passes(self) -> bool:
        return self.accuracy_passes and self.throughput_passes

    def summary(self) -> str:
        verdict = (
            "PASS -- cutover to Elixir-native as production trainer"
            if self.passes
            else (
                "FAIL -- Python trainer remains production, Elixir-native stays evaluated candidate"
            )
        )
        return (
            f"Accuracy proxy (mean abs error, LOWER is better; NOT a live task-success "
            f"rate -- see ADR-0009):\n"
            f"  Python  (component 01.3): {self.python_accuracy.mean_absolute_error:.6f} "
            f"(max {self.python_accuracy.max_absolute_error:.6f}, "
            f"n={self.python_accuracy.n_frames_evaluated} frames)\n"
            f"  Elixir  (component 01.4): {self.elixir_accuracy.mean_absolute_error:.6f} "
            f"(max {self.elixir_accuracy.max_absolute_error:.6f}, "
            f"n={self.elixir_accuracy.n_frames_evaluated} frames)\n"
            f"  Elixir/Python regression: {self.accuracy_regression_fraction:+.1%} "
            f"(threshold: <= {self.threshold.max_accuracy_regression_fraction:.0%})\n"
            f"  Accuracy sub-verdict: {'PASS' if self.accuracy_passes else 'FAIL'}\n\n"
            f"Throughput (actions/sec, HIGHER is better; each side's own infer_action "
            f"adapter):\n"
            f"  Python  (component 01.1): {self.python_throughput.actions_per_second:.4f} "
            f"actions/sec ({self.python_throughput.seconds_per_action:.4f}s/action)\n"
            f"  Elixir  (component 01.2): {self.elixir_throughput.actions_per_second:.4f} "
            f"actions/sec ({self.elixir_throughput.seconds_per_action:.4f}s/action)\n"
            f"  Elixir/Python ratio: {self.throughput_fraction:.1%} "
            f"(threshold: >= {self.threshold.min_throughput_fraction:.0%})\n"
            f"  Throughput sub-verdict: {'PASS' if self.throughput_passes else 'FAIL'}\n\n"
            f"Overall: {verdict}"
        )


def judge_cutover(
    python_accuracy: AccuracyProxyResult,
    elixir_accuracy: AccuracyProxyResult,
    python_throughput: ThroughputResult,
    elixir_throughput: ThroughputResult,
    threshold: CutoverThreshold | None = None,
) -> CutoverJudgment:
    """Applies `CutoverThreshold` to a real pair of accuracy-proxy and
    throughput measurements, producing the gate's recorded, auditable
    judgment. Pure function over already-measured real numbers -- no
    training or inference happens here."""
    threshold = threshold or CutoverThreshold()

    accuracy_regression_fraction = (
        elixir_accuracy.mean_absolute_error - python_accuracy.mean_absolute_error
    ) / python_accuracy.mean_absolute_error

    throughput_fraction = (
        elixir_throughput.actions_per_second / python_throughput.actions_per_second
    )

    accuracy_passes = (
        accuracy_regression_fraction <= threshold.max_accuracy_regression_fraction
    )
    throughput_passes = throughput_fraction >= threshold.min_throughput_fraction

    return CutoverJudgment(
        python_accuracy=python_accuracy,
        elixir_accuracy=elixir_accuracy,
        python_throughput=python_throughput,
        elixir_throughput=elixir_throughput,
        threshold=threshold,
        accuracy_regression_fraction=accuracy_regression_fraction,
        throughput_fraction=throughput_fraction,
        accuracy_passes=accuracy_passes,
        throughput_passes=throughput_passes,
    )


def _to_nested_list(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    return [list(row) if hasattr(row, "__iter__") else [row] for row in value]


def _to_flat_list(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


__all__ = [
    "AccuracyProxyResult",
    "ThroughputResult",
    "CutoverThreshold",
    "CutoverJudgment",
    "action_chunk_absolute_error",
    "compute_accuracy_proxy",
    "measure_throughput",
    "judge_cutover",
]
