"""Unit tests for the accuracy-proxy/throughput/threshold logic
(parity_gate.metrics) -- TDD directive step (2): "the accuracy-proxy metric
computation, tested against synthetic/toy data first for correctness"
-- entirely synthetic, no real trainer or checkpoint involved.
"""

import time

import pytest

from finetune_job.parity_gate.metrics import (
    AccuracyProxyResult,
    CutoverThreshold,
    action_chunk_absolute_error,
    compute_accuracy_proxy,
    judge_cutover,
    measure_throughput,
)


def test_action_chunk_absolute_error_is_zero_for_a_perfect_prediction():
    chunk = [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
    ground_truth = [1.0, 2.0, 3.0]
    assert action_chunk_absolute_error(chunk, ground_truth) == pytest.approx(0.0)


def test_action_chunk_absolute_error_matches_hand_computed_value():
    # One timestep, ground truth [0, 0], predicted [1, 3] -> |1-0| + |3-0| = 4, /2 dims = 2.0
    chunk = [[1.0, 3.0]]
    ground_truth = [0.0, 0.0]
    assert action_chunk_absolute_error(chunk, ground_truth) == pytest.approx(2.0)


def test_action_chunk_absolute_error_averages_across_multiple_timesteps():
    # timestep 1 error: |1-0|+|1-0| = 2 (avg 1.0); timestep 2 error: |3-0|+|3-0|=6 (avg 3.0)
    # overall mean over all (timestep, dim) entries: (1+1+3+3)/4 = 2.0
    chunk = [[1.0, 1.0], [3.0, 3.0]]
    ground_truth = [0.0, 0.0]
    assert action_chunk_absolute_error(chunk, ground_truth) == pytest.approx(2.0)


def test_action_chunk_absolute_error_handles_padded_action_dim_wider_than_ground_truth():
    # predicted chunk carries max_action_dim=4 slots (padding beyond the
    # real 2-dim action space, matching how both trainers pad); only the
    # first 2 dims are compared against the real 2-dim ground truth.
    chunk = [[1.0, 1.0, 999.0, 999.0]]
    ground_truth = [0.0, 0.0]
    assert action_chunk_absolute_error(chunk, ground_truth) == pytest.approx(1.0)


def test_action_chunk_absolute_error_rejects_empty_chunk():
    with pytest.raises(ValueError):
        action_chunk_absolute_error([], [0.0, 0.0])


def test_compute_accuracy_proxy_aggregates_per_episode_and_overall():
    episode_predictions = {
        0: [([[1.0, 1.0]], [0.0, 0.0])],  # error 1.0
        1: [([[2.0, 2.0]], [0.0, 0.0])],  # error 2.0
    }
    result = compute_accuracy_proxy(episode_predictions)

    assert isinstance(result, AccuracyProxyResult)
    assert result.per_episode_mean_absolute_error[0] == pytest.approx(1.0)
    assert result.per_episode_mean_absolute_error[1] == pytest.approx(2.0)
    assert result.mean_absolute_error == pytest.approx(1.5)
    assert result.max_absolute_error == pytest.approx(2.0)
    assert result.n_frames_evaluated == 2


def test_compute_accuracy_proxy_weights_by_frame_not_by_episode():
    # Episode 0 has 3 frames all with error 0; episode 1 has 1 frame with
    # error 4.0. Per-frame mean = (0+0+0+4)/4 = 1.0, NOT the per-episode
    # average of (0 + 4)/2 = 2.0 -- frame-weighted aggregation, per this
    # module's own docstring.
    episode_predictions = {
        0: [
            ([[0.0]], [0.0]),
            ([[0.0]], [0.0]),
            ([[0.0]], [0.0]),
        ],
        1: [([[4.0]], [0.0])],
    }
    result = compute_accuracy_proxy(episode_predictions)
    assert result.mean_absolute_error == pytest.approx(1.0)


def test_compute_accuracy_proxy_rejects_empty_input():
    with pytest.raises(ValueError):
        compute_accuracy_proxy({})


def test_compute_accuracy_proxy_rejects_episode_with_no_pairs():
    with pytest.raises(ValueError):
        compute_accuracy_proxy({0: []})


def test_measure_throughput_counts_real_calls_and_excludes_warmup():
    call_count = {"n": 0}

    def fake_infer():
        call_count["n"] += 1
        time.sleep(0.001)
        return call_count["n"]

    result = measure_throughput(fake_infer, n_calls=5, warmup_calls=2)

    assert call_count["n"] == 7  # 2 warmup + 5 timed
    assert result.n_calls == 5
    assert result.actions_per_second > 0
    assert result.seconds_per_action > 0
    assert result.total_seconds > 0


def test_measure_throughput_rejects_non_positive_n_calls():
    with pytest.raises(ValueError):
        measure_throughput(lambda: None, n_calls=0)


def test_judge_cutover_passes_when_elixir_matches_python():
    from finetune_job.parity_gate.metrics import ThroughputResult

    python_acc = AccuracyProxyResult(0.1, 0.2, {0: 0.1}, 10)
    elixir_acc = AccuracyProxyResult(0.1, 0.2, {0: 0.1}, 10)  # identical
    python_tp = ThroughputResult(10.0, 0.1, 5, 0.5)
    elixir_tp = ThroughputResult(10.0, 0.1, 5, 0.5)  # identical

    judgment = judge_cutover(python_acc, elixir_acc, python_tp, elixir_tp)

    assert judgment.passes
    assert judgment.accuracy_passes
    assert judgment.throughput_passes
    assert judgment.accuracy_regression_fraction == pytest.approx(0.0)
    assert judgment.throughput_fraction == pytest.approx(1.0)


def test_judge_cutover_fails_on_accuracy_regression_beyond_threshold():
    from finetune_job.parity_gate.metrics import ThroughputResult

    python_acc = AccuracyProxyResult(0.1, 0.2, {0: 0.1}, 10)
    elixir_acc = AccuracyProxyResult(0.5, 0.6, {0: 0.5}, 10)  # 5x worse
    python_tp = ThroughputResult(10.0, 0.1, 5, 0.5)
    elixir_tp = ThroughputResult(10.0, 0.1, 5, 0.5)

    judgment = judge_cutover(python_acc, elixir_acc, python_tp, elixir_tp)

    assert not judgment.passes
    assert not judgment.accuracy_passes
    assert judgment.throughput_passes


def test_judge_cutover_fails_on_throughput_far_below_threshold():
    from finetune_job.parity_gate.metrics import ThroughputResult

    python_acc = AccuracyProxyResult(0.1, 0.2, {0: 0.1}, 10)
    elixir_acc = AccuracyProxyResult(0.1, 0.2, {0: 0.1}, 10)
    python_tp = ThroughputResult(100.0, 0.01, 5, 0.05)
    elixir_tp = ThroughputResult(1.0, 1.0, 5, 5.0)  # 100x slower -> 1% of python

    judgment = judge_cutover(python_acc, elixir_acc, python_tp, elixir_tp)

    assert not judgment.passes
    assert judgment.accuracy_passes
    assert not judgment.throughput_passes


def test_judge_cutover_passes_within_the_documented_regression_band():
    from finetune_job.parity_gate.metrics import ThroughputResult

    # Exactly at the 20% regression boundary -- should still pass (<=).
    python_acc = AccuracyProxyResult(0.100, 0.2, {0: 0.100}, 10)
    elixir_acc = AccuracyProxyResult(0.120, 0.2, {0: 0.120}, 10)
    python_tp = ThroughputResult(10.0, 0.1, 5, 0.5)
    elixir_tp = ThroughputResult(10.0, 0.1, 5, 0.5)

    judgment = judge_cutover(python_acc, elixir_acc, python_tp, elixir_tp)
    assert judgment.accuracy_passes


def test_judge_cutover_uses_custom_threshold_when_given():
    from finetune_job.parity_gate.metrics import ThroughputResult

    python_acc = AccuracyProxyResult(0.100, 0.2, {0: 0.100}, 10)
    elixir_acc = AccuracyProxyResult(0.150, 0.2, {0: 0.150}, 10)  # 50% worse
    python_tp = ThroughputResult(10.0, 0.1, 5, 0.5)
    elixir_tp = ThroughputResult(10.0, 0.1, 5, 0.5)

    strict = CutoverThreshold(max_accuracy_regression_fraction=0.05)
    lenient = CutoverThreshold(max_accuracy_regression_fraction=1.0)

    assert not judge_cutover(
        python_acc, elixir_acc, python_tp, elixir_tp, strict
    ).passes
    assert judge_cutover(python_acc, elixir_acc, python_tp, elixir_tp, lenient).passes


def test_judgment_summary_reports_absolute_numbers_not_just_pass_fail():
    from finetune_job.parity_gate.metrics import ThroughputResult

    python_acc = AccuracyProxyResult(0.1234, 0.5, {0: 0.1234}, 10)
    elixir_acc = AccuracyProxyResult(0.1300, 0.5, {0: 0.1300}, 10)
    python_tp = ThroughputResult(12.5, 0.08, 5, 0.4)
    elixir_tp = ThroughputResult(0.9, 1.11, 5, 5.55)

    judgment = judge_cutover(python_acc, elixir_acc, python_tp, elixir_tp)
    summary = judgment.summary()

    # The report must show real absolute numbers, not just PASS/FAIL --
    # this chunk's own explicit acceptance criterion.
    assert "0.1234" in summary
    assert "0.1300" in summary
    assert "12.5" in summary
    assert "0.9" in summary
    assert "ADR-0009" in summary  # proxy-limitation framing present
