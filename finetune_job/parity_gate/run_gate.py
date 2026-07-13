"""The parity gate's real, end-to-end orchestrator (issue 08, the final
chunk of this build sequence).

Runs the full cutover-gate comparison per
``docs/design/model-runtime/design.md`` component 01.4's "Cutover gate":

  1. Splits the real ``lerobot/svla_so101_pickplace`` dataset into a
     training subset and a held-out evaluation subset
     (``parity_gate.split``).
  2. Fine-tunes BOTH the Python (``finetune_job.job.FineTuneJob``,
     component 01.3) and Elixir-native (``FineTuneJob``, component 01.4)
     trainers against the IDENTICAL training subset and the IDENTICAL
     starting checkpoint (``lerobot/smolvla_base``).
  3. Evaluates BOTH resulting policies' action-accuracy proxy and
     throughput against the IDENTICAL held-out episode subset -- the
     Python policy through the Python ``infer_action`` (component 01.1),
     the Elixir policy through the Elixir-native ``infer_action/4``
     (component 01.2), via ``eval_elixir_policy.exs``.
  4. Applies ``parity_gate.metrics.judge_cutover`` and writes a real,
     inspectable comparison report (JSON + a human-readable summary).

This is deliberately NOT part of the fast test gate -- a real run,
real wall-clock minutes for both trainers plus both evaluations, not
mocked. Opt in with ``RUN_PARITY_GATE=1``, this repo's own established
``RUN_*_INTEGRATION_CHECK``-style convention:

    RUN_PARITY_GATE=1 uv run python -m finetune_job.parity_gate.run_gate

Override sizing via environment variables (see ``_env_int``/``_env_str``
calls below for the full list and defaults) -- the defaults themselves are
this chunk's own documented, real, run choice (see this module's
``DEFAULT_*`` constants and the accompanying report for the reasoning).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from finetune_job.job import Episodes, FineTuneJob
from finetune_job.parity_gate.metrics import (
    AccuracyProxyResult,
    ThroughputResult,
    compute_accuracy_proxy,
    judge_cutover,
    measure_throughput,
)
from finetune_job.parity_gate.split import build_train_only_dataset, split_episodes

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CHECKPOINT = "lerobot/smolvla_base"
DEFAULT_DATASET_REPO_ID = "lerobot/svla_so101_pickplace"
DEFAULT_DATASET_ROOT = (
    "~/.cache/huggingface/lerobot/hub/datasets--lerobot--svla_so101_pickplace/"
    "snapshots/f641879e22172be7e8161d5e6c1503c2d2feb657"
)
DEFAULT_ELIXIR_CHECKPOINT = (
    "~/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/"
    "c83c3163b8ca9b7e67c509fffd9121e66cb96205"
)

# --- Real run sizing, documented (work order: "document and justify
# whatever step count/duration you choose") ---
#
# N_HOLDOUT=6 episodes (~12% of 50): see split.py's own docstring for the
# full reasoning -- enough to average out one episode's idiosyncrasies,
# small enough to leave the bulk of the dataset for training.
#
# STEPS: measured directly on this machine before picking a number (never
# guessed) -- Python/MPS trains at roughly 2s/step steady-state after a
# ~25s fixed model-load/dataset-setup cost (10 real steps measured at
# ~28s total); Elixir/emily trains at roughly 27s/step (measured directly
# via this repo's own real_checkpoint_test.exs timing, and reconfirmed
# with a fresh 2-step timed probe during this chunk's own work: 53.6s for
# 2 steps). 20 steps: ~65s for Python, ~9 minutes for Elixir -- the
# Elixir side dominates the real wall-clock budget, so 20 steps keeps the
# WHOLE gate (both trainers, both evaluations) inside roughly 10-15 real
# minutes, several times more real gradient updates than either trainer's
# own prior acceptance runs (2-6 steps) without becoming impractical to
# run in one sitting.
DEFAULT_N_HOLDOUT = 6
DEFAULT_STEPS = 20
DEFAULT_BATCH_SIZE = 2
DEFAULT_N_EVAL_FRAMES_PER_EPISODE = 5
DEFAULT_N_THROUGHPUT_CALLS = 10

_REAL_DATASET_EXTRA_ARGS = [
    "--policy.device=mps",
    "--dataset.video_backend=pyav",
    "--rename_map={"
    '"observation.images.side": "observation.images.camera1", '
    '"observation.images.up": "observation.images.camera2"}',
]


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _find_last_step_dir(checkpoints_dir: Path) -> Path:
    step_dirs = sorted(
        (p for p in checkpoints_dir.iterdir() if p.is_dir() and p.name != "last"),
        key=lambda p: int(p.name),
    )
    if not step_dirs:
        raise RuntimeError(f"no step checkpoint directories under {checkpoints_dir}")
    return step_dirs[-1]


def run_python_training(
    checkpoint: str,
    train_dataset_root: Path,
    output_dir: Path,
    *,
    steps: int,
    batch_size: int,
) -> Path:
    """Fine-tunes via the Python trainer (component 01.3) against the
    train-only dataset copy, returning the resulting pretrained_model dir
    (loadable via the Python ``infer_action``)."""
    print(f"[python] training {steps} steps, batch_size={batch_size} ...")
    t0 = time.time()
    FineTuneJob.run(
        checkpoint_path=checkpoint,
        episodes=Episodes(
            repo_id=DEFAULT_DATASET_REPO_ID, root=str(train_dataset_root)
        ),
        output_path=str(output_dir),
        steps=steps,
        batch_size=batch_size,
        extra_args=[*_REAL_DATASET_EXTRA_ARGS, f"--save_freq={steps}"],
    )
    elapsed = time.time() - t0
    print(f"[python] training done in {elapsed:.1f}s ({elapsed / steps:.2f}s/step)")

    last_step_dir = _find_last_step_dir(output_dir / "checkpoints")
    return last_step_dir / "pretrained_model"


def run_elixir_training(
    checkpoint: str,
    train_dataset_root: Path,
    output_dir: Path,
    *,
    steps: int,
    batch_size: int,
) -> Path:
    """Fine-tunes via the Elixir-native trainer (component 01.4) against
    the SAME train-only dataset copy, via a real `mix run -e` subprocess
    (this repo's own established pattern for driving a real, separate
    training entry point -- mirrors the Python trainer's own subprocess
    invocation of `lerobot-train`). Returns the resulting step-checkpoint
    dir (loadable via `SmolVLA.load/1`)."""
    print(f"[elixir] training {steps} steps, batch_size={batch_size} ...")
    t0 = time.time()

    elixir_code = f"""
    job = FineTuneJob.run(
      {elixir_string(str(checkpoint))},
      %FineTuneJob.Episodes{{root: {elixir_string(str(train_dataset_root))}}},
      {elixir_string(str(output_dir))},
      steps: {steps},
      batch_size: {batch_size},
      save_every: {steps},
      seed: 0
    )
    IO.puts("elixir training finished at step #{{job.step}}")
    """

    result = subprocess.run(
        ["mix", "run", "-e", elixir_code],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    elapsed = time.time() - t0
    print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(
            f"elixir training subprocess exited {result.returncode}:\n{result.stdout}"
        )
    print(f"[elixir] training done in {elapsed:.1f}s ({elapsed / steps:.2f}s/step)")

    return _find_last_step_dir(output_dir / "checkpoints")


def elixir_string(value: str) -> str:
    """Elixir string literal quoting for the small `mix run -e` snippets
    this module generates -- escapes backslashes/quotes, nothing fancier
    needed since every value passed through here is a real filesystem
    path this same process just constructed (no untrusted input)."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def evaluate_python_policy(
    pretrained_dir: Path,
    dataset_root: Path,
    holdout_episodes: list[int],
    *,
    n_frames_per_episode: int,
    n_throughput_calls: int,
) -> tuple[AccuracyProxyResult, ThroughputResult]:
    """Evaluates the Python-trained policy through the already-accepted
    Python ``infer_action()`` (component 01.1) against the real held-out
    episode frames, read via LeRobot's own real dataset reader.

    ``LeRobotDataset(..., episodes=holdout_episodes)`` re-indexes
    ``__getitem__`` to a LOCAL, 0-based position across only the selected
    episodes (confirmed directly: indexing at a held-out episode's own
    GLOBAL ``dataset_from_index`` raises "out of bounds" once the dataset
    is episode-filtered, since the filtered view is shorter than the full
    dataset) -- so this locates each held-out episode's real frame
    positions via the filtered dataset's own ``hf_dataset['episode_index']``
    column rather than the metadata table's global index range, which
    only applies to the UNFILTERED dataset.
    """
    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from mlx_vlm.models import smolvla

    print(
        f"[python] evaluating {pretrained_dir} against holdout episodes {holdout_episodes}"
    )
    model = smolvla.SmolVLAModel.from_pretrained(str(pretrained_dir))

    ds = LeRobotDataset(
        DEFAULT_DATASET_REPO_ID,
        root=str(dataset_root),
        episodes=holdout_episodes,
        video_backend="pyav",
    )

    local_episode_index = np.array(ds.hf_dataset["episode_index"])

    episode_predictions: dict[int, list] = {}
    for episode_index in holdout_episodes:
        local_positions = np.where(local_episode_index == episode_index)[0]
        total = len(local_positions)
        n = min(n_frames_per_episode, total)
        step = total / n
        chosen = sorted({min(total - 1, int(i * step)) for i in range(n)})

        pairs = []
        for local_idx in chosen:
            item = ds[int(local_positions[local_idx])]
            image = item["observation.images.side"].numpy()  # (3, H, W), [0, 1]
            state = item["observation.state"].numpy()
            action = item["action"].numpy()
            task = item["task"]

            chunk = model.infer_action(image, state, task)
            episode_predictions.setdefault(episode_index, []).append(
                (np.asarray(chunk), action)
            )
            pairs.append(1)
        print(
            f"[python]   episode {episode_index}: evaluated {len(pairs)}/{total} frames"
        )

    # Throughput: warm, steady-state repeated calls on a fixed frame.
    first_episode = holdout_episodes[0]
    first_local_positions = np.where(local_episode_index == first_episode)[0]
    warm_item = ds[int(first_local_positions[0])]
    warm_image = warm_item["observation.images.side"].numpy()
    warm_state = warm_item["observation.state"].numpy()
    warm_task = warm_item["task"]

    throughput = measure_throughput(
        lambda: model.infer_action(warm_image, warm_state, warm_task),
        n_calls=n_throughput_calls,
        warmup_calls=1,
    )
    print(
        f"[python] throughput: {throughput.actions_per_second:.4f} actions/sec "
        f"({throughput.seconds_per_action:.4f}s/action)"
    )

    accuracy = compute_accuracy_proxy(episode_predictions)
    return accuracy, throughput


def evaluate_elixir_policy(
    checkpoint_dir: Path,
    dataset_root: Path,
    holdout_episodes: list[int],
    *,
    n_frames_per_episode: int,
    n_throughput_calls: int,
    output_json_path: Path,
) -> tuple[AccuracyProxyResult, ThroughputResult]:
    """Evaluates the Elixir-trained policy through the already-accepted
    Elixir-native ``infer_action/4`` (component 01.2), via
    ``eval_elixir_policy.exs`` -- a real `mix run` subprocess so the
    evaluation runs through the real `emily`/Nx.Backend path, not
    reimplemented in Python."""
    print(
        f"[elixir] evaluating {checkpoint_dir} against holdout episodes {holdout_episodes}"
    )
    holdout_csv = ",".join(str(e) for e in holdout_episodes)

    result = subprocess.run(
        [
            "mix",
            "run",
            "finetune_job/parity_gate/eval_elixir_policy.exs",
            "--",
            str(checkpoint_dir),
            str(dataset_root),
            holdout_csv,
            str(n_frames_per_episode),
            str(n_throughput_calls),
            str(output_json_path),
        ],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(
            f"elixir evaluation subprocess exited {result.returncode}:\n{result.stdout}"
        )

    payload = json.loads(output_json_path.read_text())
    episode_predictions = {
        int(ep): [(chunk, gt) for chunk, gt in pairs]
        for ep, pairs in payload["episode_predictions"].items()
    }
    accuracy = compute_accuracy_proxy(episode_predictions)

    throughput_raw = payload["throughput"]
    total_seconds = throughput_raw["total_seconds"]
    n_calls = throughput_raw["n_calls"]
    throughput = ThroughputResult(
        actions_per_second=n_calls / total_seconds
        if total_seconds > 0
        else float("inf"),
        seconds_per_action=total_seconds / n_calls,
        n_calls=n_calls,
        total_seconds=total_seconds,
    )
    print(
        f"[elixir] throughput: {throughput.actions_per_second:.4f} actions/sec "
        f"({throughput.seconds_per_action:.4f}s/action)"
    )

    return accuracy, throughput


def main() -> int:
    if os.environ.get("RUN_PARITY_GATE") != "1":
        print(
            "Set RUN_PARITY_GATE=1 to run the real cutover-gate comparison "
            "(real fine-tuning runs with both trainers, real wall-clock minutes). See "
            "finetune_job/parity_gate/run_gate.py's module docstring."
        )
        return 0

    checkpoint = _env_str("SMOLVLA_CHECKPOINT", DEFAULT_CHECKPOINT)
    elixir_checkpoint = Path(
        _env_str("SMOLVLA_ELIXIR_CHECKPOINT", DEFAULT_ELIXIR_CHECKPOINT)
    ).expanduser()
    dataset_root = Path(
        _env_str("FINETUNE_DATASET_ROOT", DEFAULT_DATASET_ROOT)
    ).expanduser()
    n_holdout = _env_int("PARITY_GATE_N_HOLDOUT", DEFAULT_N_HOLDOUT)
    steps = _env_int("PARITY_GATE_STEPS", DEFAULT_STEPS)
    batch_size = _env_int("PARITY_GATE_BATCH_SIZE", DEFAULT_BATCH_SIZE)
    n_frames_per_episode = _env_int(
        "PARITY_GATE_N_EVAL_FRAMES_PER_EPISODE", DEFAULT_N_EVAL_FRAMES_PER_EPISODE
    )
    n_throughput_calls = _env_int(
        "PARITY_GATE_N_THROUGHPUT_CALLS", DEFAULT_N_THROUGHPUT_CALLS
    )

    work_dir = Path(
        _env_str("PARITY_GATE_WORK_DIR", str(REPO_ROOT / ".parity_gate_run"))
    ).expanduser()
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Parity gate (issue 08) real run -- work_dir={work_dir} ===")
    print(f"checkpoint={checkpoint} elixir_checkpoint={elixir_checkpoint}")
    print(f"dataset_root={dataset_root}")
    print(
        f"n_holdout={n_holdout} steps={steps} batch_size={batch_size} "
        f"n_frames_per_episode={n_frames_per_episode} n_throughput_calls={n_throughput_calls}"
    )

    info = json.loads((dataset_root / "meta" / "info.json").read_text())
    total_episodes = info["total_episodes"]
    split = split_episodes(total_episodes, n_holdout=n_holdout)
    print(
        f"split: {len(split.train_episodes)} train, {len(split.holdout_episodes)} holdout"
    )
    print(f"  train_episodes={list(split.train_episodes)}")
    print(f"  holdout_episodes={list(split.holdout_episodes)}")

    train_dataset_root = work_dir / "train_only_dataset"
    if train_dataset_root.exists():
        shutil.rmtree(train_dataset_root)
    build_train_only_dataset(dataset_root, train_dataset_root, split)
    print(f"train-only dataset materialized at {train_dataset_root}")

    python_output_dir = work_dir / "python_finetune_output"
    if python_output_dir.exists():
        shutil.rmtree(python_output_dir)
    python_pretrained_dir = run_python_training(
        checkpoint,
        train_dataset_root,
        python_output_dir,
        steps=steps,
        batch_size=batch_size,
    )

    elixir_output_dir = work_dir / "elixir_finetune_output"
    if elixir_output_dir.exists():
        shutil.rmtree(elixir_output_dir)
    elixir_checkpoint_dir = run_elixir_training(
        elixir_checkpoint,
        train_dataset_root,
        elixir_output_dir,
        steps=steps,
        batch_size=batch_size,
    )

    python_accuracy, python_throughput = evaluate_python_policy(
        python_pretrained_dir,
        dataset_root,
        list(split.holdout_episodes),
        n_frames_per_episode=n_frames_per_episode,
        n_throughput_calls=n_throughput_calls,
    )

    elixir_eval_json = work_dir / "elixir_eval_result.json"
    elixir_accuracy, elixir_throughput = evaluate_elixir_policy(
        elixir_checkpoint_dir,
        dataset_root,
        list(split.holdout_episodes),
        n_frames_per_episode=n_frames_per_episode,
        n_throughput_calls=n_throughput_calls,
        output_json_path=elixir_eval_json,
    )

    judgment = judge_cutover(
        python_accuracy, elixir_accuracy, python_throughput, elixir_throughput
    )

    print("\n" + "=" * 70)
    print(judgment.summary())
    print("=" * 70)

    report = {
        "run_config": {
            "checkpoint": checkpoint,
            "elixir_checkpoint": str(elixir_checkpoint),
            "dataset_root": str(dataset_root),
            "n_holdout": n_holdout,
            "steps": steps,
            "batch_size": batch_size,
            "n_frames_per_episode": n_frames_per_episode,
            "n_throughput_calls": n_throughput_calls,
        },
        "split": {
            "train_episodes": list(split.train_episodes),
            "holdout_episodes": list(split.holdout_episodes),
        },
        "python": {
            "accuracy": {
                "mean_absolute_error": python_accuracy.mean_absolute_error,
                "max_absolute_error": python_accuracy.max_absolute_error,
                "per_episode_mean_absolute_error": python_accuracy.per_episode_mean_absolute_error,
                "n_frames_evaluated": python_accuracy.n_frames_evaluated,
            },
            "throughput": {
                "actions_per_second": python_throughput.actions_per_second,
                "seconds_per_action": python_throughput.seconds_per_action,
                "n_calls": python_throughput.n_calls,
                "total_seconds": python_throughput.total_seconds,
            },
        },
        "elixir": {
            "accuracy": {
                "mean_absolute_error": elixir_accuracy.mean_absolute_error,
                "max_absolute_error": elixir_accuracy.max_absolute_error,
                "per_episode_mean_absolute_error": elixir_accuracy.per_episode_mean_absolute_error,
                "n_frames_evaluated": elixir_accuracy.n_frames_evaluated,
            },
            "throughput": {
                "actions_per_second": elixir_throughput.actions_per_second,
                "seconds_per_action": elixir_throughput.seconds_per_action,
                "n_calls": elixir_throughput.n_calls,
                "total_seconds": elixir_throughput.total_seconds,
            },
        },
        "threshold": {
            "max_accuracy_regression_fraction": judgment.threshold.max_accuracy_regression_fraction,
            "min_throughput_fraction": judgment.threshold.min_throughput_fraction,
        },
        "accuracy_regression_fraction": judgment.accuracy_regression_fraction,
        "throughput_fraction": judgment.throughput_fraction,
        "accuracy_passes": judgment.accuracy_passes,
        "throughput_passes": judgment.throughput_passes,
        "passes": judgment.passes,
        "summary": judgment.summary(),
    }

    report_path = work_dir / "parity_gate_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport written to {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
