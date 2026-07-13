"""Shared fixtures for finetune_job's fast test suite: builders for
structurally-real-but-tiny LeRobot checkpoint directories, so tests can
exercise ``validate_checkpoint`` and ``FineTuneJob.resume``'s metadata
handling against real file shapes (real safetensors headers, real JSON)
without a real multi-minute training run or real ~450M-parameter weights.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file


def _write_pretrained_model(
    pretrained_dir: Path, *, train_expert_only: bool, dataset_repo_id: str, dataset_root
):
    pretrained_dir.mkdir(parents=True, exist_ok=True)
    (pretrained_dir / "config.json").write_text(json.dumps({"type": "smolvla"}))
    save_file(
        {
            "vlm_with_expert.lm_expert.layers.0.weight": np.zeros(
                (4, 4), dtype=np.float32
            )
        },
        str(pretrained_dir / "model.safetensors"),
    )
    train_config = {
        "policy": {"type": "smolvla", "train_expert_only": train_expert_only},
        "dataset": {"repo_id": dataset_repo_id, "root": dataset_root},
        "steps": 5,
    }
    (pretrained_dir / "train_config.json").write_text(json.dumps(train_config))


def _write_training_state(training_state_dir: Path, *, step: int):
    training_state_dir.mkdir(parents=True, exist_ok=True)
    (training_state_dir / "training_step.json").write_text(json.dumps({"step": step}))


@pytest.fixture
def make_checkpoint(tmp_path):
    """Builds a real-shaped LeRobot step-checkpoint dir:
    ``<root>/checkpoints/<step>/{pretrained_model/, training_state/}``.
    Returns the step-checkpoint dir path.
    """

    def _make(
        *,
        root: Path = None,
        step: int = 5,
        train_expert_only: bool = True,
        dataset_repo_id: str = "local/demo",
        dataset_root: str = None,
    ) -> Path:
        base = root if root is not None else tmp_path / "run_output"
        step_dir = base / "checkpoints" / f"{step:06d}"
        _write_pretrained_model(
            step_dir / "pretrained_model",
            train_expert_only=train_expert_only,
            dataset_repo_id=dataset_repo_id,
            dataset_root=dataset_root,
        )
        _write_training_state(step_dir / "training_state", step=step)
        return step_dir

    return _make
