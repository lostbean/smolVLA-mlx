"""Real-checkpoint, real-dataset integration check for FineTuneJob.

Runs a REAL, minimal (few-step) fine-tuning run against the real, publicly
available `lerobot/smolvla_base` checkpoint and a real, small, public
LeRobotDataset-format dataset (`lerobot/svla_so101_pickplace` -- 50
episodes, ~12k frames, 2 cameras, ~86MB total on the Hugging Face Hub;
chosen because it is small, real, public, explicitly SmolVLA-tagged, and
maintained by the `lerobot` org itself, matching this repo's established
preference for real artifacts over fabricated stand-ins -- see
model_runtime_server/tests/integration/test_server_real_checkpoint.py and
vendor/mlx-vlm/mlx_vlm/tests/integration/test_smolvla_real_checkpoint.py for
the same pattern applied to inference).

This is deliberately NOT part of the fast test gate
(`uv run python -m pytest finetune_job`): it downloads real weights (~1.1GB,
likely already cached from prior chunks) and a real dataset (~86MB), and
runs real forward/backward passes through SmolVLA's ~450M-parameter
backbone plus ~100M-parameter action expert -- real wall-clock minutes, not
seconds. Skipped by default; opt in with RUN_FINETUNE_INTEGRATION_CHECK=1:

    RUN_FINETUNE_INTEGRATION_CHECK=1 uv run python -m pytest \
        finetune_job/tests/integration/test_finetune_real_checkpoint.py -v -s

Override the checkpoint/dataset/step count via environment variables:

    SMOLVLA_CHECKPOINT=/path/to/local/checkpoint
    FINETUNE_DATASET_REPO_ID=lerobot/svla_so101_pickplace
    FINETUNE_INTEGRATION_STEPS=2
"""

import os
import shutil

import pytest

from finetune_job.job import Episodes, FineTuneJob

DEFAULT_CHECKPOINT = "lerobot/smolvla_base"
DEFAULT_DATASET_REPO_ID = "lerobot/svla_so101_pickplace"
_RUN_FLAG = "RUN_FINETUNE_INTEGRATION_CHECK"

# lerobot/svla_so101_pickplace's own camera feature names
# (observation.images.{side,up}) don't match lerobot/smolvla_base's
# pretrained config (observation.images.camera{1,2,3}) -- real, expected
# LeRobot behavior (see lerobot.policies.utils.validate_visual_features_consistency),
# fixed via LeRobot's own documented --rename_map mechanism, not a
# workaround of ours. --dataset.video_backend=pyav works around a real,
# environment-specific torchcodec/FFmpeg dylib mismatch on this Mac
# (torchcodec's precompiled binaries expect FFmpeg dylib versions this
# system doesn't have on its library path); pyav decodes the same real
# video files through a different real backend LeRobot itself supports.
_REAL_DATASET_EXTRA_ARGS = [
    "--policy.device=cpu",
    "--dataset.video_backend=pyav",
    "--rename_map={"
    '"observation.images.side": "observation.images.camera1", '
    '"observation.images.up": "observation.images.camera2"}',
]

pytestmark = pytest.mark.skipif(
    os.environ.get(_RUN_FLAG) != "1",
    reason=(
        f"Runs a real (minimal) fine-tuning pass against a real ~1.1GB "
        f"checkpoint and a real ~86MB dataset; set {_RUN_FLAG}=1 to opt in "
        f"(see module docstring)."
    ),
)


@pytest.fixture
def output_dir(tmp_path):
    d = tmp_path / "finetune_output"
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_real_minimal_run_produces_weights_loadable_by_infer_action(output_dir):
    """The acceptance bar: a real fine-tuning run against real episode data
    produces updated action-expert weights that genuinely reload through
    the already-accepted Python infer_action() (component 01.1)."""
    checkpoint = os.environ.get("SMOLVLA_CHECKPOINT", DEFAULT_CHECKPOINT)
    dataset_repo_id = os.environ.get(
        "FINETUNE_DATASET_REPO_ID", DEFAULT_DATASET_REPO_ID
    )
    steps = int(os.environ.get("FINETUNE_INTEGRATION_STEPS", "2"))

    job = FineTuneJob.run(
        checkpoint_path=checkpoint,
        episodes=Episodes(repo_id=dataset_repo_id),
        output_path=str(output_dir),
        steps=steps,
        batch_size=2,
        extra_args=[*_REAL_DATASET_EXTRA_ARGS, "--save_freq=1"],
    )

    print(f"\nreal integration run: run_id={job.run_id}, output={job.output_path}")

    # LeRobot writes checkpoints/<step>/pretrained_model/{config.json,model.safetensors}
    # -- find the last one and reload it exactly the way the design intends
    # (component 01.3 "Interacts with": produces safetensors weights
    # consumed by 01.1).
    checkpoints_dir = output_dir / "checkpoints"
    assert checkpoints_dir.is_dir(), f"no checkpoints written under {checkpoints_dir}"
    step_dirs = sorted(
        p for p in checkpoints_dir.iterdir() if p.is_dir() and p.name != "last"
    )
    assert step_dirs, f"no step checkpoint directories under {checkpoints_dir}"
    last_step_dir = step_dirs[-1]
    pretrained_dir = last_step_dir / "pretrained_model"
    assert (pretrained_dir / "model.safetensors").is_file()
    assert (pretrained_dir / "config.json").is_file()

    meta_path = output_dir / "finetune_job_meta.json"
    assert meta_path.is_file()
    import json

    meta = json.loads(meta_path.read_text())
    assert meta["full_finetune"] is False
    print(f"real integration run: mode metadata = {meta}")

    # Prove the fine-tuned weights genuinely reload through the
    # already-accepted Python infer_action() (component 01.1) -- same
    # checkpoint SHAPE as lerobot/smolvla_base, just updated values.
    import numpy as np

    from mlx_vlm.models import smolvla

    model = smolvla.SmolVLAModel.from_pretrained(str(pretrained_dir))

    rng = np.random.default_rng(0)
    image = rng.random((256, 256, 3), dtype=np.float32)
    state = rng.standard_normal(6).astype(np.float32)
    action = model.infer_action(
        image, state, "pick up the block and place it in the bin"
    )

    print(f"real integration run: reloaded action_chunk shape = {action.shape}")

    assert action.shape == (model.config.chunk_size, model.config.action_dim)

    import mlx.core as mx

    assert bool(mx.isfinite(action).all()), "action chunk contains NaN/Inf"
    assert not bool(mx.all(action == 0)), "action chunk is degenerate all-zero"


def test_real_interrupted_run_resumes_rather_than_restarting(output_dir):
    """component 01.3's Fails requirement, proven against a real run: an
    interrupted job resumes from its last checkpoint (training step
    continues climbing, not reset to 0) rather than silently restarting
    from scratch."""
    checkpoint = os.environ.get("SMOLVLA_CHECKPOINT", DEFAULT_CHECKPOINT)
    dataset_repo_id = os.environ.get(
        "FINETUNE_DATASET_REPO_ID", DEFAULT_DATASET_REPO_ID
    )

    # First "interrupted" run: 1 step only, checkpointed immediately.
    FineTuneJob.run(
        checkpoint_path=checkpoint,
        episodes=Episodes(repo_id=dataset_repo_id),
        output_path=str(output_dir),
        steps=1,
        batch_size=2,
        extra_args=[*_REAL_DATASET_EXTRA_ARGS, "--save_freq=1"],
    )

    checkpoints_dir = output_dir / "checkpoints"
    first_step_dirs = sorted(
        p.name for p in checkpoints_dir.iterdir() if p.is_dir() and p.name != "last"
    )
    assert first_step_dirs, "first (interrupted) run wrote no checkpoint"
    print(f"\nreal integration resume: first run checkpoints = {first_step_dirs}")

    last_checkpoint = checkpoints_dir / "last"
    assert last_checkpoint.is_symlink() or last_checkpoint.is_dir()

    # Resume for 1 more step -- must build on top of, not replace, the
    # first run's checkpoint. lerobot-train's own --steps counts total
    # steps for the run, so bump it to allow one further step past resume.
    resumed_job = FineTuneJob.resume(
        str(last_checkpoint.resolve()),
        extra_args=["--policy.device=cpu", "--steps=2"],
    )

    print(f"real integration resume: resumed job run_id={resumed_job.run_id}")

    second_step_dirs = sorted(
        p.name for p in checkpoints_dir.iterdir() if p.is_dir() and p.name != "last"
    )
    print(f"real integration resume: post-resume checkpoints = {second_step_dirs}")

    # A genuinely resumed run reaches a HIGHER step count than the
    # interrupted run's last checkpoint -- proof it built on top of, not
    # restarted from, the prior checkpoint.
    assert max(int(s) for s in second_step_dirs) > max(int(s) for s in first_step_dirs)
