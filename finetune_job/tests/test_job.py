"""Fast, mocked-runner test suite for FineTuneJob.

Per the work-order's TDD directive: a real fine-tuning run against a real
~450M-parameter checkpoint is slow (minutes) and needs real episode data --
out of scope for this fast suite. Every test here injects a fake ``runner``
(mirrors ``InferActionServer``'s model-injection pattern) that records the
constructed ``lerobot-train`` argv instead of actually invoking it, so these
tests assert this module's OWN logic -- config/argv construction, the
frozen-vs-full-finetune flag wiring, resume-path validation, metadata
consistency, and the provenance-non-branching property -- at unit-test
speed. The real, opt-in, multi-minute integration run lives in
finetune_job/tests/integration/.
"""

import json
import subprocess

import pytest

from finetune_job.job import (
    CorruptCheckpointError,
    Episodes,
    FineTuneJob,
    FineTuneRunError,
    validate_checkpoint,
)


class FakeRunner:
    """Records every invocation's argv and returns a canned
    ``subprocess.CompletedProcess``-shaped result instead of running
    anything real."""

    def __init__(self, returncode=0, output="ok"):
        self.calls = []
        self.returncode = returncode
        self.output = output

    def __call__(self, args):
        self.calls.append(list(args))
        return subprocess.CompletedProcess(args, self.returncode, stdout=self.output)


class TestRunConstructsRealCLIInvocation:
    """FineTuneJob.run() must build the exact CLI shape LeRobot's own
    documented fine-tuning example uses (docs/source/smolvla.mdx,
    --policy.path=... for fine-tune-from-pretrained), plus this design's
    frozen-backbone-by-default fields."""

    def test_default_run_uses_frozen_backbone_action_expert_only(self, tmp_path):
        runner = FakeRunner()
        output_path = tmp_path / "out"

        FineTuneJob.run(
            checkpoint_path="lerobot/smolvla_base",
            episodes=Episodes(repo_id="local/demo", root=str(tmp_path / "dataset")),
            output_path=str(output_path),
            runner=runner,
        )

        assert len(runner.calls) == 1
        args = runner.calls[0]
        assert "--policy.path=lerobot/smolvla_base" in args
        assert "--policy.train_expert_only=true" in args
        assert "--policy.freeze_vision_encoder=true" in args
        assert "--policy.load_vlm_weights=true" in args
        assert "--dataset.repo_id=local/demo" in args
        assert f"--dataset.root={tmp_path / 'dataset'}" in args
        assert f"--output_dir={output_path}" in args

    def test_full_finetune_flag_switches_train_expert_only_off(self, tmp_path):
        runner = FakeRunner()
        FineTuneJob.run(
            checkpoint_path="lerobot/smolvla_base",
            episodes=Episodes(repo_id="local/demo"),
            output_path=str(tmp_path / "out"),
            full_finetune=True,
            runner=runner,
        )

        args = runner.calls[0]
        assert "--policy.train_expert_only=false" in args
        assert "--policy.freeze_vision_encoder=false" in args
        # Still initialized from the pretrained checkpoint's weights --
        # full_finetune switches WHICH params train, not the init source.
        assert "--policy.load_vlm_weights=true" in args

    def test_hub_only_dataset_omits_root_flag(self, tmp_path):
        """episodes without a local root (Hub-only) must not emit
        --dataset.root at all -- LeRobotDataset resolves repo_id against the
        Hub cache when root is unset."""
        runner = FakeRunner()
        FineTuneJob.run(
            checkpoint_path="lerobot/smolvla_base",
            episodes=Episodes(repo_id="lerobot/svla_so101_pickplace"),
            output_path=str(tmp_path / "out"),
            runner=runner,
        )

        args = runner.calls[0]
        assert not any(a.startswith("--dataset.root=") for a in args)
        assert "--dataset.repo_id=lerobot/svla_so101_pickplace" in args

    def test_steps_and_batch_size_are_configurable(self, tmp_path):
        """A fast smoke-test run needs a very low --steps; must be wired
        through, not hardcoded to a full 20000-step run."""
        runner = FakeRunner()
        FineTuneJob.run(
            checkpoint_path="lerobot/smolvla_base",
            episodes=Episodes(repo_id="local/demo", root=str(tmp_path / "d")),
            output_path=str(tmp_path / "out"),
            steps=2,
            batch_size=1,
            runner=runner,
        )

        args = runner.calls[0]
        assert "--steps=2" in args
        assert "--batch_size=1" in args


class TestRunIdentityAndMetadata:
    """component 01.3's interface note: 'identity persists across the run'
    -- run() returns a FineTuneJob usable to check status/resume later, and
    the mode (frozen vs. full) is never silently inconsistent with the
    checkpoint it produced."""

    def test_run_returns_a_finetune_job_with_stable_identity(self, tmp_path):
        job = FineTuneJob.run(
            checkpoint_path="lerobot/smolvla_base",
            episodes=Episodes(repo_id="local/demo", root=str(tmp_path / "d")),
            output_path=str(tmp_path / "out"),
            runner=FakeRunner(),
        )

        assert isinstance(job, FineTuneJob)
        assert job.run_id
        assert job.output_path == tmp_path / "out"

    def test_run_writes_retrievable_mode_metadata_alongside_output(self, tmp_path):
        output_path = tmp_path / "out"
        job = FineTuneJob.run(
            checkpoint_path="lerobot/smolvla_base",
            episodes=Episodes(repo_id="local/demo", root=str(tmp_path / "d")),
            output_path=str(output_path),
            full_finetune=True,
            runner=FakeRunner(),
        )

        meta_path = output_path / "finetune_job_meta.json"
        assert meta_path.is_file()
        meta = json.loads(meta_path.read_text())
        assert meta["full_finetune"] is True
        assert meta["run_id"] == job.run_id

    def test_default_mode_metadata_records_frozen_backbone(self, tmp_path):
        output_path = tmp_path / "out"
        FineTuneJob.run(
            checkpoint_path="lerobot/smolvla_base",
            episodes=Episodes(repo_id="local/demo", root=str(tmp_path / "d")),
            output_path=str(output_path),
            runner=FakeRunner(),
        )

        meta = json.loads((output_path / "finetune_job_meta.json").read_text())
        assert meta["full_finetune"] is False


class TestRunFailsLoudOnSubprocessFailure:
    def test_nonzero_exit_raises_finetune_run_error(self, tmp_path):
        runner = FakeRunner(returncode=1, output="boom: dataset not found")

        with pytest.raises(FineTuneRunError) as excinfo:
            FineTuneJob.run(
                checkpoint_path="lerobot/smolvla_base",
                episodes=Episodes(repo_id="local/demo", root=str(tmp_path / "d")),
                output_path=str(tmp_path / "out"),
                runner=runner,
            )

        assert excinfo.value.returncode == 1
        assert "boom" in excinfo.value.output

    def test_failed_run_does_not_write_metadata(self, tmp_path):
        """A failed subprocess must not leave a metadata file implying a
        run completed -- metadata presence should mean the run actually
        finished the invocation, not just started it."""
        output_path = tmp_path / "out"
        runner = FakeRunner(returncode=1, output="boom")

        with pytest.raises(FineTuneRunError):
            FineTuneJob.run(
                checkpoint_path="lerobot/smolvla_base",
                episodes=Episodes(repo_id="local/demo", root=str(tmp_path / "d")),
                output_path=str(output_path),
                runner=runner,
            )

        assert not (output_path / "finetune_job_meta.json").is_file()


class TestProvenanceNeverBranches:
    """CONTEXT term 'Episode': provenance (real robot vs. simulation) never
    changes an episode's shape or how a FineTuneJob adapter consumes it.
    Proven by code inspection (Episodes carries no provenance field at
    all -- see finetune_job/job.py) plus this behavioral check: two
    Episodes values that differ only in an arbitrary "origin-flavored"
    repo_id/root naming produce IDENTICAL argv shapes (same flags, same
    structure) -- nothing about the invocation depends on what the
    dataset is named or tagged."""

    def test_real_and_simulated_flavored_episodes_produce_identical_argv_shape(
        self, tmp_path
    ):
        real_runner = FakeRunner()
        sim_runner = FakeRunner()

        real_root = tmp_path / "real_robot_dataset"
        sim_root = tmp_path / "sim_dataset"

        FineTuneJob.run(
            checkpoint_path="lerobot/smolvla_base",
            episodes=Episodes(repo_id="local/real-robot-run", root=str(real_root)),
            output_path=str(tmp_path / "out_real"),
            runner=real_runner,
        )
        FineTuneJob.run(
            checkpoint_path="lerobot/smolvla_base",
            episodes=Episodes(repo_id="local/sim-run", root=str(sim_root)),
            output_path=str(tmp_path / "out_sim"),
            runner=sim_runner,
        )

        def normalize(args, root, output_path):
            # Strip the two args that legitimately differ (dataset
            # identity, output path) -- every other flag, and the flag
            # SET itself, must be identical.
            out = []
            for a in args:
                if (
                    a.startswith("--dataset.repo_id=")
                    or a.startswith("--dataset.root=")
                    or a.startswith("--output_dir=")
                    or a.startswith("--job_name=")
                ):
                    continue
                out.append(a)
            return out

        real_args = normalize(real_runner.calls[0], real_root, tmp_path / "out_real")
        sim_args = normalize(sim_runner.calls[0], sim_root, tmp_path / "out_sim")
        assert real_args == sim_args


class TestValidateCheckpoint:
    """component 01.3's Fails requirement: a corrupt checkpoint is detected
    (checksum or shape-validated) rather than silently continued from."""

    def test_valid_checkpoint_passes(self, make_checkpoint):
        checkpoint_dir = make_checkpoint()
        validate_checkpoint(checkpoint_dir)  # must not raise

    def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(CorruptCheckpointError):
            validate_checkpoint(tmp_path / "does-not-exist")

    def test_missing_model_safetensors_raises(self, make_checkpoint):
        checkpoint_dir = make_checkpoint()
        (checkpoint_dir / "pretrained_model" / "model.safetensors").unlink()

        with pytest.raises(CorruptCheckpointError, match="missing required file"):
            validate_checkpoint(checkpoint_dir)

    def test_missing_training_state_raises(self, make_checkpoint):
        checkpoint_dir = make_checkpoint()
        (checkpoint_dir / "training_state" / "training_step.json").unlink()

        with pytest.raises(CorruptCheckpointError, match="missing required file"):
            validate_checkpoint(checkpoint_dir)

    def test_truncated_safetensors_file_raises(self, make_checkpoint):
        checkpoint_dir = make_checkpoint()
        weights_path = checkpoint_dir / "pretrained_model" / "model.safetensors"
        original = weights_path.read_bytes()
        # Truncate mid-header/mid-data -- a realistic "corrupted during
        # write/transfer" failure mode.
        weights_path.write_bytes(original[: len(original) // 2])

        with pytest.raises(CorruptCheckpointError):
            validate_checkpoint(checkpoint_dir)

    def test_zero_byte_safetensors_file_raises(self, make_checkpoint):
        checkpoint_dir = make_checkpoint()
        weights_path = checkpoint_dir / "pretrained_model" / "model.safetensors"
        weights_path.write_bytes(b"")

        with pytest.raises(CorruptCheckpointError):
            validate_checkpoint(checkpoint_dir)

    def test_malformed_training_step_json_raises(self, make_checkpoint):
        checkpoint_dir = make_checkpoint()
        (checkpoint_dir / "training_state" / "training_step.json").write_text(
            "{not valid json"
        )

        with pytest.raises(CorruptCheckpointError):
            validate_checkpoint(checkpoint_dir)

    def test_training_step_missing_step_field_raises(self, make_checkpoint):
        checkpoint_dir = make_checkpoint()
        (checkpoint_dir / "training_state" / "training_step.json").write_text(
            json.dumps({"not_step": 1})
        )

        with pytest.raises(CorruptCheckpointError):
            validate_checkpoint(checkpoint_dir)


class TestResume:
    """component 01.3's Fails requirement: a job interrupted mid-run
    resumes from its last checkpoint, never silently restarts from scratch
    nor silently continues from a corrupt checkpoint."""

    def test_resume_validates_before_invoking_lerobot_train(self, make_checkpoint):
        checkpoint_dir = make_checkpoint()
        (checkpoint_dir / "pretrained_model" / "model.safetensors").unlink()
        runner = FakeRunner()

        with pytest.raises(CorruptCheckpointError):
            FineTuneJob.resume(str(checkpoint_dir), runner=runner)

        # Corrupt checkpoint must never reach the subprocess call.
        assert runner.calls == []

    def test_resume_invokes_real_resume_flags(self, make_checkpoint):
        checkpoint_dir = make_checkpoint()
        runner = FakeRunner()

        FineTuneJob.resume(str(checkpoint_dir), runner=runner)

        assert len(runner.calls) == 1
        args = runner.calls[0]
        assert "--resume=true" in args
        config_path = checkpoint_dir / "pretrained_model" / "train_config.json"
        assert f"--config_path={config_path}" in args

    def test_resume_recovers_mode_from_lerobot_train_config_without_sidecar(
        self, make_checkpoint
    ):
        """Even if this module's own metadata sidecar is absent (e.g. a
        checkpoint produced by a run this process didn't launch),
        LeRobot's own train_config.json already redundantly records
        train_expert_only -- resume must recover the correct mode from it
        alone."""
        checkpoint_dir = make_checkpoint(train_expert_only=False)
        runner = FakeRunner()

        job = FineTuneJob.resume(str(checkpoint_dir), runner=runner)

        assert job.full_finetune is True

    def test_resume_recovers_frozen_backbone_mode(self, make_checkpoint):
        checkpoint_dir = make_checkpoint(train_expert_only=True)
        runner = FakeRunner()

        job = FineTuneJob.resume(str(checkpoint_dir), runner=runner)

        assert job.full_finetune is False

    def test_resume_returns_finetune_job_and_rewrites_metadata(self, make_checkpoint):
        checkpoint_dir = make_checkpoint()
        runner = FakeRunner()

        job = FineTuneJob.resume(str(checkpoint_dir), runner=runner)

        assert isinstance(job, FineTuneJob)
        assert (job.output_path / "finetune_job_meta.json").is_file()

    def test_resume_fails_loud_on_subprocess_failure(self, make_checkpoint):
        checkpoint_dir = make_checkpoint()
        runner = FakeRunner(
            returncode=1, output="resume failed: optimizer state mismatch"
        )

        with pytest.raises(FineTuneRunError):
            FineTuneJob.resume(str(checkpoint_dir), runner=runner)


class TestEpisodesValueObject:
    """CONTEXT term 'Episode': a value object; Episodes carries no identity
    and no provenance field."""

    def test_episodes_has_no_provenance_field(self):
        fields = {f for f in Episodes.__dataclass_fields__}
        assert fields == {"repo_id", "root"}

    def test_episodes_is_a_frozen_value_object(self):
        episodes = Episodes(repo_id="local/demo", root="/tmp/x")
        with pytest.raises(Exception):
            episodes.repo_id = "changed"  # frozen dataclass -> raises
