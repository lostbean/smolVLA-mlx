"""FineTuneJob: drives LeRobot's own real ``lerobot-train`` entry point to
fine-tune SmolVLA's action expert.

Per docs/design/model-runtime/design.md component 01.3 and the model-runtime
foundation's "Not a second training framework" no-goal: this module writes
no training loop, no dataset loader, and no checkpoint format of its own --
it constructs a real ``lerobot-train`` CLI invocation (LeRobot's own
documented, ``@parser.wrap()``-decorated entry point, also installed as the
``lerobot-train`` console script) and drives it via ``subprocess``.

Why subprocess rather than importing ``lerobot.scripts.lerobot_train.train``
and calling it in-process with a hand-built ``TrainPipelineConfig``:
``TrainPipelineConfig.validate()`` (called from inside ``train()``) resolves
``--policy.path``/``--reward_model.path``/``--config_path`` by re-reading
``sys.argv[1:]`` directly (see ``lerobot.configs.parser.get_path_arg`` and
``TrainPipelineConfig._resolve_pretrained_from_cli``, read in full on
2026-07-13) -- the pretrained-checkpoint and resume-checkpoint resolution
mechanisms are structurally argv-coupled, not purely dataclass-construction
driven. Hand-building a ``TrainPipelineConfig`` and calling ``train(cfg)``
directly would either silently skip that CLI-arg resolution path (surprising
for callers who expect ``--policy.path``-equivalent behavior) or require
reimplementing argv-splicing ourselves -- exactly the kind of "reinventing
LeRobot's own conventions" the no-goal warns against. Driving the real,
documented, officially-tested CLI entry point via ``subprocess`` is the
faithful adapter: it is the same invocation shape as LeRobot's own
``docs/source/smolvla.mdx`` fine-tuning example, and matches this repo's own
established pattern (``model_runtime_server``'s integration test already
drives a real subprocess for the same "exercise the real, separate-process
entry point" reason).
"""

from __future__ import annotations

import json
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

_METADATA_FILENAME = "finetune_job_meta.json"


class CorruptCheckpointError(Exception):
    """Raised when :meth:`FineTuneJob.resume` is asked to resume from a
    checkpoint directory that fails structural validation -- missing
    required files, a safetensors header that won't parse, or a training
    step record that won't parse. Never silently continued from (the
    design's "Fails" requirement, component 01.3)."""


class FineTuneRunError(Exception):
    """Raised when the underlying ``lerobot-train`` subprocess exits
    non-zero. Wraps the captured stdout/stderr for diagnosis."""

    def __init__(self, message: str, returncode: int, output: str):
        super().__init__(message)
        self.returncode = returncode
        self.output = output


@dataclass
class FineTuneJob:
    """One fine-tuning run's identity, persisting across :meth:`run` and
    :meth:`resume` (component 01.3's interface note: "identity persists
    across the run").

    Constructed by :meth:`run` or :meth:`resume` -- not directly -- so its
    fields always reflect a run that has actually been launched (or is
    about to be, for the dependency-injected ``runner`` path tests use).
    """

    run_id: str
    output_path: Path
    full_finetune: bool
    checkpoint_path: str
    dataset_repo_id: str
    dataset_root: Optional[str] = None
    steps: int = 20_000
    extra_args: Sequence[str] = field(default_factory=tuple)

    # ------------------------------------------------------------------
    # Public interface (component 01.3, exact per the work-order):
    #   FineTuneJob.run(checkpoint_path, episodes, output_path) -> FineTuneJob
    #   FineTuneJob.resume(checkpoint_path) -> FineTuneJob
    # ------------------------------------------------------------------

    @classmethod
    def run(
        cls,
        checkpoint_path: str,
        episodes: "Episodes",
        output_path: str,
        *,
        full_finetune: bool = False,
        steps: int = 20_000,
        batch_size: int = 8,
        job_name: Optional[str] = None,
        extra_args: Sequence[str] = (),
        runner=None,
    ) -> "FineTuneJob":
        """Fine-tunes ``checkpoint_path`` against ``episodes``, writing the
        result to ``output_path``.

        Default training path (matching the paper, per the model-runtime
        foundation's "Fine-tune locally, from real or simulated episodes
        alike" goal): the VLM backbone is frozen and only the action expert
        is updated (``train_expert_only=True``, ``freeze_vision_encoder=True``),
        initialized from the pretrained checkpoint's own weights
        (``load_vlm_weights=True``). Pass ``full_finetune=True`` to instead
        train every parameter (``train_expert_only=False``) -- this never
        silently desyncs a run from its checkpoint's recorded mode: see
        ``_write_metadata`` below, and note LeRobot's own checkpoint already
        redundantly records the same flag in
        ``pretrained_model/train_config.json`` (``policy.train_expert_only``),
        since ``TrainPipelineConfig`` is saved alongside every checkpoint
        LeRobot writes.

        ``episodes`` (an ``Episodes`` value -- see this module's
        ``Episodes`` helper) carries a LeRobotDataset ``repo_id`` and
        optionally a local ``root``. Its provenance (real robot vs.
        simulation) is never inspected here: nothing in this function
        branches on it, matching the CONTEXT term "Episode"'s "never
        changes its shape or how a FineTuneJob adapter consumes it".

        ``runner`` is dependency-injected (defaults to a real subprocess
        call) so fast tests can substitute a fake without invoking a real,
        multi-minute training run -- mirrors ``InferActionServer``'s
        model-injection pattern in ``model_runtime_server/server.py``.
        """
        run_id = uuid.uuid4().hex
        output = Path(output_path)
        job = cls(
            run_id=run_id,
            output_path=output,
            full_finetune=full_finetune,
            checkpoint_path=checkpoint_path,
            dataset_repo_id=episodes.repo_id,
            dataset_root=episodes.root,
            steps=steps,
            extra_args=tuple(extra_args),
        )

        args = job._build_run_args(
            batch_size=batch_size,
            job_name=job_name or f"finetune-{run_id[:8]}",
        )
        _invoke(args, runner=runner)

        job._write_metadata()
        return job

    @classmethod
    def resume(
        cls,
        checkpoint_path: str,
        *,
        extra_args: Sequence[str] = (),
        runner=None,
    ) -> "FineTuneJob":
        """Resumes an interrupted run from its last checkpoint.

        ``checkpoint_path`` must be a step-checkpoint directory in LeRobot's
        own layout (``<output_dir>/checkpoints/<step>/``, containing
        ``pretrained_model/`` and ``training_state/`` -- see
        ``lerobot.common.train_utils.save_checkpoint``'s docstring) or the
        ``last`` symlink pointing at one. Validated structurally
        (:func:`validate_checkpoint`) before LeRobot's own resume mechanism
        ever runs, so a corrupt checkpoint is detected here rather than
        silently continued from (component 01.3's "Fails" requirement) --
        this is on top of, not instead of, LeRobot's own loud failure when
        ``training_state/`` is altogether missing
        (``load_training_state`` raises ``NotADirectoryError``).

        Reads this job's own metadata sidecar (written by :meth:`run`) to
        recover ``full_finetune`` and dataset identity for the returned
        ``FineTuneJob``'s record-keeping; falls back to LeRobot's own
        ``train_config.json`` (always present in ``pretrained_model/``) if
        the sidecar is absent, since that file alone is sufficient to
        recover ``train_expert_only``.
        """
        checkpoint_dir = Path(checkpoint_path)
        validate_checkpoint(checkpoint_dir)

        pretrained_dir = checkpoint_dir / "pretrained_model"
        config_path = pretrained_dir / "train_config.json"

        train_config = json.loads(config_path.read_text())
        policy_cfg = train_config.get("policy") or {}
        full_finetune = not bool(policy_cfg.get("train_expert_only", True))
        dataset_cfg = train_config.get("dataset") or {}

        meta = _read_metadata(checkpoint_dir)

        run_id = meta.get("run_id") if meta else uuid.uuid4().hex
        output_path = Path(meta["output_path"]) if meta else checkpoint_dir
        steps = int(train_config.get("steps", 20_000))

        job = cls(
            run_id=run_id,
            output_path=output_path,
            full_finetune=meta.get("full_finetune", full_finetune)
            if meta
            else full_finetune,
            checkpoint_path=str(checkpoint_dir),
            dataset_repo_id=meta.get("dataset_repo_id", dataset_cfg.get("repo_id"))
            if meta
            else dataset_cfg.get("repo_id"),
            dataset_root=meta.get("dataset_root", dataset_cfg.get("root"))
            if meta
            else dataset_cfg.get("root"),
            steps=steps,
            extra_args=tuple(extra_args),
        )

        args = [
            sys.executable,
            "-m",
            "lerobot.scripts.lerobot_train",
            f"--config_path={config_path}",
            "--resume=true",
            *extra_args,
        ]
        _invoke(args, runner=runner)

        job._write_metadata()
        return job

    # ------------------------------------------------------------------
    # Internals.
    # ------------------------------------------------------------------

    def _build_run_args(self, *, batch_size: int, job_name: str) -> list:
        args = [
            sys.executable,
            "-m",
            "lerobot.scripts.lerobot_train",
            f"--policy.path={self.checkpoint_path}",
            "--policy.load_vlm_weights=true",
            f"--policy.train_expert_only={'false' if self.full_finetune else 'true'}",
            f"--policy.freeze_vision_encoder={'false' if self.full_finetune else 'true'}",
            f"--dataset.repo_id={self.dataset_repo_id}",
            f"--batch_size={batch_size}",
            f"--steps={self.steps}",
            f"--output_dir={self.output_path}",
            f"--job_name={job_name}",
            "--policy.push_to_hub=false",
        ]
        if self.dataset_root is not None:
            args.append(f"--dataset.root={self.dataset_root}")
        args.extend(self.extra_args)
        return args

    def _write_metadata(self) -> None:
        """Records which mode (frozen-backbone default vs. full fine-tune)
        produced this run, alongside the output weights -- the design's
        invariant that a run and its checkpoint are "never silently
        inconsistent" about training mode. This is a convenience sidecar on
        top of LeRobot's own redundant record: every checkpoint LeRobot
        writes already carries its resolved ``TrainPipelineConfig``
        (``pretrained_model/train_config.json``, via
        ``TrainPipelineConfig._save_pretrained``), including
        ``policy.train_expert_only`` -- so the mode is recoverable even if
        this sidecar is lost, but this file makes it directly retrievable
        from ``output_path`` without parsing LeRobot's full config shape.
        """
        self.output_path.mkdir(parents=True, exist_ok=True)
        metadata = {
            "run_id": self.run_id,
            "full_finetune": self.full_finetune,
            "checkpoint_path": self.checkpoint_path,
            "dataset_repo_id": self.dataset_repo_id,
            "dataset_root": self.dataset_root,
            "output_path": str(self.output_path),
            "steps": self.steps,
        }
        (self.output_path / _METADATA_FILENAME).write_text(
            json.dumps(metadata, indent=2)
        )


@dataclass(frozen=True)
class Episodes:
    """A set of LeRobotDataset-format episodes, either from the Hugging Face
    Hub (``repo_id`` alone) or a local directory (``repo_id`` plus ``root``
    pointing at the directory containing ``meta/info.json``).

    Deliberately carries no provenance field (real robot vs. simulation):
    per the CONTEXT term "Episode", provenance never changes an episode
    value's shape or how a FineTuneJob adapter consumes it, so there is
    nothing here to distinguish -- a simulation-sourced local dataset and a
    real-robot-sourced local dataset are both just a ``repo_id``/``root``
    pair, indistinguishable to every code path in this module.
    """

    repo_id: str
    root: Optional[str] = None


def validate_checkpoint(checkpoint_dir: Path) -> None:
    """Structurally validates a LeRobot step-checkpoint directory before
    handing it to ``lerobot-train --resume=true``, raising
    :class:`CorruptCheckpointError` on any failure -- shape/structure
    validated, per component 01.3's "Fails" requirement ("never silently
    continues from a corrupt checkpoint (checksum or shape-validated on
    resume)").

    Checks, in order: the expected directory layout exists
    (``pretrained_model/{config.json,model.safetensors,train_config.json}``,
    ``training_state/training_step.json``); ``model.safetensors``' header
    parses and declares at least one tensor (a real, non-empty weight file,
    not zero bytes or truncated mid-header); ``training_step.json`` parses
    as JSON with an integer ``step``. This does not re-verify LeRobot's own
    deeper failure paths (e.g. ``load_training_state`` raising
    ``NotADirectoryError`` if ``training_state/`` is altogether absent --
    already loud) -- it exists to catch corruption LeRobot's own resume
    path would otherwise walk into lazily or silently.
    """
    if not checkpoint_dir.is_dir():
        raise CorruptCheckpointError(
            f"checkpoint directory does not exist: {checkpoint_dir}"
        )

    pretrained_dir = checkpoint_dir / "pretrained_model"
    training_state_dir = checkpoint_dir / "training_state"

    required = [
        pretrained_dir / "config.json",
        pretrained_dir / "model.safetensors",
        pretrained_dir / "train_config.json",
        training_state_dir / "training_step.json",
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise CorruptCheckpointError(
            f"checkpoint at {checkpoint_dir} is missing required file(s): {missing}"
        )

    _validate_safetensors_header(pretrained_dir / "model.safetensors")
    _validate_training_step(training_state_dir / "training_step.json")


def _validate_safetensors_header(path: Path) -> None:
    try:
        from safetensors import safe_open
    except (
        ImportError
    ) as exc:  # pragma: no cover -- safetensors is a real, always-installed dep
        raise CorruptCheckpointError(
            f"cannot validate {path}: safetensors not importable"
        ) from exc

    try:
        with safe_open(str(path), framework="numpy") as f:
            keys = list(f.keys())
    except Exception as exc:
        raise CorruptCheckpointError(
            f"checkpoint weight file {path} failed to parse as safetensors: {exc}"
        ) from exc

    if not keys:
        raise CorruptCheckpointError(
            f"checkpoint weight file {path} declares zero tensors"
        )

    # Shape-validate every declared tensor's header is internally consistent
    # (safe_open's lazy get_slice still reads each tensor's shape/dtype from
    # the header without materializing the data) -- catches a header that
    # parses but describes an inconsistent/truncated file.
    try:
        with safe_open(str(path), framework="numpy") as f:
            for key in keys:
                _ = f.get_slice(key).get_shape()
    except Exception as exc:
        raise CorruptCheckpointError(
            f"checkpoint weight file {path} has an inconsistent tensor header for a declared key: {exc}"
        ) from exc


def _validate_training_step(path: Path) -> None:
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        raise CorruptCheckpointError(
            f"training step file {path} is not valid JSON: {exc}"
        ) from exc

    step = payload.get("step")
    if not isinstance(step, int):
        raise CorruptCheckpointError(
            f"training step file {path} has a missing/non-integer 'step' field: {payload!r}"
        )


def _read_metadata(checkpoint_dir: Path) -> Optional[dict]:
    """Walks up from a step-checkpoint dir looking for this module's own
    ``finetune_job_meta.json`` sidecar, written at ``output_path`` by
    :meth:`FineTuneJob.run`/``resume`` -- ``output_path`` is an ancestor of
    every step-checkpoint dir LeRobot writes under it
    (``output_path/checkpoints/<step>/``), so this looks in the checkpoint
    dir itself, its parent, and its grandparent."""
    for candidate in (
        checkpoint_dir / _METADATA_FILENAME,
        checkpoint_dir.parent / _METADATA_FILENAME,
        checkpoint_dir.parent.parent / _METADATA_FILENAME,
    ):
        if candidate.is_file():
            try:
                return json.loads(candidate.read_text())
            except Exception:
                return None
    return None


def _invoke(args: Sequence[str], *, runner=None) -> None:
    """Runs the ``lerobot-train`` subprocess (or the injected fake
    ``runner``), raising :class:`FineTuneRunError` on non-zero exit."""
    call = runner or _default_runner
    result = call(list(args))
    if result.returncode != 0:
        raise FineTuneRunError(
            f"lerobot-train exited with code {result.returncode}",
            returncode=result.returncode,
            output=getattr(result, "stdout", "") or "",
        )


def _default_runner(args: Sequence[str]):
    return subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


__all__ = [
    "FineTuneJob",
    "Episodes",
    "CorruptCheckpointError",
    "FineTuneRunError",
    "validate_checkpoint",
]
