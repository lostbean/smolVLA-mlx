defmodule FineTuneJobTest do
  @moduledoc """
  Fast, synthetic tests for `FineTuneJob`'s checkpoint validation and
  resume-identity logic -- TDD directive step (6): "resume-from-checkpoint
  and corrupt-checkpoint-detection". Real fine-tuning runs against the
  real checkpoint/dataset are covered separately (tagged `:real_checkpoint`,
  see `test/finetune_job/real_checkpoint_test.exs`).
  """
  use ExUnit.Case, async: true

  alias FineTuneJob.CorruptCheckpointError

  setup do
    dir = Path.join(System.tmp_dir!(), "finetune_job_test_#{System.unique_integer([:positive])}")
    File.mkdir_p!(dir)
    on_exit(fn -> File.rm_rf!(dir) end)
    {:ok, dir: dir}
  end

  defp write_valid_checkpoint(dir, opts \\ []) do
    step = Keyword.get(opts, :step, 3)
    run_id = Keyword.get(opts, :run_id, "abc123")

    tensors = %{"a" => Nx.tensor([1.0, 2.0, 3.0])}
    weights_path = Path.join(dir, "model.safetensors")
    Safetensors.write!(weights_path, tensors)

    checksum =
      weights_path
      |> File.stream!(2048)
      |> Enum.reduce(:crypto.hash_init(:sha256), &:crypto.hash_update(&2, &1))
      |> :crypto.hash_final()
      |> Base.encode16(case: :lower)

    training_state = %{"step" => step, "run_id" => run_id, "weights_checksum_sha256" => checksum}

    Path.join(dir, "training_state.json")
    |> File.write!(Jason.encode!(training_state))

    :ok
  end

  describe "validate_checkpoint!/1" do
    test "passes for a real, well-formed checkpoint directory", %{dir: dir} do
      write_valid_checkpoint(dir)
      assert :ok = FineTuneJob.validate_checkpoint!(dir)
    end

    test "raises on a nonexistent directory" do
      refute File.dir?("/nonexistent/path/xyz")

      assert_raise CorruptCheckpointError, ~r/does not exist/, fn ->
        FineTuneJob.validate_checkpoint!("/nonexistent/path/xyz")
      end
    end

    test "raises when model.safetensors is missing", %{dir: dir} do
      write_valid_checkpoint(dir)
      File.rm!(Path.join(dir, "model.safetensors"))

      assert_raise CorruptCheckpointError, ~r/missing required file/, fn ->
        FineTuneJob.validate_checkpoint!(dir)
      end
    end

    test "raises when training_state.json is missing", %{dir: dir} do
      write_valid_checkpoint(dir)
      File.rm!(Path.join(dir, "training_state.json"))

      assert_raise CorruptCheckpointError, ~r/missing required file/, fn ->
        FineTuneJob.validate_checkpoint!(dir)
      end
    end

    test "raises when model.safetensors is truncated/corrupt (not a silent continue)", %{dir: dir} do
      write_valid_checkpoint(dir)
      File.write!(Path.join(dir, "model.safetensors"), <<1, 2, 3>>)

      assert_raise CorruptCheckpointError, ~r/failed to parse as safetensors/, fn ->
        FineTuneJob.validate_checkpoint!(dir)
      end
    end

    test "raises when training_state.json is not valid JSON", %{dir: dir} do
      write_valid_checkpoint(dir)
      File.write!(Path.join(dir, "training_state.json"), "{not json")

      assert_raise CorruptCheckpointError, ~r/not valid JSON/, fn ->
        FineTuneJob.validate_checkpoint!(dir)
      end
    end

    test "raises when training_state.json has no integer step", %{dir: dir} do
      write_valid_checkpoint(dir)
      Path.join(dir, "training_state.json") |> File.write!(Jason.encode!(%{"step" => "three"}))

      assert_raise CorruptCheckpointError, ~r/missing\/non-integer 'step'/, fn ->
        FineTuneJob.validate_checkpoint!(dir)
      end
    end

    test "raises on a checksum mismatch -- the weights file was altered after the training_state.json was written",
         %{dir: dir} do
      write_valid_checkpoint(dir)

      # Simulate corruption: the weights file changes after
      # training_state.json recorded its checksum.
      Safetensors.write!(Path.join(dir, "model.safetensors"), %{"a" => Nx.tensor([9.0, 9.0, 9.0])})

      assert_raise CorruptCheckpointError, ~r/checksum validation/, fn ->
        FineTuneJob.validate_checkpoint!(dir)
      end
    end
  end

  describe "resume/1" do
    test "raises (never silently restarts or continues) from a corrupt checkpoint", %{dir: dir} do
      write_valid_checkpoint(dir)
      File.rm!(Path.join(dir, "model.safetensors"))

      assert_raise CorruptCheckpointError, fn ->
        FineTuneJob.resume(dir)
      end
    end

    test "recovers this run's own identity from a valid checkpoint + metadata sidecar", %{
      dir: dir
    } do
      write_valid_checkpoint(dir, step: 7, run_id: "same-run-id")

      # Metadata sidecar lives at output_path (the checkpoint dir's
      # grandparent, per FineTuneJob's own checkpoints/<step>/ layout) --
      # here dir stands in directly for a step-checkpoint dir, so write
      # the sidecar alongside it (mirrors the "found in the checkpoint
      # dir itself" walk-up case).
      metadata = %{
        "run_id" => "same-run-id",
        "full_finetune" => false,
        "checkpoint_path" => "/some/checkpoint",
        "dataset_root" => "/some/dataset",
        "output_path" => dir
      }

      Path.join(dir, "finetune_job_meta.json") |> File.write!(Jason.encode!(metadata))

      job = FineTuneJob.resume(dir)

      assert job.run_id == "same-run-id"
      assert job.step == 7
      assert job.full_finetune == false
      assert job.dataset_root == "/some/dataset"
    end
  end
end
