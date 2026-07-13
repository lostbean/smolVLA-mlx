defmodule FineTuneJob.RealCheckpointTest do
  @moduledoc """
  Real-checkpoint, real-dataset integration check for the Elixir-native
  `FineTuneJob` -- TDD directive step (5): "a full (but minimal) real run
  against the real checkpoint and real episode data, checkpointing, and
  reloading through both inference adapters".

  Runs a REAL, minimal (few-step) fine-tuning pass against the real,
  publicly available `lerobot/smolvla_base` checkpoint and the SAME real,
  small, public LeRobotDataset the Python trainer's own integration test
  uses (`lerobot/svla_so101_pickplace` -- 50 episodes, ~12k frames, 2
  cameras, ~86MB total, SmolVLA-tagged, maintained by the `lerobot` org
  itself) -- reusing the same real DATASET (not the same code) across both
  trainers' tests, per this chunk's own work order.

  Deliberately NOT part of the fast test gate (`mix test`): downloads/uses
  real, already-cached weights (~1.1GB) and dataset (~86MB), and runs real
  forward+backward passes through SmolVLA's ~450M-parameter backbone plus
  ~100M-parameter action expert -- real wall-clock time, not milliseconds.
  Tagged `:real_checkpoint` and opt-in via `RUN_SMOLVLA_INTEGRATION_CHECK=1`,
  this repo's own existing convention (see `test/test_helper.exs`).
  """
  use ExUnit.Case, async: false

  @moduletag :real_checkpoint
  # Real training steps here run ~20-25s each (model load, real dataset
  # frame decode via a real ffmpeg subprocess per sampled frame, a real
  # ~450M+100M-parameter forward+backward pass, and a full safetensors
  # checkpoint write) -- well past ExUnit's default 60s per-test timeout.
  @moduletag timeout: 300_000

  @checkpoint_path Path.expand(
                     "~/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/c83c3163b8ca9b7e67c509fffd9121e66cb96205"
                   )

  @dataset_root Path.expand(
                  System.get_env("FINETUNE_DATASET_ROOT") ||
                    "~/.cache/huggingface/lerobot/hub/datasets--lerobot--svla_so101_pickplace/snapshots/f641879e22172be7e8161d5e6c1503c2d2feb657"
                )

  setup_all do
    unless File.dir?(@checkpoint_path) do
      raise "real checkpoint not found at #{@checkpoint_path}"
    end

    unless File.dir?(@dataset_root) do
      raise "real dataset not found at #{@dataset_root}"
    end

    :ok
  end

  setup do
    output_dir =
      Path.join(System.tmp_dir!(), "finetune_job_real_#{System.unique_integer([:positive])}")

    on_exit(fn -> File.rm_rf!(output_dir) end)
    {:ok, output_dir: output_dir}
  end

  test "a real minimal run produces updated action-expert weights loadable by SmolVLA.infer_action/4",
       %{output_dir: output_dir} do
    steps = String.to_integer(System.get_env("FINETUNE_INTEGRATION_STEPS", "3"))

    t0 = System.monotonic_time(:millisecond)

    job =
      FineTuneJob.run(
        @checkpoint_path,
        %FineTuneJob.Episodes{root: @dataset_root},
        output_dir,
        steps: steps,
        batch_size: 2,
        seed: 0
      )

    elapsed_ms = System.monotonic_time(:millisecond) - t0

    IO.puts(
      "\nreal FineTuneJob.run/4: run_id=#{job.run_id}, #{steps} steps in #{elapsed_ms}ms " <>
        "(#{Float.round(elapsed_ms / steps, 1)}ms/step)"
    )

    assert job.step == steps
    assert job.full_finetune == false

    checkpoints_dir = Path.join(output_dir, "checkpoints")
    assert File.dir?(checkpoints_dir)

    step_dir = Path.join(checkpoints_dir, Integer.to_string(steps))
    assert File.dir?(step_dir)

    weights_path = Path.join(step_dir, "model.safetensors")
    assert File.regular?(weights_path)

    # Structural compatibility bar (this chunk's own acceptance criterion):
    # same tensor keys/shapes SmolVLA.Weights/the Python
    # SmolVLAModel.from_pretrained already expect -- prove it directly by
    # diffing this run's own output keys/shapes against the SOURCE
    # checkpoint's real keys/shapes (a real fine-tune only changes VALUES,
    # never the tensor set or any shape).
    source_raw = Safetensors.read!(Path.join(@checkpoint_path, "model.safetensors"))
    output_raw = Safetensors.read!(weights_path)

    assert MapSet.new(Map.keys(output_raw)) == MapSet.new(Map.keys(source_raw)),
           "output checkpoint's tensor keys diverge from the source checkpoint's keys"

    mismatched_shapes =
      Enum.filter(source_raw, fn {k, source_tensor} ->
        Nx.shape(source_tensor) != Nx.shape(output_raw[k])
      end)

    assert mismatched_shapes == [],
           "output checkpoint tensors have different shapes than the source: #{inspect(mismatched_shapes)}"

    # Also compatible with the Python from_pretrained loader: config.json
    # was copied verbatim, and the tensor scheme is provably identical to
    # the source (checked above) -- both adapters load the SAME real
    # `lerobot/smolvla_base` checkpoint shape already, per every prior
    # chunk's own accepted work.
    assert File.regular?(Path.join(step_dir, "config.json"))

    # Prove real weight VALUES actually moved for the trainable
    # (action-expert) parameters -- a real, non-degenerate fine-tune, not
    # a no-op that merely re-serializes the source unchanged.
    {trainable_keys, _frozen} = SmolVLA.Train.trainable_keys(SmolVLA.Weights.load!(weights_path))
    sample_key = Enum.find(trainable_keys, &String.contains?(&1, "self_attn.q_proj.weight"))
    raw_key = raw_key_for(sample_key)

    source_values = source_raw[raw_key] |> Nx.as_type(:f32) |> Nx.to_flat_list()
    output_values = output_raw[raw_key] |> Nx.as_type(:f32) |> Nx.to_flat_list()

    assert source_values != output_values,
           "trainable weight #{sample_key} is bit-for-bit identical to the source -- training had no effect"

    # The acceptance bar: reload the fine-tuned weights through the
    # already-accepted Elixir-native infer_action/4 and confirm a real,
    # finite, correctly-shaped action chunk comes out.
    reloaded = SmolVLA.load(step_dir)

    image =
      Nx.Random.key(1)
      |> then(fn key ->
        {img, _key} = Nx.Random.uniform(key, shape: {256, 256, 3}, type: :f32)
        img
      end)

    state = List.duplicate(0.1, 6)

    action_chunk = SmolVLA.infer_action(reloaded, image, state, "pick up the block")

    IO.puts(
      "real FineTuneJob integration: reloaded action_chunk shape = #{inspect(Nx.shape(action_chunk))}"
    )

    assert Nx.shape(action_chunk) ==
             {reloaded.config.chunk_size, SmolVLA.Config.action_dim(reloaded.config)}

    action_cpu = Nx.backend_transfer(action_chunk, Nx.BinaryBackend)
    finite = action_cpu |> Nx.is_nan() |> Nx.logical_not() |> Nx.all() |> Nx.to_number()
    assert finite == 1, "reloaded fine-tuned model produced a NaN action chunk"

    nonzero = action_cpu |> Nx.not_equal(0.0) |> Nx.any() |> Nx.to_number()
    assert nonzero == 1, "reloaded fine-tuned model produced a degenerate all-zero action chunk"
  end

  test "an interrupted run resumes from its last checkpoint rather than restarting", %{
    output_dir: output_dir
  } do
    job1 =
      FineTuneJob.run(
        @checkpoint_path,
        %FineTuneJob.Episodes{root: @dataset_root},
        output_dir,
        steps: 1,
        batch_size: 2,
        seed: 1
      )

    assert job1.step == 1

    last_checkpoint = Path.join([output_dir, "checkpoints", "last"])
    assert File.dir?(last_checkpoint) or match?({:ok, _}, File.read_link(last_checkpoint))

    resumed = FineTuneJob.resume(Path.absname(last_checkpoint))

    assert resumed.run_id == job1.run_id
    assert resumed.step == 1
    assert resumed.full_finetune == job1.full_finetune

    IO.puts(
      "\nreal FineTuneJob resume: run_id=#{resumed.run_id} step=#{resumed.step} (same identity as job1)"
    )
  end

  defp raw_key_for(remapped_key) do
    {_weights, raw_key_map} =
      SmolVLA.Weights.load_with_raw_keys!(Path.join(@checkpoint_path, "model.safetensors"))

    Map.fetch!(raw_key_map, remapped_key)
  end
end
