defmodule SmolVLA.DatasetTest do
  @moduledoc """
  TDD directive step (3): "real episode-data loading (LeRobotDataset
  parquet/video format -- verify you can read real frames/actions from the
  real dataset)".

  Reads the SAME real, public, SmolVLA-tagged dataset the Python trainer's
  own integration test uses (`lerobot/svla_so101_pickplace`) directly off
  disk -- no fabricated/synthetic episode data, per this chunk's own
  escalation-rule guidance ("report this rather than fabricating synthetic
  episode data that wouldn't actually exercise the real format").

  Tagged `:real_checkpoint` (reused, not a new tag -- this repo's existing
  convention per `test/test_helper.exs`) and opt-in via
  `RUN_SMOLVLA_INTEGRATION_CHECK=1`, even though this test reads dataset
  files rather than the model checkpoint, because it has the same
  real-wall-clock-time/real-large-download shape every other
  `:real_checkpoint` test in this repo already has (video decoding via a
  real `ffmpeg` subprocess per frame is not instant).
  """
  use ExUnit.Case, async: false

  alias SmolVLA.Dataset

  @moduletag :real_checkpoint

  @dataset_root Path.expand(
                  System.get_env("FINETUNE_DATASET_ROOT") ||
                    "~/.cache/huggingface/lerobot/hub/datasets--lerobot--svla_so101_pickplace/snapshots/f641879e22172be7e8161d5e6c1503c2d2feb657"
                )

  setup_all do
    unless File.dir?(@dataset_root) do
      raise "real dataset not found at #{@dataset_root} -- this test requires the " <>
              "real lerobot/svla_so101_pickplace dataset already cached locally " <>
              "(same dataset the Python finetune_job integration test uses)."
    end

    {:ok, dataset: Dataset.open(@dataset_root)}
  end

  test "open/1 reads real meta/info.json and the real episode/task index", %{dataset: dataset} do
    assert Dataset.total_episodes(dataset) == 50
    assert dataset.info["codebase_version"] == "v3.0"
    assert dataset.info["robot_type"] == "so100_follower"
    assert map_size(dataset.tasks) >= 1
  end

  test "frames/2 reads real per-frame action/state and real decoded camera images for episode 0",
       %{
         dataset: dataset
       } do
    frames = Dataset.frames(dataset, 0)

    refute Enum.empty?(frames)
    assert Enum.all?(frames, &(&1.episode_index == 0))

    # frame_index is contiguous from 0, matching the real parquet's own
    # per-episode frame_index column.
    frame_indices = Enum.map(frames, & &1.frame_index)
    assert frame_indices == Enum.to_list(0..(length(frames) - 1))

    first = hd(frames)
    assert is_binary(first.task) and first.task != ""
    assert length(first.state) == 6
    assert length(first.action) == 6
    assert Enum.all?(first.state, &is_number/1)
    assert Enum.all?(first.action, &is_number/1)

    assert Nx.shape(first.image) == {480, 640, 3}
    assert Nx.type(first.image) == {:u, 8}

    # A real decoded camera frame is not a degenerate all-zero/all-same
    # image.
    assert Nx.to_number(Nx.reduce_max(first.image)) > 0
    unique_values = first.image |> Nx.to_flat_list() |> Enum.uniq() |> length()
    assert unique_values > 10
  end

  test "frames/2 raises on an out-of-range episode index rather than silently returning nothing",
       %{
         dataset: dataset
       } do
    assert_raise ArgumentError, ~r/no episode/, fn ->
      Dataset.frames(dataset, 999_999)
    end
  end
end
