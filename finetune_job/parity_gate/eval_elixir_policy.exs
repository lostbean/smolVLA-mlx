## One-shot evaluation driver for the parity gate (issue 08) -- NOT a
## permanent module, mirrors this repo's own convention of driving a
## real, separate-process entry point for a real run (see
## `finetune_job/job.py`'s own moduledoc on why the Python trainer shells
## out to `lerobot-train` rather than importing it in-process). Invoked via
## `mix run finetune_job/parity_gate/eval_elixir_policy.exs -- <args>` from
## the Python orchestrator (`finetune_job/parity_gate/run_gate.py`), which
## owns the actual accuracy-proxy/threshold computation -- this script's
## only job is to produce real `(predicted_action_chunk, ground_truth_action)`
## pairs from the real Elixir-native `SmolVLA.infer_action/4` (component
## 01.2) against a real fine-tuned checkpoint (component 01.4's output) and
## real held-out episode frames (`SmolVLA.Dataset`, the same real reader
## `FineTuneJob.run/4`'s own training path uses), plus real per-call
## timing for the throughput measurement -- so both languages' accuracy
## numbers are computed by the SAME formula (`finetune_job.parity_gate.metrics`),
## not two independently-written (and possibly subtly different) ones.
##
## Args (all required, positional, space-separated after `--`):
##   1. checkpoint_dir     -- the fine-tuned checkpoint to evaluate (a real
##                             SmolVLA.load/1-loadable directory)
##   2. dataset_root        -- the FULL real LeRobotDataset root (held-out
##                             episodes must still be readable, even though
##                             training only ever saw the train-only copy)
##   3. holdout_episodes_csv -- comma-separated episode indices, e.g. "0,8,16"
##   4. n_frames_per_episode -- how many frames to subsample per held-out
##                             episode (evenly spaced across the episode)
##   5. n_throughput_calls  -- how many timed infer_action calls for the
##                             throughput probe (after 1 untimed warmup)
##   6. output_json_path    -- where to write this script's real results
##
## Output JSON shape:
##   {
##     "episode_predictions": {"<episode_index>": [[chunk, ground_truth], ...], ...},
##     "throughput": {"n_calls": N, "total_seconds": S}
##   }

# `mix run <script>.exs -- a b c` includes the literal "--" separator in
# `System.argv()` for a plain script (confirmed directly: unlike `mix run
# -e`, which does not) -- strip it if present so this script works
# identically whether invoked with or without the separator.
[
  checkpoint_dir,
  dataset_root,
  holdout_csv,
  n_frames_str,
  n_throughput_calls_str,
  output_json_path
] =
  case System.argv() do
    ["--" | rest] -> rest
    args -> args
  end

n_frames_per_episode = String.to_integer(n_frames_str)
n_throughput_calls = String.to_integer(n_throughput_calls_str)
holdout_episodes = holdout_csv |> String.split(",") |> Enum.map(&String.to_integer/1)

IO.puts(
  "eval_elixir_policy: checkpoint=#{checkpoint_dir} dataset_root=#{dataset_root} " <>
    "holdout=#{inspect(holdout_episodes)} n_frames_per_episode=#{n_frames_per_episode}"
)

# Real, direct finding from this chunk's own work: WITHOUT setting the
# emily/MLX GPU backend explicitly, `SmolVLA.infer_action/4` silently runs
# on Elixir's default `Nx.BinaryBackend` (pure CPU, no MLX acceleration)
# instead of `emily`'s GPU-backed `Emily.Backend` -- confirmed directly by
# a real isolated probe during this chunk's own work: ~7.3-7.7s/call on
# the default backend vs. ~1.2s/call (matching this repo's own documented
# component 01.2 warm-latency figure) with this backend set explicitly.
# `SmolVLA.infer_action/4` itself never sets a global backend -- every
# other real caller in this repo (`FineTuneJob.run/4`,
# `test/smol_vla/control_loop_integration_test.exs`) sets it once at its
# own entry point, and this evaluation script is exactly that kind of
# entry point too.
Nx.global_default_backend({Emily.Backend, device: :gpu})
Nx.Defn.global_default_options(compiler: Emily.Compiler)

model = SmolVLA.load(checkpoint_dir)
dataset = SmolVLA.Dataset.open(dataset_root)

# Evenly-spaced subsample within one episode's frame list -- same
# rationale as the held-out episode SELECTION itself (split.py's module
# doc): spread across the episode's real duration rather than clustering
# at its start, without needing every frame (keeping the total real
# `infer_action` call count, and therefore real wall-clock time, bounded).
pick_indices = fn total, n ->
  if n >= total do
    Enum.to_list(0..(total - 1))
  else
    step = total / n

    0..(n - 1)
    |> Enum.map(fn i -> min(total - 1, trunc(i * step)) end)
    |> Enum.uniq()
  end
end

episode_predictions =
  Enum.map(holdout_episodes, fn episode_index ->
    frames = SmolVLA.Dataset.frames(dataset, episode_index)
    total = length(frames)
    indices = pick_indices.(total, n_frames_per_episode)
    frame_arr = List.to_tuple(frames)

    pairs =
      Enum.map(indices, fn i ->
        frame = elem(frame_arr, i)

        action_chunk =
          SmolVLA.infer_action(model, frame.image, frame.state, frame.task)

        chunk_list = action_chunk |> Nx.backend_transfer(Nx.BinaryBackend) |> Nx.to_list()

        # A 2-element LIST, not a tuple -- `Jason.Encoder` has no
        # implementation for `Tuple` (confirmed directly: this script
        # originally built a tuple here and `Jason.encode!/1` raised
        # `Protocol.UndefinedError` for it), and the Python side
        # (`run_gate.py`'s `evaluate_elixir_policy`) already reads this
        # back as `[chunk, ground_truth]` pairs either way.
        [chunk_list, frame.action]
      end)

    IO.puts(
      "eval_elixir_policy: episode #{episode_index} -- evaluated #{length(pairs)}/#{total} frames"
    )

    {Integer.to_string(episode_index), pairs}
  end)
  |> Map.new()

# Real throughput timing: re-run `infer_action` on a fixed frame (already
# resident/decoded -- warm, steady-state calls, matching this repo's own
## warm/cold latency distinction, e.g. component 01.2's "warm latency
## (~1.2s)" design text) -- one warmup call excluded from the timed window.
first_episode = List.first(holdout_episodes)
warmup_frame = List.first(SmolVLA.Dataset.frames(dataset, first_episode))

_ = SmolVLA.infer_action(model, warmup_frame.image, warmup_frame.state, warmup_frame.task)

t0 = System.monotonic_time(:microsecond)

Enum.each(1..n_throughput_calls, fn _ ->
  SmolVLA.infer_action(model, warmup_frame.image, warmup_frame.state, warmup_frame.task)
end)

elapsed_seconds = (System.monotonic_time(:microsecond) - t0) / 1_000_000

IO.puts(
  "eval_elixir_policy: throughput probe -- #{n_throughput_calls} calls in " <>
    "#{Float.round(elapsed_seconds, 3)}s (#{Float.round(elapsed_seconds / n_throughput_calls, 3)}s/call)"
)

result = %{
  "episode_predictions" => episode_predictions,
  "throughput" => %{
    "n_calls" => n_throughput_calls,
    "total_seconds" => elapsed_seconds
  }
}

File.write!(output_json_path, Jason.encode!(result))
IO.puts("eval_elixir_policy: wrote results to #{output_json_path}")
