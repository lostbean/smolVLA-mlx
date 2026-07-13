defmodule SmolVLA.ControlLoopIntegrationTest do
  @moduledoc """
  Proves this chunk's second acceptance criterion: `ControlLoop`
  (already built, already accepted) runs UNMODIFIED against the new
  `:emily_native` adapter -- only `start_link/1`'s `adapter`/
  `adapter_module`/`adapter_client` options change; no code in
  `ControlLoop`'s own queue/timing/state-machine logic changes. This
  test starts a REAL `ControlLoop`, wired to a REAL `SmolVLA.t()` (the
  real checkpoint) through `SmolVLA.Adapter`, ticks it for real, and
  confirms the queue/timing mechanics genuinely work against the new
  adapter -- not a mock standing in for it.

  `ControlLoop.start_link/1`'s own source is unmodified in its
  queue/timing/state-machine logic; the only change made anywhere in
  `lib/control_loop.ex` for this chunk is the `:emily_native` dispatch
  clause in `start_link/1` itself, which now starts the SAME generic
  `GenServer.start_link` path `:zeromq_fallback` already used, once a
  real `adapter_module`/`adapter_client` are supplied -- see that
  function's own comment.
  """
  use ExUnit.Case, async: false

  @checkpoint_dir Path.expand(
                    "~/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/c83c3163b8ca9b7e67c509fffd9121e66cb96205"
                  )

  setup_all do
    Nx.global_default_backend({Emily.Backend, device: :gpu})
    Nx.Defn.global_default_options(compiler: Emily.Compiler)
    :ok
  end

  @tag :real_checkpoint
  @tag timeout: 30_000
  test "a real ControlLoop, wired to :emily_native, ticks and triggers a real infer_action" do
    model = SmolVLA.load(@checkpoint_dir)
    test_pid = self()

    # A small queue so its pre-pop depth is already below the low-water
    # threshold on the very first tick -- triggers a real infer_action
    # immediately, same trigger condition the existing :zeromq_fallback
    # tests use.
    initial_queue =
      ControlLoop.ActionQueue.new()
      |> ControlLoop.ActionQueue.enqueue(for i <- 1..5, do: [i * 1.0])

    {:ok, pid} =
      ControlLoop.start_link(
        adapter: :emily_native,
        adapter_module: SmolVLA.Adapter,
        adapter_client: model,
        initial_queue: initial_queue,
        low_water_threshold: 25,
        actuator_sink: fn action -> send(test_pid, {:sent, action}) end
      )

    assert Process.alive?(pid)

    :ok = ControlLoop.tick(pid)
    assert_receive {:sent, [1.0]}

    # The real infer_action call runs on a background Task (real
    # inference against the real checkpoint, real wall-clock seconds --
    # see the chunk report for the measured latency) and re-enters the
    # queue via the same enqueue path :zeromq_fallback uses. Poll rather
    # than a flat sleep, since real inference latency varies.
    depth = wait_for_queue_growth(pid, 4, 20_000)

    # 5 initial - 1 popped + 50 from the real action chunk = 54.
    assert depth == 4 + 50
  end

  defp wait_for_queue_growth(pid, baseline_depth, timeout_ms) do
    deadline = System.monotonic_time(:millisecond) + timeout_ms
    poll_until_grown(pid, baseline_depth, deadline)
  end

  defp poll_until_grown(pid, baseline_depth, deadline) do
    depth = ControlLoop.queue_depth(pid)

    cond do
      depth > baseline_depth ->
        depth

      System.monotonic_time(:millisecond) > deadline ->
        flunk("queue depth never grew past #{baseline_depth} within the timeout (still #{depth})")

      true ->
        Process.sleep(50)
        poll_until_grown(pid, baseline_depth, deadline)
    end
  end
end
