defmodule Demo.ClosedLoopTest do
  @moduledoc """
  Fast-suite behavioral tests for the demo assembly (demo design component
  01.2, the [sim node](../../docs/design/demo/CONTEXT.md#term-sim-node) wiring
  and its end-to-end walkthrough).

  This proves the WHOLE two-node closed loop WITHOUT loading the ~1GB model or
  launching MuJoCo, by putting fakes at the two -- and only the two -- true
  external boundaries:

    * the heavy SmolVLA model -> `InferenceServer` with an injected
      `StubInferenceAdapter` (the same seam `inference_server_test.exs` uses);
    * the Python sim -> `SimEnvAdapter` bound to a real ZeroMQ REP peer,
      `ControlLoop.Test.FakeSimServer` (the same stand-in
      `sim_env_adapter_test.exs` uses).

  Everything between -- `ControlLoop`, `ActionQueue`, `SimEnvAdapter`,
  `InferenceServer`, and the `Demo.*` assembly -- is the REAL code, reused
  unchanged. The inference server runs on a SECOND, separate BEAM node reached
  by `{InferenceServer, node}` over real distribution (the `:peer` mechanism
  `inference_server_test.exs` establishes), so the cross-node `infer_action`
  port is exercised for real.

  What is proven end to end here (the closed loop, driven a few ticks):

    * an action produced by (stub) inference on the OTHER node reaches the
      sim's `step` (criterion 2);
    * an observation drawn from the sim reaches inference across the node
      boundary, and it reflects the arm's prior movement (criterion 3);
    * `infer_action` fires only when the queue crosses the low-water
      threshold, and the async call never blocks the tick (criterion 4);
    * a lost connection to the inference node degrades through ControlLoop's
      own path -- the loop keeps draining its queue (criterion 6).
  """
  use ExUnit.Case, async: false

  alias ControlLoop.Test.FakeSimServer
  alias ControlLoop.ActionQueue
  alias InferenceServer.Test.StubInferenceAdapter, as: Stub

  @instruction "pick up the cube and place it on the target"

  # A stub inference chunk of full-width (>= 6) actions so the sim's 32->6
  # slice is well-formed. Every row is a valid 6+-DoF action.
  @stub_chunk for r <- 1..10, do: for(i <- 1..6, do: (r * 10 + i) * 0.01)

  setup do
    ensure_distributed()
    :ok
  end

  describe "the full closed loop across two BEAM nodes (fakes only at the two externals)" do
    test "inference on the other node drives the sim, and the moved sim feeds the next inference" do
      # ---- external #1: the sim, a real ZeroMQ REP peer standing in for Python.
      {:ok, sim_server} = FakeSimServer.start_link()
      sim_adapter = start_sim_adapter(sim_server)

      # ---- external #2: the heavy model, a stub adapter inside a REAL
      # InferenceServer, running on a SECOND BEAM node.
      {peer, inference_node} = start_inference_node_on_peer()

      # ---- the assembly under test: the sim node wires the real ControlLoop
      # to the sim adapter's two seams and addresses the inference server on
      # the peer via the thin shim. The cluster is already formed (the peer
      # connected in start_inference_node_on_peer).
      {:ok, %{loop: loop}} =
        Demo.SimNode.start(
          inference_node: inference_node,
          sim_adapter: sim_adapter,
          # Start with an empty queue and a low-water threshold above 0 so the
          # very first tick sees depth 0 < threshold and fires inference.
          low_water_threshold: 3,
          initial_queue: ActionQueue.new()
        )

      # Tick 1: queue is empty (0 < 3) -> fires async cross-node infer_action;
      # the result comes back asynchronously and is enqueued. Drive a few
      # ticks so the round trip completes and actions start flowing to the sim.
      obs_before = SimEnvAdapter.observe(sim_adapter)

      :ok = Demo.SimNode.run_loop(loop, 8, 5)

      # The queue was refilled by inference across the cluster: it is no longer
      # empty (the stub returned a 10-row chunk; 8 ticks popped at most 8).
      assert ControlLoop.queue_depth(loop) > 0

      # Criterion 2: an action produced by inference reached the sim's step --
      # the fake sim recorded at least one `step` request.
      requests = FakeSimServer.requests(sim_server)
      assert Enum.any?(requests, &match?(%{"op" => "step"}, &1))

      # Criterion 3: the loop is genuinely closed -- the sim advanced, so the
      # observation the next inference will see differs from the starting one.
      obs_after = SimEnvAdapter.observe(sim_adapter)
      refute obs_after.state == obs_before.state
      refute obs_after.image == obs_before.image

      stop_peer_quietly(peer)
    end

    test "the observation that crossed to the inference node reflects the arm's prior movement" do
      # This nails criterion 3 at the inference seam itself: the stub folds the
      # observation's state sum into its returned chunk's first row, so we can
      # read back WHAT observation inference saw. After the sim has moved, the
      # observation carried across the node boundary must be the MOVED one.
      {:ok, sim_server} = FakeSimServer.start_link()
      sim_adapter = start_sim_adapter(sim_server)

      # Move the sim first (via the real adapter) so its cached observation is
      # non-zero state, then confirm inference on the peer computes from it.
      full_action = for i <- 1..32, do: i * 1.0
      :ok = SimEnvAdapter.actuate(sim_adapter, full_action)
      moved_obs = SimEnvAdapter.observe(sim_adapter)
      assert Enum.sum(moved_obs.state) != 0.0

      {peer, inference_node} = start_inference_node_on_peer(mode: :marker)

      # Call infer_action across the cluster with the MOVED observation, the
      # exact term ControlLoop's observation_source would hand the shim.
      client = Demo.InferenceClient.new(inference_node)
      assert {:ok, chunk} = Demo.InferenceClient.infer_action(client, moved_obs)

      # The stub's marker row = [instruction_length, state_sum]. The state_sum
      # it computed remotely equals the MOVED sim's state sum -- proving the
      # post-movement observation is what crossed the wire and drove inference.
      [[_instr_len, state_sum] | _] = chunk
      assert_in_delta state_sum, Enum.sum(moved_obs.state), 1.0e-9

      stop_peer_quietly(peer)
    end

    test "infer_action fires only when the queue is below low-water; a full queue does not call inference" do
      {:ok, sim_server} = FakeSimServer.start_link()
      sim_adapter = start_sim_adapter(sim_server)
      {peer, inference_node} = start_inference_node_on_peer()

      # Seed the queue ABOVE the low-water threshold so ticks pop-and-actuate
      # without ever triggering a cross-node inference call.
      full_queue =
        Enum.reduce(1..10, ActionQueue.new(), fn r, q ->
          ActionQueue.enqueue(q, [for(i <- 1..6, do: (r + i) * 1.0)])
        end)

      {:ok, %{loop: loop}} =
        Demo.SimNode.start(
          inference_node: inference_node,
          sim_adapter: sim_adapter,
          low_water_threshold: 3,
          initial_queue: full_queue
        )

      # 5 ticks: depth starts at 10, stays >= 3 the whole time (10 -> 5), so no
      # inference is ever triggered. Actions still flow to the sim each tick.
      :ok = Demo.SimNode.run_loop(loop, 5, 0)

      # The inference server on the peer never received a request: assert via
      # the sim -- exactly 5 `step`s (one per tick), and the queue drained by 5
      # (10 - 5 = 5) with no refill.
      steps = Enum.count(FakeSimServer.requests(sim_server), &match?(%{"op" => "step"}, &1))
      assert steps == 5
      assert ControlLoop.queue_depth(loop) == 5

      stop_peer_quietly(peer)
    end

    test "the async cross-node call never blocks the tick loop -- ticks keep draining while a call is in flight" do
      # A deliberately SLOW inference server on the peer: the stub sleeps before
      # replying. Ticks must keep popping the already-queued actions during the
      # in-flight call, never stalling on it.
      {:ok, sim_server} = FakeSimServer.start_link()
      sim_adapter = start_sim_adapter(sim_server)
      {peer, inference_node} = start_inference_node_on_peer(slow_ms: 400)

      # Seed a queue with a few actions AND a low-water threshold high enough
      # that tick 1 triggers a (slow) inference while there is still stock to
      # drain.
      seed =
        Enum.reduce(1..4, ActionQueue.new(), fn r, q ->
          ActionQueue.enqueue(q, [for(i <- 1..6, do: (r + i) * 1.0)])
        end)

      {:ok, %{loop: loop}} =
        Demo.SimNode.start(
          inference_node: inference_node,
          sim_adapter: sim_adapter,
          low_water_threshold: 10,
          initial_queue: seed
        )

      # Fire 4 quick ticks with NO delay. The slow (400ms) inference is still
      # in flight the whole time; each tick must still return promptly and pop
      # an action to the sim. Time the whole burst: it must be far under the
      # 400ms the inference call takes -- proving the ticks did not block on it.
      {elapsed_us, :ok} =
        :timer.tc(fn -> Demo.SimNode.run_loop(loop, 4, 0) end)

      assert elapsed_us < 400_000,
             "4 ticks took #{div(elapsed_us, 1000)}ms -- they blocked on the in-flight inference call"

      # All 4 seeded actions reached the sim while the call was still pending.
      steps = Enum.count(FakeSimServer.requests(sim_server), &match?(%{"op" => "step"}, &1))
      assert steps == 4

      stop_peer_quietly(peer)
    end

    test "a lost inference node degrades through ControlLoop's own path -- the loop keeps draining its queue" do
      {:ok, sim_server} = FakeSimServer.start_link()
      sim_adapter = start_sim_adapter(sim_server)
      {peer, inference_node} = start_inference_node_on_peer()

      # Seed a queue so the loop has stock to drain after the node drops.
      seed =
        Enum.reduce(1..6, ActionQueue.new(), fn r, q ->
          ActionQueue.enqueue(q, [for(i <- 1..6, do: (r + i) * 1.0)])
        end)

      {:ok, %{loop: loop}} =
        Demo.SimNode.start(
          inference_node: inference_node,
          sim_adapter: sim_adapter,
          # Low threshold so the first ticks pop without triggering inference.
          low_water_threshold: 2,
          initial_queue: seed,
          infer_timeout: 500
        )

      # Sever the inference node.
      stop_peer_quietly(peer)
      Node.disconnect(inference_node)

      # The loop keeps ticking: it drains its existing 6-deep queue. Even once
      # depth crosses the low-water threshold and it TRIES a cross-node call,
      # that call errors (the shim catches the distributed exit into
      # {:error, ...}) and ControlLoop logs-and-continues -- never crashes.
      assert :ok = Demo.SimNode.run_loop(loop, 6, 0)

      # The queue drained (6 actions -> 6 steps to the sim); the loop survived
      # a failed cross-node call with no demo-specific failure mode.
      steps = Enum.count(FakeSimServer.requests(sim_server), &match?(%{"op" => "step"}, &1))
      assert steps == 6
      assert Process.alive?(loop)
    end

    test "the shim translates a distributed-call exit into {:error, ...} rather than raising" do
      # Direct unit-level assertion on the shim's failure contract: a call to a
      # name on an unreachable node must not raise past ControlLoop's adapter
      # boundary; it returns {:error, reason}.
      {peer, inference_node} = start_inference_node_on_peer()
      stop_peer_quietly(peer)
      Node.disconnect(inference_node)

      client = Demo.InferenceClient.new(inference_node, timeout: 500)

      obs = %{
        image: <<0, 0, 0>>,
        image_shape: {1, 1, 3},
        state: [1.0],
        instruction: "x"
      }

      assert {:error, {:inference_node_unreachable, _reason}} =
               Demo.InferenceClient.infer_action(client, obs)
    end
  end

  # ------------------------------------------------------------------
  # Helpers -- the two-node + fake-sim scaffolding.
  # ------------------------------------------------------------------

  defp start_sim_adapter(sim_server) do
    port = FakeSimServer.port(sim_server)

    {:ok, adapter} =
      SimEnvAdapter.start_link(address: "tcp://localhost:#{port}", instruction: @instruction)

    adapter
  end

  # Bring up a second BEAM node and start a REAL InferenceServer there with a
  # stub adapter injected -- so the cross-node port is exercised for real but
  # no heavy model loads.
  #
  # `:mode` selects the stub's chunk shape:
  #   * `:actions` (default) -- `FixedChunkAdapter`, every row a valid 6-DoF
  #     action, so a full-loop drive test can pop every row into the sim;
  #   * `:marker` -- `StubInferenceAdapter`, whose first row folds in the
  #     observation's state sum, so a test can read back what observation
  #     crossed the node boundary (criterion 3 at the inference seam).
  # `:slow_ms` makes the adapter sleep before replying (non-blocking-tick test).
  defp start_inference_node_on_peer(opts \\ []) do
    {:ok, peer, peer_node} = start_peer_node()
    load_code_on_peer(peer_node)

    mode = Keyword.get(opts, :mode, :actions)
    slow_ms = Keyword.get(opts, :slow_ms, 0)
    {adapter_module, model} = stub_adapter_and_model(mode, slow_ms)

    # `:erpc.call` runs the MFA in a TEMPORARY process that exits when the call
    # returns; a `start_link`ed server would die WITH it (a `:noproc` at the
    # next cross-node call). So on the peer we start it from a persistent owner
    # process and unlink, standing in for the node's real (supervised) boot
    # where `Demo.InferenceNode.start/1` is called from a long-lived tree.
    {:ok, _pid} =
      :erpc.call(peer_node, Demo.Test.PeerHelper, :start_inference_server, [
        [adapter_module: adapter_module, model: model]
      ])

    {peer, peer_node}
  end

  # The (adapter_module, model) pair for a mode. A non-zero slow_ms selects the
  # slow wrapper (only defined for the actions mode -- the one the timing test
  # uses). All these modules live on the peer's code path (test/support,
  # compiled into the app; loaded via load_code_on_peer).
  defp stub_adapter_and_model(:actions, 0),
    do: {Demo.Test.FixedChunkAdapter, Demo.Test.FixedChunkAdapter.model(@stub_chunk)}

  defp stub_adapter_and_model(:actions, _slow_ms),
    do: {Demo.Test.SlowFixedChunkAdapter, Demo.Test.FixedChunkAdapter.model(@stub_chunk)}

  defp stub_adapter_and_model(:marker, 0),
    do: {Stub, Stub.model(action_chunk: @stub_chunk, max_state_dim: 6)}

  # ---- distribution scaffolding (mirrors inference_server_test.exs) ----

  defp ensure_distributed do
    _ = System.cmd("epmd", ["-daemon"], stderr_to_stdout: true)

    case :net_kernel.start([:"demo_closed_loop_test@127.0.0.1", :longnames]) do
      {:ok, _} -> :ok
      {:error, {:already_started, _}} -> :ok
      {:error, reason} -> flunk("could not bring test node up distributed: #{inspect(reason)}")
    end

    Node.set_cookie(:demo_closed_loop_cookie)
    :ok
  end

  defp start_peer_node do
    Node.set_cookie(:demo_closed_loop_cookie)

    {:ok, peer, node_name} =
      :peer.start_link(%{
        name: :"peer_#{System.unique_integer([:positive])}",
        host: ~c"127.0.0.1",
        longnames: true,
        args: [~c"-setcookie", ~c"demo_closed_loop_cookie"]
      })

    true = Node.connect(node_name)
    {:ok, peer, node_name}
  end

  defp load_code_on_peer(peer_node) do
    :erpc.call(peer_node, :code, :add_paths, [:code.get_path()])
    :erpc.call(peer_node, Application, :ensure_all_started, [:control_loop])
    :ok
  end

  defp stop_peer_quietly(peer) do
    try do
      :peer.stop(peer)
    catch
      :exit, _ -> :ok
    end

    :ok
  end
end
