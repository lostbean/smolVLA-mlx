defmodule InferenceServerTest do
  @moduledoc """
  Fast-suite behavioral tests for `InferenceServer` (model-runtime design
  component 01.5), asserting through its PUBLIC interface only
  (`start_link` + `infer_action`).

  The heavy real emily-native model is an external for test purposes
  (gated behind RUN_SMOLVLA_INTEGRATION_CHECK -- see
  `inference_server_real_checkpoint_test.exs`). Here a lightweight stub
  model/adapter (`InferenceServer.Test.StubInferenceAdapter`) is injected
  at the adapter seam, exercising the GenServer wrapper + the BEAM
  distribution mechanism WITHOUT loading ~1GB. What is proven with the
  stub:

    * the wrapper delegates and returns the adapter's own reply verbatim;
    * a bad checkpoint fails loud AT START (no stub needed -- the real
      loader raises on a garbage path);
    * a caller on a SECOND, separate BEAM node gets the identical result
      via `GenServer.call` to `{name, node}` -- the distribution
      mechanism itself, which is model-agnostic;
    * `max_state_dim` is rejected before the forward pass, identically
      for a local and a remote caller;
    * a remote caller that loses the connection sees a standard
      distributed `GenServer.call` exit, and the server never blocks.

  Acceptance criterion 2 (a real observation -> a well-formed chunk
  in-process) and a real-checkpoint variant of the cross-node proof live
  in the gated `inference_server_real_checkpoint_test.exs`.
  """
  use ExUnit.Case, async: false

  alias InferenceServer.Test.StubInferenceAdapter, as: Stub

  defp observation(state, instruction \\ "pick up the cube") do
    %{
      image: <<0, 0, 0>>,
      image_shape: {1, 1, 3},
      state: state,
      instruction: instruction
    }
  end

  describe "start_link: model loads once, bad checkpoint fails loud at start" do
    test "a nonexistent checkpoint path fails loud at start, not a half-loaded server" do
      # No stub: exercise the REAL loader so the fail-loud-at-start
      # behavior is the genuine one. A missing checkpoint dir means
      # SmolVLA.load/2 raises File.Error, which init/1 turns into a clean
      # start failure.
      Process.flag(:trap_exit, true)

      assert {:error, reason} =
               InferenceServer.start_link("/nonexistent/definitely/not/a/checkpoint")

      # The failure carries the loader's own error -- loud and local.
      assert match?(%File.Error{}, reason) or match?(%ArgumentError{}, reason)
    end

    test "a garbage (non-checkpoint) directory fails loud at start" do
      dir =
        Path.join(
          System.tmp_dir!(),
          "inference_server_garbage_#{System.unique_integer([:positive])}"
        )

      File.mkdir_p!(dir)
      File.write!(Path.join(dir, "config.json"), "{ this is not valid json")
      on_exit(fn -> File.rm_rf!(dir) end)

      Process.flag(:trap_exit, true)

      assert {:error, reason} = InferenceServer.start_link(dir)
      assert match?(%ArgumentError{}, reason)
    end

    test "with an injected stub model, start succeeds and holds it (no heavy load)" do
      {:ok, pid} =
        InferenceServer.start_link("ignored", adapter_module: Stub, model: Stub.model())

      assert Process.alive?(pid)
      GenServer.stop(pid)
    end
  end

  describe "infer_action: in-process delegation" do
    test "a valid observation returns a well-formed {:ok, action_chunk}" do
      {:ok, pid} =
        InferenceServer.start_link("ignored", adapter_module: Stub, model: Stub.model())

      assert {:ok, chunk} = InferenceServer.infer_action(pid, observation([1.0, 2.0, 3.0]))
      assert is_list(chunk)
      assert Enum.all?(chunk, fn row -> is_list(row) and Enum.all?(row, &is_float/1) end)

      GenServer.stop(pid)
    end

    test "the reply is the adapter's reply verbatim -- computed from the observation" do
      {:ok, pid} =
        InferenceServer.start_link("ignored", adapter_module: Stub, model: Stub.model())

      # The stub folds instruction length + state sum into the first row,
      # so equal observations give equal chunks and differing ones differ.
      {:ok, chunk_a} = InferenceServer.infer_action(pid, observation([1.0, 2.0], "go"))
      {:ok, chunk_b} = InferenceServer.infer_action(pid, observation([1.0, 2.0], "go"))
      {:ok, chunk_c} = InferenceServer.infer_action(pid, observation([9.0], "goooo"))

      assert chunk_a == chunk_b
      assert chunk_a != chunk_c

      GenServer.stop(pid)
    end
  end

  describe "max_state_dim fail-loud, local caller" do
    test "an oversized state vector is rejected before the forward pass as {:error, ...}" do
      {:ok, pid} =
        InferenceServer.start_link("ignored",
          adapter_module: Stub,
          model: Stub.model(max_state_dim: 6)
        )

      # 7 > 6: rejected. The adapter catches the ArgumentError into
      # {:error, {:smol_vla_raised, %ArgumentError{}}} -- the same
      # surfacing SmolVLA.Adapter produces for the real model.
      assert {:error, {:smol_vla_raised, %ArgumentError{} = err}} =
               InferenceServer.infer_action(pid, observation([1, 2, 3, 4, 5, 6, 7]))

      assert err.message =~ "max_state_dim"

      # A within-bound state (6) still works -- the bound is fail-loud on
      # excess, not a blanket rejection.
      assert {:ok, _chunk} = InferenceServer.infer_action(pid, observation([1, 2, 3, 4, 5, 6]))

      GenServer.stop(pid)
    end
  end

  describe "cross-node distribution (the crux) -- automated, stub model" do
    @describetag :distributed

    setup do
      # Bring THIS node up as a distributed node so it can host a named
      # server a peer reaches by {name, node}. :peer requires the parent
      # to be alive/distributed.
      ensure_distributed()
      :ok
    end

    test "a caller on a SECOND BEAM node gets the identical result via GenServer.call" do
      # Start the named server on THIS node. A named server is what a
      # remote caller addresses as {name, node} -- exactly the design's
      # `{name, remote_node}` target.
      {:ok, server} =
        InferenceServer.start_link("ignored",
          adapter_module: Stub,
          model: Stub.model(),
          name: InferenceServer
        )

      on_exit(fn -> stop_quietly(server) end)

      obs = observation([1.0, 2.0, 3.0], "reach the cube")

      # The in-process (local) result, for the identical-result assertion.
      {:ok, local_chunk} = InferenceServer.infer_action(InferenceServer, obs)

      {:ok, peer, peer_node} = start_peer_node()
      on_exit(fn -> stop_peer_quietly(peer) end)

      # Load this app's code onto the peer so it can build the observation
      # term and call InferenceServer.infer_action/2. Only the CALL crosses
      # the wire; the model + forward pass stay on this node.
      load_code_on_peer(peer_node)

      host_node = node()

      # Run the remote caller ON the peer node: it calls
      # InferenceServer.infer_action({InferenceServer, host_node}, obs) --
      # a plain GenServer.call to a {name, remote_node} target. No
      # serialization-format change at the call site: obs goes over as a
      # native BEAM term, the chunk comes back as a native BEAM term.
      remote_chunk =
        :erpc.call(peer_node, InferenceServer, :infer_action, [
          {InferenceServer, host_node},
          obs
        ])

      assert {:ok, ^local_chunk} = remote_chunk,
             "the remote node must get the byte-identical chunk the local caller got"
    end

    test "max_state_dim is rejected identically for a remote caller (local == remote)" do
      {:ok, server} =
        InferenceServer.start_link("ignored",
          adapter_module: Stub,
          model: Stub.model(max_state_dim: 6),
          name: InferenceServer
        )

      on_exit(fn -> stop_quietly(server) end)

      oversized = observation([1, 2, 3, 4, 5, 6, 7])

      # Local rejection.
      local_reply = InferenceServer.infer_action(InferenceServer, oversized)
      assert {:error, {:smol_vla_raised, %ArgumentError{}}} = local_reply

      {:ok, peer, peer_node} = start_peer_node()
      on_exit(fn -> stop_peer_quietly(peer) end)
      load_code_on_peer(peer_node)
      host_node = node()

      remote_reply =
        :erpc.call(peer_node, InferenceServer, :infer_action, [
          {InferenceServer, host_node},
          oversized
        ])

      # Identical error shape: the rejection happens inside the server
      # process before any transport, so distribution cannot change it.
      assert {:error, {:smol_vla_raised, %ArgumentError{}}} = remote_reply

      assert error_message(local_reply) == error_message(remote_reply),
             "local and remote max_state_dim rejection must be byte-identical"
    end

    test "a caller whose target node is unreachable sees a distributed call exit; server never blocks" do
      # The real server lives on THIS node, healthy.
      {:ok, server} =
        InferenceServer.start_link("ignored",
          adapter_module: Stub,
          model: Stub.model(),
          name: InferenceServer
        )

      on_exit(fn -> stop_quietly(server) end)

      # A peer is brought up, reached once, then torn down so its node is
      # unreachable -- standing in for a caller's target node that has
      # dropped out of the cluster.
      {:ok, peer, peer_node} = start_peer_node()
      load_code_on_peer(peer_node)
      host_node = node()

      # Sanity: the cross-node port works while the peer is connected.
      assert {:ok, _} =
               :erpc.call(peer_node, InferenceServer, :infer_action, [
                 {InferenceServer, host_node},
                 observation([1.0])
               ])

      # Sever the peer: stop the node, then drop the (now stale) connection
      # so `{name, peer_node}` is genuinely unreachable.
      :peer.stop(peer)
      Node.disconnect(peer_node)

      # A caller targeting a name on the now-unreachable node: a plain
      # distributed GenServer.call. It must EXIT with a standard
      # distributed failure ({:nodedown, _} / :noconnection), captured
      # here as a value -- it must NOT hang the caller. This is exactly
      # the failure a ControlLoop caller would observe if the node holding
      # InferenceServer dropped out; the caller owns what it means for the
      # tick (design 01.5 "Fails").
      call_exit =
        try do
          GenServer.call({InferenceServer, peer_node}, {:infer_action, observation([1.0])}, 1_000)
          :did_not_exit
        catch
          :exit, reason -> {:call_exited, reason}
        end

      assert match?({:call_exited, _}, call_exit),
             "a call to a name on an unreachable node must exit for the caller, got: #{inspect(call_exit)}"

      # The real server on THIS node is unharmed and still answering --
      # it never blocked on the dead peer.
      assert Process.alive?(server)
      assert {:ok, _} = InferenceServer.infer_action(InferenceServer, observation([1.0]))
    end
  end

  # ---- distribution helpers ----

  defp ensure_distributed do
    # :peer / distribution needs EPMD running; the mix-test VM does not
    # start it on its own. Start it idempotently (no-op if already up).
    _ = System.cmd("epmd", ["-daemon"], stderr_to_stdout: true)

    case :net_kernel.start([:"inference_server_test@127.0.0.1", :longnames]) do
      {:ok, _} -> :ok
      {:error, {:already_started, _}} -> :ok
      {:error, reason} -> flunk("could not bring test node up distributed: #{inspect(reason)}")
    end

    Node.set_cookie(:inference_server_test_cookie)
    :ok
  end

  defp start_peer_node do
    Node.set_cookie(:inference_server_test_cookie)

    {:ok, peer, node_name} =
      :peer.start_link(%{
        name: :"peer_#{System.unique_integer([:positive])}",
        host: ~c"127.0.0.1",
        longnames: true,
        args: [~c"-setcookie", ~c"inference_server_test_cookie"]
      })

    # :peer.start_link brings the node up but does not itself join it into
    # THIS node's cluster; connect explicitly so a {name, node} GenServer
    # call between them resolves over real BEAM distribution.
    true = Node.connect(node_name)

    {:ok, peer, node_name}
  end

  # Make this app's compiled code (InferenceServer, the stub, etc.)
  # loadable on the peer by mirroring this node's code path and ensuring
  # the app is started there. Only the CALL crosses the wire at runtime;
  # this just lets the peer resolve the module for the call site.
  defp load_code_on_peer(peer_node) do
    :erpc.call(peer_node, :code, :add_paths, [:code.get_path()])
    :erpc.call(peer_node, Application, :ensure_all_started, [:control_loop])
    :ok
  end

  # :peer.stop uses proc_lib.stop internally, which re-raises the peer
  # control process's shutdown exit if it is already going down -- swallow
  # it so the on_exit runner never crashes on cleanup.
  defp stop_peer_quietly(peer) do
    try do
      :peer.stop(peer)
    catch
      :exit, _ -> :ok
    end

    :ok
  end

  defp error_message({:error, {:smol_vla_raised, %{message: msg}}}), do: msg
  defp error_message(other), do: other

  # Best-effort server teardown that never itself crashes the on_exit
  # runner. The server is linked to the (already-terminated) test process
  # by the time on_exit runs, so GenServer.stop can observe a shutdown
  # exit and re-raise it into the cleanup callback. Kill it directly and
  # wait on a monitor instead -- a brute-force teardown that cannot exit
  # the caller.
  defp stop_quietly(server) do
    if Process.alive?(server) do
      ref = Process.monitor(server)
      Process.exit(server, :kill)

      receive do
        {:DOWN, ^ref, :process, _, _} -> :ok
      after
        1_000 -> :ok
      end
    end

    :ok
  end
end
