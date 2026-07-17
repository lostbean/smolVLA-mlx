defmodule Demo.ClosedLoopRealTest do
  @moduledoc """
  The REAL end-to-end demo (gated) -- demo design component 01.2 and its
  end-to-end walkthrough, driven with NOTHING faked:

    * the real Python `sim_server` (`python -m sim_server`) wrapping the actual
      MuJoCo SO-101 gym env -- real physics, a separate OS process, over ZeroMQ;
    * the real ~1.1GB `lerobot/smolvla_base` checkpoint loaded into a real
      `InferenceServer` -- real emily-native forward passes;
    * two REAL BEAM nodes: the inference server on a `:peer` node, the sim node
      (sim env adapter + production `ControlLoop`) on THIS node, joined into one
      cluster, so the cross-node `infer_action` is native BEAM distribution for
      real.

  Excluded from the fast suite; opt in with
  `RUN_SMOLVLA_INTEGRATION_CHECK=1 mix test --include real_checkpoint`
  (matching the repo convention -- see `test_helper.exs`).

  What it proves (the closed loop, end to end, no fakes):

    * criterion 1 -- the sim node calls the inference node's `InferenceServer`
      across distribution as the infer_action port;
    * criterion 2 -- a sustained run flows sim frame -> observation -> async
      cross-node infer_action -> real action chunk -> queue -> actions popped
      one per tick to the sim, and the arm's motion is observable;
    * criterion 3 -- the loop is genuinely closed: the arm's state evolves
      across the run because each action changes the state the next observation
      (and thus the next inference) is drawn from.
  """
  use ExUnit.Case, async: false

  @moduletag :real_checkpoint
  # Real model load + real MuJoCo + real forward passes across two nodes.
  @moduletag timeout: 600_000

  alias ControlLoop.ActionQueue

  @checkpoint_dir Path.expand(
                    "~/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/c83c3163b8ca9b7e67c509fffd9121e66cb96205"
                  )

  @instruction "pick up the cube and place it on the target"

  setup do
    ensure_distributed()
    :ok
  end

  test "the full closed loop runs end to end across two real nodes: real inference moves the real arm" do
    # ---- the sim seam: launch the REAL Python sim server (MuJoCo SO-101).
    sim_port = Enum.random(5600..5699)
    server_port_ref = launch_sim_server(sim_port)
    on_exit(fn -> stop_sim_server(server_port_ref) end)

    # ---- the inference node: a SECOND real BEAM node loading the REAL model.
    {:ok, peer, inference_node} = start_peer_node()
    on_exit(fn -> stop_peer_quietly(peer) end)
    load_code_on_peer(inference_node)

    # Load the real ~1.1GB checkpoint into a named InferenceServer ON the peer.
    # Configure the peer's Nx backend to the emily GPU backend first (the real
    # forward pass needs it), exactly as the real InferenceServer test does.
    :ok = configure_emily_on_peer(inference_node)

    {:ok, _srv} =
      :erpc.call(
        inference_node,
        Demo.Test.PeerHelper,
        :start_real_inference_server,
        [@checkpoint_dir],
        300_000
      )

    # ---- the sim node: the sim env adapter + production ControlLoop, wired to
    # address the InferenceServer on the peer across the (already-formed)
    # cluster. Start with an empty queue and a low-water threshold so tick 1
    # fires the first real cross-node inference.
    {:ok, %{adapter: adapter, loop: loop}} =
      Demo.SimNode.start(
        inference_node: inference_node,
        sim_address: "tcp://127.0.0.1:#{sim_port}",
        instruction: @instruction,
        sim_timeout_ms: 30_000,
        low_water_threshold: 20,
        initial_queue: ActionQueue.new()
      )

    obs_before = SimEnvAdapter.observe(adapter)

    # Drive the loop for a sustained run. A small inter-tick period gives the
    # async cross-node inference round trip time to land a chunk in the queue
    # (the first ticks pop nothing until the first real chunk arrives).
    :ok = Demo.SimNode.run_loop(loop, 60, 50)

    depth_after = ControlLoop.queue_depth(loop)

    # Criterion 2: real action chunks landed in the queue via cross-node
    # inference -- the queue is populated after the run.
    assert depth_after > 0,
           "no real action chunk ever reached the queue across the cluster"

    # Criterion 2 & 3: the arm moved and the loop is genuinely closed -- the
    # observation after the sustained run differs from the starting one, so the
    # frame/state SmolVLA saw evolved as the arm moved under its own inference.
    obs_after = SimEnvAdapter.observe(adapter)

    # Positive evidence in the test log: the queue was really refilled by
    # cross-node inference, and the arm's state really evolved.
    IO.puts("""

    [real e2e] closed loop ran across two nodes:
      inference node = #{inspect(inference_node)}
      queue depth after 60 ticks = #{depth_after} (refilled by real cross-node inference)
      arm state before = #{inspect(obs_before.state)}
      arm state after  = #{inspect(obs_after.state)}
    """)

    refute obs_after.state == obs_before.state,
           "the arm's state never changed -- the closed loop did not actually drive the sim"

    refute obs_after.image == obs_before.image,
           "the rendered frame never changed -- the sim did not advance under inference"
  end

  # ------------------------------------------------------------------
  # Real sim-server subprocess (same mechanism as sim_env_adapter_real_sim_test).
  # ------------------------------------------------------------------

  defp launch_sim_server(port) do
    repo_root = Path.expand("../..", __DIR__)
    python = Path.join(repo_root, ".venv/bin/python")

    unless File.exists?(python) do
      flunk("expected the project venv python at #{python} -- run `uv sync`")
    end

    port_ref =
      Port.open({:spawn_executable, python}, [
        :binary,
        :exit_status,
        {:args, ["-m", "sim_server", "--address", "tcp://127.0.0.1:#{port}"]},
        {:cd, to_charlist(repo_root)}
      ])

    wait_until_ready(port, 60_000)
    port_ref
  end

  defp wait_until_ready(port, timeout_ms) do
    deadline = System.monotonic_time(:millisecond) + timeout_ms
    poll_ready(port, deadline)
  end

  defp poll_ready(port, deadline) do
    if System.monotonic_time(:millisecond) > deadline do
      flunk("real sim server did not become ready within the deadline")
    end

    case probe_reset(port) do
      :ok ->
        :ok

      :not_ready ->
        Process.sleep(500)
        poll_ready(port, deadline)
    end
  end

  defp probe_reset(port) do
    {:ok, socket} = :chumak.socket(:req)
    {:ok, _} = :chumak.connect(socket, :tcp, ~c"127.0.0.1", port)

    task =
      Task.async(fn ->
        :ok = :chumak.send(socket, Msgpax.pack!(%{"op" => "reset"}, iodata: false))

        case :chumak.recv(socket) do
          {:ok, raw} ->
            case Msgpax.unpack(raw) do
              {:ok, %{"ok" => _}} -> :ok
              _ -> :not_ready
            end

          _ ->
            :not_ready
        end
      end)

    result = Task.yield(task, 2_000) || Task.shutdown(task, :brutal_kill)
    if Process.alive?(socket), do: GenServer.stop(socket, :normal, 500)

    case result do
      {:ok, :ok} -> :ok
      _ -> :not_ready
    end
  catch
    _, _ -> :not_ready
  end

  defp stop_sim_server(port_ref) do
    os_pid =
      case is_port(port_ref) and Port.info(port_ref, :os_pid) do
        {:os_pid, pid} -> pid
        _ -> nil
      end

    if os_pid do
      System.cmd("pkill", ["-9", "-P", Integer.to_string(os_pid)], stderr_to_stdout: true)
      System.cmd("kill", ["-9", Integer.to_string(os_pid)], stderr_to_stdout: true)
    end

    if is_port(port_ref) and Port.info(port_ref) != nil, do: Port.close(port_ref)
    Process.sleep(500)
    :ok
  catch
    _, _ -> :ok
  end

  # ------------------------------------------------------------------
  # Distribution scaffolding (same mechanism as inference_server tests).
  # ------------------------------------------------------------------

  defp ensure_distributed do
    _ = System.cmd("epmd", ["-daemon"], stderr_to_stdout: true)

    case :net_kernel.start([:"demo_closed_loop_real_test@127.0.0.1", :longnames]) do
      {:ok, _} -> :ok
      {:error, {:already_started, _}} -> :ok
      {:error, reason} -> flunk("could not bring test node up distributed: #{inspect(reason)}")
    end

    Node.set_cookie(:demo_closed_loop_real_cookie)
    :ok
  end

  defp start_peer_node do
    Node.set_cookie(:demo_closed_loop_real_cookie)

    {:ok, peer, node_name} =
      :peer.start_link(%{
        name: :"inference_#{System.unique_integer([:positive])}",
        host: ~c"127.0.0.1",
        longnames: true,
        args: [~c"-setcookie", ~c"demo_closed_loop_real_cookie"]
      })

    true = Node.connect(node_name)
    {:ok, peer, node_name}
  end

  defp load_code_on_peer(peer_node) do
    :erpc.call(peer_node, :code, :add_paths, [:code.get_path()])
    :erpc.call(peer_node, Application, :ensure_all_started, [:control_loop], 120_000)
    :ok
  end

  # The peer must run the real forward pass on the emily GPU backend, so set
  # its global Nx backend/compiler before the model loads (same as the real
  # InferenceServer test's setup_all does on its own node).
  defp configure_emily_on_peer(peer_node) do
    :erpc.call(peer_node, Nx, :global_default_backend, [{Emily.Backend, [device: :gpu]}])
    :erpc.call(peer_node, Nx.Defn, :global_default_options, [[compiler: Emily.Compiler]])
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
