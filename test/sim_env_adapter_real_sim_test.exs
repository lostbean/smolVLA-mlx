defmodule SimEnvAdapterRealSimTest do
  @moduledoc """
  Gated, real-sim tests for `SimEnvAdapter` (demo design component 01.1).

  These launch the ACTUAL Python `sim_server` (`python -m sim_server`) wrapping
  the real MuJoCo SO-101 gym env -- real physics, real wall-clock, a separate
  OS process -- so they are excluded from the fast default suite and opt in via
  `RUN_SMOLVLA_INTEGRATION_CHECK=1 mix test --include real_checkpoint`,
  matching this repo's convention (see `test_helper.exs`).

  What these prove against the REAL sim:
    * acceptance criterion 1 -- `observe/1` returns a well-formed observation
      (480x640x3 frame + 6-DoF state) from the real env;
    * acceptance criterion 2 & 4 -- a canned sequence of `actuate/2` calls
      advances the real simulation and the arm's motion is observable (the
      returned observation changes), standalone with no ControlLoop present.
  """
  use ExUnit.Case, async: false

  @moduletag :real_checkpoint
  # Launching the MuJoCo env + real physics steps: generous wall-clock.
  @moduletag timeout: 180_000

  @instruction "pick up the cube and place it on the target"

  setup do
    # A per-test port so two sequential tests never race to rebind one fixed
    # port (a real OS process holding a bound socket does not release it
    # instantly on kill).
    port = Enum.random(5600..5699)
    server_port = launch_sim_server(port)
    on_exit(fn -> stop_sim_server(server_port) end)

    {:ok, adapter} =
      SimEnvAdapter.start_link(
        address: "tcp://127.0.0.1:#{port}",
        instruction: @instruction,
        timeout_ms: 30_000
      )

    %{adapter: adapter}
  end

  test "criterion 1: observe/1 returns a well-formed real observation", %{adapter: adapter} do
    obs = SimEnvAdapter.observe(adapter)

    assert %{
             image: image,
             image_shape: {480, 640, 3},
             state: state,
             instruction: @instruction
           } = obs

    assert is_binary(image)
    assert byte_size(image) == 480 * 640 * 3
    assert length(state) == 6
    assert Enum.all?(state, &is_float/1)
  end

  test "criterion 2 & 4: a canned actuate sequence moves the real arm (observation changes), no ControlLoop",
       %{adapter: adapter} do
    before = SimEnvAdapter.observe(adapter)

    # A canned sequence of 32-dim actions (the leading 6 drive the SO-101).
    # A non-trivial joint command so the physics actually moves the arm.
    states =
      for k <- 1..5 do
        action = for i <- 1..32, do: 0.3 * :math.sin(k + i * 0.1)
        :ok = SimEnvAdapter.actuate(adapter, action)
        SimEnvAdapter.observe(adapter).state
      end

    last = List.last(states)

    # The arm genuinely moved: the final state differs from the initial one,
    # and the sequence produced more than one distinct state.
    refute last == before.state
    assert length(Enum.uniq([before.state | states])) > 1
  end

  # ------------------------------------------------------------------
  # Real sim-server subprocess management.
  # ------------------------------------------------------------------

  # Launch `python -m sim_server` on a fixed port as a Port, then poll a raw
  # ZeroMQ REQ probe until it answers a `reset` -- so the adapter's own connect
  # is not racing MuJoCo's ~1s startup.
  defp launch_sim_server(port) do
    repo_root = Path.expand("..", __DIR__)
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

  # A raw chumak REQ probe: succeeds once the real server answers a reset.
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
    # Terminate the real subprocess and free the port before setup rebinds it.
    # SIGKILL (not SIGTERM): MuJoCo's teardown can ignore/trap SIGTERM and
    # leave the socket bound, orphaning the port -- a test teardown wants the
    # process gone, not a graceful shutdown. `pkill -P` also sweeps any child
    # the `python -m` launcher spawned so nothing keeps the port held.
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

    # Give the OS a moment to release the bound port before setup rebinds it.
    Process.sleep(500)
    :ok
  catch
    _, _ -> :ok
  end
end
