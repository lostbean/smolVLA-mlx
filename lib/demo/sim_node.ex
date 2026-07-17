defmodule Demo.SimNode do
  @moduledoc """
  The [sim node](../../docs/design/demo/CONTEXT.md#term-sim-node) role of the
  [demo rig](../../docs/design/demo/CONTEXT.md#term-demo-rig) (demo design
  component 01.2): the assembly that stands up the closed loop on the loop
  node and joins the [inference node](../../docs/design/demo/CONTEXT.md#term-inference-node)'s
  cluster.

  It composes three already-built parts and adds only wiring:

    * `SimEnvAdapter` -- owns the simulation seam (ZeroMQ to the Python
      [sim server](../../docs/design/demo/CONTEXT.md#term-sim-server));
    * the production `ControlLoop` + `ActionQueue` -- reused UNCHANGED, with
      its [observation source](../../docs/design/control-loop/CONTEXT.md#term-observation-source)
      bound to `SimEnvAdapter.observation_source/1` and its actuator sink to
      `SimEnvAdapter.actuator_sink/1`;
    * `Demo.InferenceClient` -- the thin distribution-addressing shim wired as
      ControlLoop's `adapter_module`/`adapter_client`, so `infer_action`
      resolves to `{InferenceServer, inference_node}` across the cluster
      ([ADR-0010](../../docs/adr/0010-beam-distribution-orthogonal-to-infer-action-port.md#adr-0010)).

  This module owns NO model or queue logic (demo foundation invariant): it is
  pure assembly and distribution wiring. Every inference call routes over the
  cluster to the inference node's `InferenceServer`; every queue operation
  lives inside the unchanged `ControlLoop`.

  ## Cluster

  `connect_cluster/2` joins the inference node by cookie + `Node.connect`; the
  local node must already be running distributed (a boot flag). `start/1`
  optionally connects first, then starts the supervision tree.
  """

  require Logger

  @doc """
  Joins the [inference node](../../docs/design/demo/CONTEXT.md#term-inference-node)
  into this node's cluster: sets the shared cookie and connects.

  Returns `:ok` on a successful connect, or `{:error, :not_connected}` if the
  target node could not be reached -- the caller decides what an unreachable
  inference node means (the loop can still start and will surface the failed
  cross-node call through ControlLoop's own degraded path). The local node
  must already be distributed for `Node.connect` to work.
  """
  @spec connect_cluster(node(), atom()) :: :ok | {:error, :not_connected}
  def connect_cluster(inference_node, cookie) when is_atom(inference_node) and is_atom(cookie) do
    Node.set_cookie(cookie)

    case Node.connect(inference_node) do
      true ->
        :ok

      _ ->
        Logger.warning(
          "Demo.SimNode: could not connect to inference node #{inspect(inference_node)}; " <>
            "the loop will start but cross-node infer_action calls will error until it joins"
        )

        {:error, :not_connected}
    end
  end

  @doc """
  Starts the sim node's loop: the `SimEnvAdapter` (connected to the sim
  server) and the production `ControlLoop`, wired so the loop's observation
  source and actuator sink are the adapter and its `infer_action` resolves to
  the `InferenceServer` on `inference_node` across the cluster.

  Required options:
    * `:inference_node` -- the BEAM node hosting the `InferenceServer`.

  Cluster options:
    * `:cookie` -- if given, `connect_cluster/2` is called first with it;
      omit to assume the caller already formed the cluster;
    * `:server_name` -- the registered `InferenceServer` name on the inference
      node (default `InferenceServer`);
    * `:infer_timeout` -- per cross-node call bound (default `:infinity`; the
      call runs off the tick loop, so it never blocks a tick).

  Sim-adapter options (forwarded to `SimEnvAdapter.start_link/1`):
    * `:sim_address` -- the sim server's ZeroMQ address
      (default `SimEnvAdapter`'s own default);
    * `:instruction` -- the fixed demo instruction;
    * `:sim_timeout_ms` -- per sim round-trip bound.
      A pre-started adapter may be injected as `:sim_adapter` instead (a fast
      test passes a `SimEnvAdapter` already bound to a fake sim server).

  Loop options (forwarded to `ControlLoop.start_link/1`):
    * `:low_water_threshold`, `:initial_queue` -- ControlLoop's own policy
      knobs (Elixir owns the timing -- never hardcoded here);
    * `:loop_name` -- register the ControlLoop under a name.

  Returns `{:ok, %{adapter: adapter_pid, loop: loop_pid}}` so a caller (the
  demo driver or a test) can tick the loop and observe the adapter.
  """
  @spec start(keyword()) :: {:ok, %{adapter: pid(), loop: pid()}} | {:error, term()}
  def start(opts) do
    inference_node = Keyword.fetch!(opts, :inference_node)

    with :ok <- maybe_connect(opts, inference_node),
         {:ok, adapter} <- start_or_reuse_adapter(opts),
         {:ok, loop} <- start_loop(opts, adapter, inference_node) do
      {:ok, %{adapter: adapter, loop: loop}}
    end
  end

  defp maybe_connect(opts, inference_node) do
    case Keyword.fetch(opts, :cookie) do
      {:ok, cookie} ->
        # A connect failure is not fatal to starting the loop -- the loop
        # degrades through ControlLoop's own path if the node never joins.
        _ = connect_cluster(inference_node, cookie)
        :ok

      :error ->
        :ok
    end
  end

  defp start_or_reuse_adapter(opts) do
    case Keyword.fetch(opts, :sim_adapter) do
      {:ok, adapter} ->
        {:ok, adapter}

      :error ->
        sim_opts =
          []
          |> put_if(opts, :sim_address, :address)
          |> put_if(opts, :instruction, :instruction)
          |> put_if(opts, :sim_timeout_ms, :timeout_ms)

        SimEnvAdapter.start_link(sim_opts)
    end
  end

  defp start_loop(opts, adapter, inference_node) do
    client =
      Demo.InferenceClient.new(inference_node,
        server_name: Keyword.get(opts, :server_name, InferenceServer),
        timeout: Keyword.get(opts, :infer_timeout, :infinity)
      )

    loop_opts =
      [
        # The emily-native adapter is the port; distribution is orthogonal
        # (ADR-0010), so from ControlLoop's view this is the emily-native
        # adapter reached through the addressing shim.
        adapter: :emily_native,
        adapter_module: Demo.InferenceClient,
        adapter_client: client,
        observation_source: SimEnvAdapter.observation_source(adapter),
        actuator_sink: SimEnvAdapter.actuator_sink(adapter)
      ]
      |> put_if(opts, :low_water_threshold, :low_water_threshold)
      |> put_if(opts, :initial_queue, :initial_queue)
      |> put_if(opts, :loop_name, :name)

    ControlLoop.start_link(loop_opts)
  end

  @doc """
  Drives the loop for `ticks` ticks at `period_ms` between ticks, calling
  `ControlLoop.tick/1` each time -- the demo's tick clock.

  This is demo-driver scaffolding, not loop logic: `ControlLoop` owns the tick
  semantics (pop, actuate, refill-when-low); this just calls `tick/1` on a
  cadence so the closed loop runs for a while and the arm's motion is
  observable. Returns `:ok` when the run finishes.

  A real demo would run this indefinitely; a bounded count keeps it a finite,
  observable run.
  """
  @spec run_loop(pid(), pos_integer(), non_neg_integer()) :: :ok
  def run_loop(loop, ticks, period_ms \\ 0)
      when is_integer(ticks) and ticks > 0 and is_integer(period_ms) and period_ms >= 0 do
    Enum.each(1..ticks, fn _ ->
      ControlLoop.tick(loop)
      if period_ms > 0, do: Process.sleep(period_ms)
    end)

    :ok
  end

  # Copy opts[from_key] into acc under to_key only when present -- lets
  # defaults live in the reused modules, not duplicated here.
  defp put_if(acc, opts, from_key, to_key) do
    case Keyword.fetch(opts, from_key) do
      {:ok, value} -> Keyword.put(acc, to_key, value)
      :error -> acc
    end
  end
end
