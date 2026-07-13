defmodule ControlLoop do
  @moduledoc """
  Owns the bb bot's tick loop end to end (`docs/design/control-loop/design.md`
  component 01.1): a GenServer holding the current `ControlLoop.ActionQueue`,
  popping one action per tick, and triggering an `infer_action` call through
  the active adapter when the queue's depth drops below its low-water
  threshold.

  Per this context's foundation goals/invariants:

    * "Elixir owns the queue and the timing, not Python" (ADR-0002) -- the
      low-water threshold and tick timing are configured here, at
      `start_link/1`, never inherited from the Python side;
    * "Two adapters behind one port, swappable without upstream change" --
      `ControlLoop` calls `infer_action` through whichever adapter is
      configured (`adapter_module`/`adapter_client`) the same way regardless
      of which one is active. Both `:zeromq_fallback`
      (`ControlLoop.ZeroMQClient`) and `:emily_native` (`SmolVLA.Adapter`,
      backed by an in-process `SmolVLA.t()`) dispatch through the identical
      generic path -- see `start_link/1`;
    * "the queue is never read past its safe depth" -- each tick checks the
      queue's depth *before* popping. If that pre-pop depth is already below
      the low-water threshold, the next `infer_action` call is fired first,
      then the pop happens -- so a tick that would cross (or has already
      crossed) the threshold has always triggered the refill before it
      executes, matching the design's state machine
      (`queue_healthy`/`queue_low` in component 01.1's mermaid diagram)
      exactly;
    * "an action is never executed twice" -- popping and "sending" (to the
      configured `actuator_sink`) happen in the same `handle_call(:tick,
      ...)` as one atomic GenServer state transition; no other code path
      pops.

  **Not robot control logic** (no-goal): `actuator_sink` is a stub seam (a
  plain function, defaulting to a debug log) -- real bb bot
  actuator/kinematics/safety wiring is out of scope here.
  """

  use GenServer
  require Logger

  alias ControlLoop.ActionQueue

  @default_low_water_threshold 25

  @type adapter :: :emily_native | :zeromq_fallback

  @type start_opt ::
          {:adapter, adapter()}
          | {:adapter_module, module()}
          | {:adapter_client, term()}
          | {:initial_queue, ActionQueue.t()}
          | {:low_water_threshold, pos_integer()}
          | {:actuator_sink, (term() -> any())}
          | {:telemetry_sink, (term() -> any())}
          | GenServer.option()

  @doc """
  Starts a `ControlLoop`.

  Both `adapter: :zeromq_fallback` and `adapter: :emily_native` have real
  implementations -- pass `adapter_module` (e.g. `ControlLoop.ZeroMQClient`
  or `SmolVLA.Adapter`) and `adapter_client` (the value that module's own
  `infer_action/2` expects as its first argument, e.g. a connected
  `ControlLoop.ZeroMQClient.t()` or a loaded `SmolVLA.t()`). A bare
  `adapter: :emily_native` with no `adapter_module`/`adapter_client`
  supplied still returns `{:error, {:not_yet_implemented, :emily_native}}`
  rather than crashing on a missing option.

  Other options:
    * `low_water_threshold` -- the `ActionQueue` depth below which the next
      `infer_action` call is triggered (default
      #{@default_low_water_threshold}, i.e. half of SmolVLA's own default
      50-action chunk size -- mirrors SmolVLA's own reference queueing
      policy per `docs/design/control-loop/CONTEXT.md`'s "Low-water
      threshold" term; configurable per this context's "Elixir owns the
      timing" goal, never hardcoded policy);
    * `initial_queue` -- an already-populated `ActionQueue.t()` to start
      from (default: empty -- a caller starting from nothing should seed
      one synchronous `infer_action` call before the first tick, since an
      empty starting queue with `depth 0 < threshold` will fire an async
      call on tick 1 and have nothing to send that same tick);
    * `actuator_sink` -- `(action -> any())`, called once per tick with the
      popped action (default: a debug log, per the "not robot control
      logic" no-goal -- this is a stub seam, not real actuator wiring);
    * `telemetry_sink` -- `(event -> any())`, called for degraded
      conditions worth surfacing (currently just
      `{:queue_exhausted, depth}` -- default: a warning log).

  Any other `GenServer.start_link/3` option (e.g. `:name`) is passed
  through.
  """
  @spec start_link([start_opt()]) :: {:ok, pid()} | {:error, term()}
  def start_link(opts) do
    case Keyword.fetch(opts, :adapter) do
      {:ok, :zeromq_fallback} ->
        start_with_adapter(opts)

      {:ok, :emily_native} ->
        # `:emily_native` dispatches through the identical generic path as
        # `:zeromq_fallback` once a real adapter_module/adapter_client are
        # supplied (model-runtime design component 01.2's SmolVLA.Adapter,
        # e.g.) -- ControlLoop itself stays adapter-agnostic, per this
        # context's "two adapters behind one port, swappable without
        # upstream change" goal. A bare `adapter: :emily_native` with no
        # adapter wired in yet (this option's only caller before this
        # chunk) still returns the same not-yet-implemented error it
        # always has, rather than crashing on a missing adapter_module.
        if Keyword.has_key?(opts, :adapter_module) and Keyword.has_key?(opts, :adapter_client) do
          start_with_adapter(opts)
        else
          {:error, {:not_yet_implemented, :emily_native}}
        end

      {:ok, other} ->
        {:error, {:unknown_adapter, other}}

      :error ->
        {:error, :adapter_required}
    end
  end

  defp start_with_adapter(opts) do
    {genserver_opts, init_opts} =
      Keyword.split(opts, [:name, :timeout, :debug, :spawn_opt, :hibernate_after])

    GenServer.start_link(__MODULE__, init_opts, genserver_opts)
  end

  @doc "Invoked on the tick timer: pops one action, sends it, and tops up the queue if it's running low."
  @spec tick(GenServer.server()) :: :ok
  def tick(server), do: GenServer.call(server, :tick)

  @doc false
  # Test/introspection seam -- not part of the design's pinned interface,
  # only used to assert queue state from outside in tests.
  @spec queue_depth(GenServer.server()) :: non_neg_integer()
  def queue_depth(server), do: GenServer.call(server, :queue_depth)

  # ------------------------------------------------------------------
  # GenServer callbacks
  # ------------------------------------------------------------------

  defstruct [
    :queue,
    :adapter_module,
    :adapter_client,
    :low_water_threshold,
    :actuator_sink,
    :telemetry_sink,
    :infer_action_in_flight
  ]

  @impl true
  def init(opts) do
    state = %__MODULE__{
      queue: Keyword.get(opts, :initial_queue, ActionQueue.new()),
      adapter_module: Keyword.fetch!(opts, :adapter_module),
      adapter_client: Keyword.fetch!(opts, :adapter_client),
      low_water_threshold: Keyword.get(opts, :low_water_threshold, @default_low_water_threshold),
      actuator_sink: Keyword.get(opts, :actuator_sink, &default_actuator_sink/1),
      telemetry_sink: Keyword.get(opts, :telemetry_sink, &default_telemetry_sink/1),
      infer_action_in_flight: false
    }

    {:ok, state}
  end

  @impl true
  def handle_call(:tick, _from, state) do
    depth_before_pop = ActionQueue.depth(state.queue)

    state =
      if depth_before_pop < state.low_water_threshold do
        maybe_trigger_infer_action(state)
      else
        state
      end

    state =
      if ActionQueue.depth(state.queue) > 0 do
        {action, remaining_queue} = ActionQueue.pop(state.queue)
        state.actuator_sink.(action)
        %{state | queue: remaining_queue}
      else
        # the queue emptied out entirely before a triggered infer_action
        # result returned -- a real degraded condition (component 01.1's
        # "Fails" note): surfaced via telemetry, never silently ignored,
        # but never crashed on and never robot-control policy (no-goal) --
        # deciding what to *do* about it is out of scope here.
        state.telemetry_sink.({:queue_exhausted, 0})
        state
      end

    {:reply, :ok, state}
  end

  def handle_call(:queue_depth, _from, state) do
    {:reply, ActionQueue.depth(state.queue), state}
  end

  @impl true
  def handle_info({:infer_action_result, {:ok, action_chunk}}, state) do
    {:noreply,
     %{
       state
       | queue: ActionQueue.enqueue(state.queue, action_chunk),
         infer_action_in_flight: false
     }}
  end

  def handle_info({:infer_action_result, {:error, reason}}, state) do
    Logger.warning(
      "ControlLoop: infer_action failed (#{inspect(reason)}); queue keeps draining on what it already has"
    )

    {:noreply, %{state | infer_action_in_flight: false}}
  end

  # An in-flight infer_action Task completing normally also sends the
  # standard `{ref, result}` / `:DOWN` messages from `Task.async`'s
  # underlying monitor; since this GenServer does the send itself via
  # `Task.start/1` semantics (fire-and-forget, no Task.await anywhere on
  # this process), no extra monitor traffic reaches here -- see
  # `maybe_trigger_infer_action/1`.

  defp maybe_trigger_infer_action(%{infer_action_in_flight: true} = state), do: state

  defp maybe_trigger_infer_action(state) do
    server = self()
    adapter_module = state.adapter_module
    adapter_client = state.adapter_client
    observation = build_observation()

    {:ok, _pid} =
      Task.start(fn ->
        result =
          try do
            adapter_module.infer_action(adapter_client, observation)
          rescue
            error -> {:error, {:adapter_raised, error}}
          catch
            kind, reason -> {:error, {:adapter_raised, {kind, reason}}}
          end

        send(server, {:infer_action_result, result})
      end)

    %{state | infer_action_in_flight: true}
  end

  # This chunk's job is the queue/timing/adapter-dispatch seam, not
  # observation sourcing (that is the bb bot's actual camera/proprioception
  # wiring, out of scope per the "not robot control logic" no-goal) -- a
  # fixed placeholder observation stands in until a real sensor source is
  # wired up in a later chunk.
  defp build_observation do
    %{
      image: :binary.copy(<<0>>, 224 * 224 * 3),
      image_shape: {224, 224, 3},
      state: List.duplicate(0.0, 6),
      instruction: "placeholder instruction"
    }
  end

  defp default_actuator_sink(action),
    do: Logger.debug("ControlLoop: sending action #{inspect(action)}")

  defp default_telemetry_sink(event), do: Logger.warning("ControlLoop: #{inspect(event)}")
end
