defmodule SimEnvAdapter do
  @moduledoc """
  The demo's sim env adapter (demo design component 01.1, term
  `#term-sim-env-adapter` in `docs/design/demo/CONTEXT.md`; ADR-0011 and
  ADR-0012).

  A GenServer that owns the whole simulation seam: it drives ONE LeRobot/MuJoCo
  SO-101 gym env through the Python `sim_server` -- `reset` / `step` / `render`
  over a `chumak` ZeroMQ REQ socket with MessagePack framing (the same
  transport machinery as `ControlLoop.ZeroMQClient`, per ADR-0012) -- and
  exposes both interfaces the production `ControlLoop` needs:

    * as the loop's **observation source**, `observe/1` returns the current
      `%{image, image_shape, state, instruction}` observation;
    * as the loop's **actuator sink**, `actuate/2` applies one action via
      `step`, advancing the simulation.

  These are ONE unit because in a gym env producing the next observation and
  consuming the action are two halves of a single `env.step` cycle -- they
  cannot be separated.

  ## Why observe returns the last step/reset result, not a fresh render

  A gym `step` reply carries BOTH the rendered frame AND the arm's
  proprioceptive `state`; a `render` carries only the frame (it does not
  advance the sim and has no state). The design's invariant is that
  obs-out and action-in stay one coupled unit: "a `step` both applies the
  action and yields the frame the next `observe` returns, so the adapter never
  fabricates an observation divorced from the arm's real state." So `observe/1`
  returns the observation cached from the most recent `reset`/`step` -- the
  arm's real, current state paired with the frame that matches it -- rather
  than issuing a state-less `render`.

  ## The 32->6 action mapping

  SmolVLA's action-chunk actions are 32-dimensional (the model's action space),
  but the SO-101 sim env's `step` takes exactly 6 joint targets
  (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper, in
  order). `actuate/2` maps the incoming action down to those 6 by taking the
  FIRST 6 elements -- see `map_action/1`, the single place this contract lives.
  An action with fewer than 6 elements is malformed and raises (never padded).

  ## Fails loud and local

  A sim-server error (dead process, lost socket, an env that raised -> an
  `{"error", ...}` reply, or a request timeout) surfaces loud and local:
  `observe/1` and `actuate/2` RAISE rather than fabricating a blank frame or
  silently dropping an action, so a broken simulation stops the demo visibly.
  """

  use GenServer
  require Logger

  @type observation :: %{
          image: binary(),
          image_shape: {non_neg_integer(), non_neg_integer(), non_neg_integer()},
          state: [float()],
          instruction: String.t()
        }

  @type start_opt ::
          {:address, String.t()}
          | {:instruction, String.t()}
          | {:timeout_ms, pos_integer()}
          | GenServer.option()

  # The SO-101 arm's actuated joint count -- the exact length the sim's `step`
  # accepts, and the leading slice of SmolVLA's wider action vector.
  @so101_dof 6

  @default_address "tcp://localhost:5556"
  @default_instruction "pick up the cube and place it on the target"
  @default_timeout_ms 5_000

  # ------------------------------------------------------------------
  # Public API
  # ------------------------------------------------------------------

  @doc """
  Starts the adapter: connects to the sim server and issues an initial
  `reset`. Fails loud (`{:error, reason}`) if the sim server is unreachable
  at start -- a broken sim stops the demo visibly rather than starting a
  half-connected adapter.

  Options:
    * `:address` -- the sim server's ZeroMQ address (default
      `#{@default_address}`);
    * `:instruction` -- the fixed demo instruction paired with every
      observation (default a pick-and-place instruction);
    * `:timeout_ms` -- per-request round-trip bound (default
      #{@default_timeout_ms}ms).

  Any other `GenServer.start_link/3` option (e.g. `:name`) is passed through.
  """
  @spec start_link([start_opt()]) :: {:ok, pid()} | {:error, term()}
  def start_link(opts \\ []) do
    {genserver_opts, init_opts} =
      Keyword.split(opts, [:name, :timeout, :debug, :spawn_opt, :hibernate_after])

    GenServer.start_link(__MODULE__, init_opts, genserver_opts)
  end

  @doc """
  Returns the current observation -- the frame and state from the sim's most
  recent `reset`/`step`, paired with the fixed demo instruction. Raises on a
  sim-server error.
  """
  @spec observe(GenServer.server()) :: observation()
  def observe(adapter) do
    case GenServer.call(adapter, :observe) do
      {:ok, observation} -> observation
      {:error, reason} -> raise sim_error(:observe, reason)
    end
  end

  @doc """
  Applies one action: maps it 32->6 (first 6 elements) and issues a `step`,
  advancing the simulation. The next `observe/1` reflects the movement.
  Raises on a malformed action or a sim-server error.
  """
  @spec actuate(GenServer.server(), [float()]) :: :ok
  def actuate(adapter, action) do
    # Map BEFORE the GenServer call so a malformed action raises in the
    # caller's own process (a clear ArgumentError), not inside the server.
    mapped = map_action(action)

    case GenServer.call(adapter, {:actuate, mapped}) do
      :ok -> :ok
      {:error, reason} -> raise sim_error(:actuate, reason)
    end
  end

  @doc """
  The zero-arity observation-source closure `ControlLoop` wires as its
  `observation_source:` option -- `(-> observation)`, closing over `adapter`.
  """
  @spec observation_source(GenServer.server()) :: (-> observation())
  def observation_source(adapter), do: fn -> observe(adapter) end

  @doc """
  The actuator-sink closure `ControlLoop` wires as its `actuator_sink:`
  option -- `(action -> :ok)`, closing over `adapter`.
  """
  @spec actuator_sink(GenServer.server()) :: ([float()] -> :ok)
  def actuator_sink(adapter), do: fn action -> actuate(adapter, action) end

  # ------------------------------------------------------------------
  # The 32->6 action mapping -- the single place this contract lives.
  # ------------------------------------------------------------------

  # SmolVLA emits a 32-dim action; the SO-101 sim's `step` takes exactly 6
  # joint targets. The defined contract for this slice is take-the-first-6
  # (the SO-101's six controlled joints, in order). Fewer than 6 elements is a
  # malformed action -- raise, never pad. A later concern can refine this
  # mapping if the demo shows it wrong; this is the one obvious place to change.
  @spec map_action([float()]) :: [float()]
  defp map_action(action) when is_list(action) and length(action) >= @so101_dof do
    action
    |> Enum.take(@so101_dof)
    |> Enum.map(&(&1 * 1.0))
  end

  defp map_action(action) do
    raise ArgumentError,
          "malformed action: expected at least #{@so101_dof} elements " <>
            "(the SO-101's actuated joints), got #{inspect(action)}"
  end

  # ------------------------------------------------------------------
  # GenServer callbacks
  # ------------------------------------------------------------------

  defstruct [:client, :instruction, :last_observation]

  @impl true
  def init(opts) do
    address = Keyword.get(opts, :address, @default_address)
    instruction = Keyword.get(opts, :instruction, @default_instruction)
    timeout_ms = Keyword.get(opts, :timeout_ms, @default_timeout_ms)

    {host, port} = parse_address(address)

    with {:ok, client} <- SimEnvAdapter.Client.connect(host, port, timeout_ms: timeout_ms),
         {:ok, payload} <- SimEnvAdapter.Client.request(client, %{"op" => "reset"}) do
      state = %__MODULE__{
        client: client,
        instruction: instruction,
        last_observation: to_observation(payload, instruction)
      }

      {:ok, state}
    else
      {:error, reason} -> {:stop, {:sim_server_unreachable, reason}}
    end
  end

  @impl true
  def handle_call(:observe, _from, state) do
    {:reply, {:ok, state.last_observation}, state}
  end

  def handle_call({:actuate, action}, _from, state) do
    case SimEnvAdapter.Client.request(state.client, %{"op" => "step", "action" => action}) do
      {:ok, payload} ->
        observation = to_observation(payload, state.instruction)
        {:reply, :ok, %{state | last_observation: observation}}

      {:error, reason} ->
        {:reply, {:error, reason}, state}
    end
  end

  # ------------------------------------------------------------------
  # Internals
  # ------------------------------------------------------------------

  # A step/reset payload -> the standard observation map. The state length is
  # 6 (the checkpoint's max_state_dim), enforced at the sim boundary; we assert
  # it here too so a malformed observation never reaches the port.
  defp to_observation(
         %{"image" => image, "image_shape" => [h, w, c], "state" => state},
         instruction
       )
       when is_binary(image) and is_list(state) do
    # Invariant (demo design 01.1): the produced observation's state vector
    # never exceeds the checkpoint's max_state_dim -- honored at the point of
    # production so a malformed observation never reaches the infer_action
    # port. Fail loud here rather than forwarding an over-long state.
    if length(state) > @so101_dof do
      raise RuntimeError,
            "sim server returned a state vector of length #{length(state)}, " <>
              "exceeding max_state_dim (#{@so101_dof})"
    end

    %{
      image: image,
      image_shape: {h, w, c},
      state: Enum.map(state, &(&1 * 1.0)),
      instruction: instruction
    }
  end

  defp to_observation(other, _instruction) do
    raise RuntimeError,
          "sim server returned a malformed observation payload: #{inspect(other)}"
  end

  # "tcp://host:port" -> {host_charlist, port_integer}.
  defp parse_address("tcp://" <> rest) do
    case String.split(rest, ":", parts: 2) do
      [host, port] -> {to_charlist(host), String.to_integer(port)}
      _ -> raise ArgumentError, "malformed sim server address: tcp://#{rest}"
    end
  end

  defp parse_address(other) do
    raise ArgumentError, "unsupported sim server address (expected tcp://host:port): #{other}"
  end

  defp sim_error(op, reason) do
    RuntimeError.exception("sim server #{op} failed: #{format_reason(reason)}")
  end

  defp format_reason({:server_error, message}), do: message
  defp format_reason(other), do: inspect(other)
end
