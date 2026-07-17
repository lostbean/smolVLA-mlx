defmodule InferenceServer do
  @moduledoc """
  A named GenServer that loads ONE emily-native `SmolVLA` model once (in
  `init/1`) and holds it as process state, then answers each
  `infer_action` call against it -- model-runtime design component 01.5.

  This is purely the process wrapper that makes the in-process
  emily-native adapter (`SmolVLA.Adapter`, component 01.2's client half)
  long-lived and cluster-addressable. It adds NO forward-pass logic: each
  request is a plain delegation to `SmolVLA.Adapter.infer_action/2`
  against the held `%SmolVLA{}` model.

  Because `infer_action/2` is a plain `GenServer.call`, the caller may
  target the server either as a local name/pid OR as `{name,
  remote_node}` across a BEAM cluster -- the port contract is unchanged
  by where the caller sits (ADR-0010: BEAM distribution is a
  deployment-topology axis orthogonal to the `infer_action` port, not a
  third adapter). The observation goes in and the action chunk comes out
  as native BEAM terms in both directions -- no MessagePack/ZeroMQ
  serialization at the call site; that wire format stays exclusively the
  Python fallback's.

  ## Interface (design 01.5)

      InferenceServer.start_link(checkpoint_path) :: {:ok, pid()}
      InferenceServer.infer_action(server, observation) ::
        {:ok, ActionChunk.t()} | {:error, reason}
      # server may be a local name or {name, remote_node}

  ## Fail-loud at start

  The model loads ONCE at `start_link` -- a bad or missing checkpoint
  fails there, loud and local (`init/1` stops with the load error),
  exactly as `SmolVLA.load/2` raises, never a lazily half-loaded server.

  ## max_state_dim, identical local and remote

  An observation whose state vector exceeds the checkpoint's
  `max_state_dim` is rejected before the forward pass by
  `SmolVLA.infer_action/4`'s own `ArgumentError`, which
  `SmolVLA.Adapter.infer_action/2` catches into
  `{:error, {:smol_vla_raised, %ArgumentError{}}}`. That surfacing is the
  server's reply verbatim, so a local caller and a remote caller see the
  identical `{:error, ...}` -- the rejection happens inside the server
  process, before any transport, so distribution cannot change it.
  """

  use GenServer

  @type observation :: SmolVLA.Adapter.observation()
  @type action_chunk :: SmolVLA.Adapter.action_chunk()
  @type server :: GenServer.server()

  # Held as process state: the loaded model and the adapter module that
  # runs the forward pass against it. `adapter_module` is injectable
  # (defaults to `SmolVLA.Adapter`) purely as the ports-and-adapters seam
  # for the heavy real model -- a fast test injects a lightweight stub
  # implementing the same `infer_action(model, observation) ->
  # {:ok, chunk} | {:error, reason}` contract, so the GenServer wrapper +
  # the distribution mechanism are testable without loading ~1GB. Nothing
  # in production wiring passes this option; the default IS the one
  # emily-native adapter (ADR-0010: still exactly one adapter).
  defmodule State do
    @moduledoc false
    @enforce_keys [:model, :adapter_module]
    defstruct [:model, :adapter_module]
  end

  @doc """
  Starts the server, loading the emily-native `SmolVLA` model ONCE from
  `checkpoint_path`. A bad or missing checkpoint fails loud AT START --
  `init/1` stops with the load error -- never a lazily half-loaded
  server.

  Options (all optional, passed through to `GenServer.start_link/3`):

    * `:name` -- register the server under a name (typically the module
      itself) so a remote caller reaches it as `{name, node}`.
    * `:adapter_module` -- the module running the forward pass against
      the held model; defaults to `SmolVLA.Adapter`. This is the
      ports-and-adapters seam for the heavy external model; a fast test
      injects a lightweight stub here. When it is set, `:model` may also
      be supplied directly, bypassing `SmolVLA.load/2` (the checkpoint
      path is then ignored) -- again purely so a stub model can stand in
      without loading a real checkpoint.
    * `:model` -- inject an already-built model (test seam; requires
      `:adapter_module`). Ignored in production, where the model is
      always loaded from `checkpoint_path`.

  All other options are forwarded to `GenServer.start_link/3`.
  """
  @spec start_link(Path.t(), keyword()) :: GenServer.on_start()
  def start_link(checkpoint_path, opts \\ []) do
    {init_keys, gen_opts} = Keyword.split(opts, [:adapter_module, :model])
    init_arg = Keyword.put(init_keys, :checkpoint_path, checkpoint_path)
    GenServer.start_link(__MODULE__, init_arg, gen_opts)
  end

  @doc """
  Runs one `infer_action` against the server's held model.

  `server` is anything `GenServer.call/3` accepts -- a local name/pid OR
  `{name, remote_node}` across a BEAM cluster. This is a thin wrapper
  over `GenServer.call`: the observation goes in and the action chunk
  comes back as native BEAM terms, identically wherever the caller sits.

  Returns `{:ok, action_chunk}` or `{:error, reason}`.

  A `timeout` (ms, default `:infinity`) is passed straight to
  `GenServer.call/3`. A remote caller whose cluster connection drops sees
  a standard distributed `GenServer.call` timeout/exit -- the server
  itself never blocks waiting on a dead caller (it replies and moves on).
  """
  @spec infer_action(server(), observation(), timeout()) ::
          {:ok, action_chunk()} | {:error, term()}
  def infer_action(server, observation, timeout \\ :infinity) do
    GenServer.call(server, {:infer_action, observation}, timeout)
  end

  @impl true
  def init(opts) do
    adapter_module = Keyword.get(opts, :adapter_module, SmolVLA.Adapter)

    # The model loads ONCE here, and a bad checkpoint fails loud AT START
    # (SmolVLA.load/2 raises). `init/1` returning `{:stop, reason}` turns
    # that raise into a clean start_link failure rather than a
    # half-loaded, lazily-failing server. A stub-injected `:model` (with
    # its own `:adapter_module`) skips the heavy load entirely -- the
    # test seam.
    try do
      model =
        case Keyword.fetch(opts, :model) do
          {:ok, model} -> model
          :error -> SmolVLA.load(Keyword.fetch!(opts, :checkpoint_path))
        end

      {:ok, %State{model: model, adapter_module: adapter_module}}
    rescue
      error -> {:stop, error}
    catch
      kind, reason -> {:stop, {kind, reason}}
    end
  end

  @impl true
  def handle_call({:infer_action, observation}, _from, %State{} = state) do
    # Pure delegation to the held model's adapter -- no forward-pass logic
    # here. The adapter already catches any raise (including the
    # max_state_dim ArgumentError) into `{:error, reason}`, so this reply
    # is identical for a local and a remote caller: the rejection happens
    # inside this process, before any transport.
    reply = state.adapter_module.infer_action(state.model, observation)
    {:reply, reply, state}
  end
end
