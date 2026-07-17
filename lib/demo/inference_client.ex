defmodule Demo.InferenceClient do
  @moduledoc """
  The [sim node](../../docs/design/demo/CONTEXT.md#term-sim-node)'s
  distribution-addressing shim (demo design component 01.2): the module the
  production `ControlLoop` calls through its `adapter_module`/`adapter_client`
  seam, whose only job is to ADDRESS the
  [inference server](../../docs/design/model-runtime/design.md) sitting on the
  other BEAM node and forward the call.

  It implements ControlLoop's adapter contract --
  `infer_action(client, observation) -> {:ok, chunk} | {:error, reason}` --
  but adds NO inference logic. The `client` value ControlLoop holds is a
  `%Demo.InferenceClient{}` carrying the inference node's name and the
  registered server name; `infer_action/2` turns it into the `{name, node}`
  target and calls `InferenceServer.infer_action/2`.

  Per [ADR-0010](../../docs/adr/0010-beam-distribution-orthogonal-to-infer-action-port.md#adr-0010),
  BEAM distribution is a deployment-topology axis orthogonal to the
  `infer_action` port -- it is NOT a third adapter. This shim is not a new
  adapter either: it is pure addressing of a remote process. The one and only
  emily-native adapter still lives inside the `InferenceServer` on the
  inference node; only the `GenServer.call` crosses the wire, as a native BEAM
  term in both directions (no MessagePack, no ZeroMQ -- that transport stays
  the Python fallback's, and the sim seam's).

  Because the whole point of this shim is remote addressing, a lost cluster
  connection to the inference node is NOT swallowed here: the distributed
  `GenServer.call` failure surfaces to `ControlLoop`, which owns what a failed
  `infer_action` means for the tick (its async trigger catches it into
  `{:error, ...}` and keeps draining the queue it already has -- demo design
  01.2 "Fails", control-loop design 01.1). This shim only translates the exit
  into an `{:error, reason}` return so ControlLoop's adapter contract (never
  raise past this boundary) holds identically to the in-process adapters.
  """

  @enforce_keys [:inference_node]
  defstruct [:inference_node, server_name: InferenceServer, timeout: :infinity]

  @type t :: %__MODULE__{
          inference_node: node(),
          server_name: atom(),
          timeout: timeout()
        }

  @type observation :: InferenceServer.observation()
  @type action_chunk :: InferenceServer.action_chunk()

  @doc """
  Builds the `adapter_client` value `ControlLoop` holds for this shim.

    * `inference_node` -- the BEAM node hosting the `InferenceServer`
      (e.g. `:"inference@127.0.0.1"`);
    * `:server_name` -- the registered name of the server on that node
      (default `InferenceServer`);
    * `:timeout` -- the per-call bound passed to `InferenceServer.infer_action/3`
      (default `:infinity`, matching the server's own default -- the async
      trigger runs off the tick loop, so a slow call never blocks a tick).
  """
  @spec new(node(), keyword()) :: t()
  def new(inference_node, opts \\ []) when is_atom(inference_node) do
    %__MODULE__{
      inference_node: inference_node,
      server_name: Keyword.get(opts, :server_name, InferenceServer),
      timeout: Keyword.get(opts, :timeout, :infinity)
    }
  end

  @doc """
  ControlLoop's `adapter_module` entry point. Resolves the client to the
  `{server_name, inference_node}` target and forwards to
  `InferenceServer.infer_action/3` across the cluster.

  A distributed-call failure (the inference node dropped out of the cluster,
  or the call timed out) is caught into `{:error, reason}` so this never
  raises past ControlLoop's adapter boundary -- ControlLoop then logs it and
  keeps draining its existing queue (demo design 01.2 "Fails").
  """
  @spec infer_action(t(), observation()) :: {:ok, action_chunk()} | {:error, term()}
  def infer_action(%__MODULE__{} = client, observation) do
    target = {client.server_name, client.inference_node}
    InferenceServer.infer_action(target, observation, client.timeout)
  catch
    :exit, reason -> {:error, {:inference_node_unreachable, reason}}
  end
end
