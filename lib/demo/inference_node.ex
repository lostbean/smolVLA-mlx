defmodule Demo.InferenceNode do
  @moduledoc """
  The [inference node](../../docs/design/demo/CONTEXT.md#term-inference-node)
  role of the [demo rig](../../docs/design/demo/CONTEXT.md#term-demo-rig)
  (demo design component 01.2): pure assembly that stands up the production
  [inference server](../../docs/design/model-runtime/design.md) as a named,
  cluster-addressable process holding the loaded emily-native SmolVLA model.

  This module owns NO model or forward-pass logic (demo foundation invariant):
  it only starts `InferenceServer` under a name so the
  [sim node](../../docs/design/demo/CONTEXT.md#term-sim-node) can reach it as
  `{InferenceServer, inference_node}` across BEAM distribution. The server
  process, the model, and the `infer_action` port are all owned by
  model-runtime; this is the demo's boot wiring for the node's role.

  ## Distribution

  The node must already be running distributed (a `--sname`/`--name` + a
  shared cookie) for a remote caller to reach the named server. `start/1`
  registers the server under `:name` (default `InferenceServer`) but does not
  itself start distribution -- that is a boot flag (see `scripts/demo` /
  `Demo.SimNode` for the shared-cookie side).
  """

  @default_server_name InferenceServer

  @doc """
  Starts the named `InferenceServer` for this node, loading the emily-native
  SmolVLA model ONCE from `checkpoint_path` (fail-loud at start, per
  model-runtime 01.5).

  Options:
    * `:name` -- the registered server name a remote caller addresses as
      `{name, node}` (default `#{inspect(@default_server_name)}`);
    * any other option is forwarded to `InferenceServer.start_link/2`
      (e.g. `:adapter_module` + `:model` for a stubbed inference node in a
      fast test -- the only injection point, so the two-node wiring is
      testable without loading ~1GB).
  """
  @spec start(Path.t(), keyword()) :: {:ok, pid()} | {:error, term()}
  def start(checkpoint_path, opts \\ []) do
    {name, server_opts} = Keyword.pop(opts, :name, @default_server_name)
    InferenceServer.start_link(checkpoint_path, Keyword.put(server_opts, :name, name))
  end
end
