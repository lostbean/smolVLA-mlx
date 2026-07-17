defmodule Demo do
  @moduledoc """
  Entry points for launching the [demo rig](../docs/design/demo/CONTEXT.md#term-demo-rig)
  (demo design component 01.2) -- two BEAM nodes on one machine running the
  closed perception→action loop.

  This module owns NO model or queue logic (demo foundation invariant); it only
  composes `Demo.InferenceNode` (the inference-node role) and `Demo.SimNode`
  (the sim-node role) into the two boot flows a human runs.

  ## Running the demo (two terminals, one machine)

  First start the Python sim server (the SIM seam, ZeroMQ -- ADR-0012):

      $ .venv/bin/python -m sim_server --address 'tcp://127.0.0.1:5556'

  Then the two BEAM nodes, each `iex` distributed with the SAME cookie
  (`Node.connect` between them is native BEAM distribution -- ADR-0010):

  **Terminal A -- inference node** (loads the real checkpoint):

      $ iex --name inference@127.0.0.1 --cookie demo -S mix
      iex> Demo.start_inference_node("~/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/<snap>")

  **Terminal B -- sim node** (drives the loop, calls A across the cluster):

      $ iex --name sim@127.0.0.1 --cookie demo -S mix
      iex> {:ok, %{loop: loop}} =
      ...>   Demo.start_sim_node(inference_node: :"inference@127.0.0.1", cookie: :demo)
      iex> Demo.SimNode.run_loop(loop, 200, 50)   # 200 ticks, 50ms apart

  The arm moves in simulation under SmolVLA's own inference; each action changes
  the next frame, which drives the next inference -- the closed loop.
  """

  @doc """
  Boots the [inference node](../docs/design/demo/CONTEXT.md#term-inference-node)
  role: starts the named `InferenceServer` loading the real checkpoint. Run in
  the `iex --name inference@... --cookie <c>` session. See the moduledoc.
  """
  @spec start_inference_node(Path.t(), keyword()) :: {:ok, pid()} | {:error, term()}
  def start_inference_node(checkpoint_path, opts \\ []),
    do: Demo.InferenceNode.start(Path.expand(checkpoint_path), opts)

  @doc """
  Boots the [sim node](../docs/design/demo/CONTEXT.md#term-sim-node) role:
  connects to the inference node's cluster and starts the sim env adapter +
  production `ControlLoop`. Run in the `iex --name sim@... --cookie <c>`
  session, passing the inference node and the shared cookie. See the moduledoc
  and `Demo.SimNode.start/1` for the full option list.
  """
  @spec start_sim_node(keyword()) :: {:ok, %{adapter: pid(), loop: pid()}} | {:error, term()}
  def start_sim_node(opts), do: Demo.SimNode.start(opts)
end
