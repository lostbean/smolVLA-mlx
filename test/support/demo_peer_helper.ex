defmodule Demo.Test.PeerHelper do
  @moduledoc """
  Runs ON a `:peer` node in the demo closed-loop test. Lives in `test/support`
  (compiled into the app) so it is loadable on the peer via the same code-path
  mirroring the other distribution tests use -- the test module's own beam is
  NOT on the peer's path, so peer-side MFAs must live here.
  """

  @doc """
  Starts the demo's named `InferenceServer` on this (peer) node via
  `Demo.InferenceNode.start/2`, then unlinks it from the transient `:erpc`
  worker that invoked this call so the server outlives the call -- standing in
  for the real node boot where the server is started from a long-lived
  supervision tree.
  """
  @spec start_inference_server(keyword()) :: {:ok, pid()} | {:error, term()}
  def start_inference_server(server_opts) do
    Demo.InferenceNode.start("ignored", server_opts)
    |> unlink_started()
  end

  @doc """
  Starts the demo's named `InferenceServer` on this (peer) node loading the
  REAL checkpoint from `checkpoint_path` -- the gated real-e2e path. Unlinks
  the server from the transient `:erpc` worker so it outlives the call.
  """
  @spec start_real_inference_server(Path.t()) :: {:ok, pid()} | {:error, term()}
  def start_real_inference_server(checkpoint_path) do
    Demo.InferenceNode.start(checkpoint_path)
    |> unlink_started()
  end

  defp unlink_started({:ok, pid}) do
    Process.unlink(pid)
    {:ok, pid}
  end

  defp unlink_started(other), do: other
end
