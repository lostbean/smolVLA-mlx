defmodule Demo.Test.FixedChunkAdapter do
  @moduledoc """
  A minimal `InferenceServer.Adapter` for the demo closed-loop drive tests: it
  returns a fixed, well-formed action chunk whose every row is a valid
  6+-DoF action (so the SO-101 sim's 32->6 slice is well-formed on every popped
  action).

  Unlike `InferenceServer.Test.StubInferenceAdapter`, it prepends NO
  observation-marker row -- that stub's 2-wide marker is a valid probe when a
  test inspects the chunk directly, but it is not a valid ACTION for the sim to
  step, so a full-loop drive test that pops every row needs an all-actions
  chunk. Inference itself remains a true external; this is just the shape the
  sim consumes.

  The "model" it operates on is `%{action_chunk: [[float()]]}`.
  """

  @behaviour InferenceServer.Adapter

  @doc "Builds a model whose infer_action returns `chunk` verbatim."
  def model(chunk), do: %{action_chunk: chunk}

  @impl true
  def infer_action(%{action_chunk: chunk}, %{
        image: _,
        image_shape: {_, _, _},
        state: _,
        instruction: _
      }) do
    {:ok, chunk}
  end
end
