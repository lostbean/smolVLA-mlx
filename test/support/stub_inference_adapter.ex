defmodule InferenceServer.Test.StubInferenceAdapter do
  @moduledoc """
  A lightweight stand-in for the heavy emily-native model at the
  `InferenceServer` adapter seam. Implements the SAME
  `infer_action(model, observation) -> {:ok, chunk} | {:error, reason}`
  contract as `SmolVLA.Adapter`, so it can be injected via
  `InferenceServer.start_link(_, adapter_module: __MODULE__, model: ...)`
  to test the GenServer wrapper + the BEAM-distribution mechanism WITHOUT
  loading ~1GB of real weights.

  The "model" it operates on is a plain map carrying:

    * `:max_state_dim` -- mimics the real checkpoint's `config.max_state_dim`
      bound. An observation whose `state` exceeds it is rejected BEFORE any
      "forward pass", the same fail-loud shape the real
      `SmolVLA.infer_action/4` produces: it raises `ArgumentError`, which
      this adapter catches into `{:error, {:smol_vla_raised, error}}` --
      byte-identical surfacing to `SmolVLA.Adapter`'s own catch clause, so a
      test asserts against the real error shape, not a stub-specific one.
    * `:action_chunk` -- the well-formed `[[float()]]` chunk to return on a
      valid observation. Defaults to a small fixed chunk. Making it depend
      on the observation's `instruction`/`state` lets a distribution test
      prove the REMOTE reply is computed from the REMOTELY-passed
      observation (not a constant), i.e. the observation really crossed the
      node boundary as a native term.
  """

  @behaviour InferenceServer.Adapter

  @default_chunk [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

  @doc "Builds a stub model map with the given `max_state_dim` and optional fixed chunk."
  def model(opts \\ []) do
    %{
      max_state_dim: Keyword.get(opts, :max_state_dim, 6),
      action_chunk: Keyword.get(opts, :action_chunk, @default_chunk)
    }
  end

  @impl true
  def infer_action(model, %{
        image: _image,
        image_shape: {_h, _w, _c},
        state: state,
        instruction: instruction
      }) do
    state_dim = length(state)

    if state_dim > model.max_state_dim do
      raise ArgumentError,
            "infer_action/4 got a state vector of dimensionality #{state_dim}, which " <>
              "exceeds this checkpoint's max_state_dim=#{model.max_state_dim}."
    end

    # Fold the instruction length and the state sum into the returned
    # chunk so a distribution test can prove the reply was computed from
    # the observation that crossed the node boundary, not a constant.
    marker = [String.length(instruction) * 1.0, Enum.sum(state) * 1.0]
    {:ok, [marker | model.action_chunk]}
  rescue
    error -> {:error, {:smol_vla_raised, error}}
  catch
    kind, reason -> {:error, {:smol_vla_raised, {kind, reason}}}
  end
end
