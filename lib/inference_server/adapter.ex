defmodule InferenceServer.Adapter do
  @moduledoc """
  The adapter-seam contract `InferenceServer` delegates each
  `infer_action` request to. This is the ports-and-adapters seam for the
  heavy external model (model-runtime component 01.2): the server holds a
  loaded model and an adapter module implementing this behaviour, and each
  request is a plain call to `infer_action(model, observation)`.

  The production implementation is `SmolVLA.Adapter` (the emily-native
  adapter's client half); a fast test injects a lightweight stub
  implementing the same callback, so the GenServer wrapper and the BEAM
  distribution mechanism are testable without loading ~1GB of weights.

  ADR-0010: naming this seam does NOT introduce a new adapter --
  `SmolVLA.Adapter` remains exactly one adapter. It only makes explicit
  the contract `InferenceServer` was already calling, so a stub can stand
  in at the seam.
  """

  @typedoc "One observation: image bytes + shape, robot state, instruction."
  @type observation :: SmolVLA.Adapter.observation()

  @typedoc "One action chunk: a list of action rows."
  @type action_chunk :: SmolVLA.Adapter.action_chunk()

  @doc """
  Runs one `infer_action` against `model`, returning `{:ok, action_chunk}`
  or `{:error, reason}`. Must never raise past this boundary -- a
  forward-pass failure (including the `max_state_dim` fail-loud raise) is
  caught and returned as `{:error, reason}`.
  """
  @callback infer_action(model :: term(), observation()) ::
              {:ok, action_chunk()} | {:error, term()}
end
