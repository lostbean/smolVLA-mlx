defmodule SmolVLA.Adapter do
  @moduledoc """
  The `:emily_native` adapter's client half: `ControlLoop`'s
  `adapter_module` contract (`infer_action(adapter_client, observation)
  -> {:ok, action_chunk} | {:error, reason}`), backed by an in-process
  `SmolVLA.t()` instead of a network round trip -- the emily-native
  counterpart to `ControlLoop.ZeroMQClient` (ADR-0003).

  `ControlLoop` itself is adapter-agnostic (per this context's "two
  adapters behind one port, swappable without upstream change" goal) --
  this module is the ONLY new code `ControlLoop`'s wiring needs to run
  against a real in-process SmolVLA model: `adapter_client` is a
  `SmolVLA.t()` (from `SmolVLA.load/2`), `adapter_module` is
  `SmolVLA.Adapter`.

  Converts between `ControlLoop`'s wire-agnostic observation map (the
  same shape `ControlLoop.ZeroMQClient.encode_request/1` serializes) and
  `SmolVLA.infer_action/4`'s own `(model, image, state, instruction)`
  argument shape, and between `SmolVLA.infer_action/4`'s `Nx.Tensor`
  return and `ZeroMQClient`'s own `action_chunk :: [[float()]]` shape
  (per this chunk's design-call: matching the already-accepted
  `ZeroMQClient` wire shape rather than inventing a second
  representation, since `ActionQueue.enqueue/2` and `ControlLoop`'s own
  code already only assume "a list of actions").

  Never raises past this boundary: an `SmolVLA.infer_action/4` failure
  (e.g. the "Fails" invariant's loud shape-mismatch raise) is caught and
  returned as `{:error, reason}`, matching `ZeroMQClient.infer_action/2`'s
  own "fails loud and local, never raises past the adapter boundary"
  contract -- `ControlLoop`'s own `maybe_trigger_infer_action/1` already
  wraps every adapter call in a rescue/catch as a second line of
  defense, but this module fails the same documented way `ZeroMQClient`
  does rather than relying solely on that.
  """

  alias ControlLoop.ZeroMQClient

  # The emily-native adapter is the production implementation of the
  # `InferenceServer` adapter seam (component 01.5) -- declaring the
  # behaviour makes that conformance explicit and compiler-checked. It
  # does NOT make this a second adapter (ADR-0010): the seam only names
  # the contract InferenceServer already delegated to.
  @behaviour InferenceServer.Adapter

  @type observation :: ZeroMQClient.observation()
  @type action_chunk :: ZeroMQClient.action_chunk()

  @doc """
  Runs `SmolVLA.infer_action/4` in-process against `model` (a
  `SmolVLA.t()`) for `observation`, returning `{:ok, action_chunk}` or
  `{:error, reason}` -- matching `ControlLoop.ZeroMQClient.infer_action/2`'s
  own shape exactly, so `ControlLoop` does not need to know which
  adapter is active.
  """
  @impl InferenceServer.Adapter
  @spec infer_action(SmolVLA.t(), observation()) :: {:ok, action_chunk()} | {:error, term()}
  def infer_action(%SmolVLA{} = model, %{
        image: image_binary,
        image_shape: {height, width, channels},
        state: state,
        instruction: instruction
      }) do
    image =
      image_binary
      |> Nx.from_binary(:u8)
      |> Nx.reshape({height, width, channels})

    action_chunk = SmolVLA.infer_action(model, image, state, instruction)

    {:ok, tensor_to_action_chunk(action_chunk)}
  rescue
    error -> {:error, {:smol_vla_raised, error}}
  catch
    kind, reason -> {:error, {:smol_vla_raised, {kind, reason}}}
  end

  defp tensor_to_action_chunk(action_chunk) do
    action_chunk
    |> Nx.as_type(:f32)
    |> Nx.backend_transfer(Nx.BinaryBackend)
    |> Nx.to_list()
  end
end
