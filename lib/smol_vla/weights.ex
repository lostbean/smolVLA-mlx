defmodule SmolVLA.Weights do
  @moduledoc """
  Loads a real SmolVLA (LeRobot-native) checkpoint's `model.safetensors`
  and remaps its `vlm_with_expert.*` tensor-name prefixes onto the flat
  key shape this port's own modules read from
  (`SmolVLA.Vision`/`SmolVLA.Expert`/`SmolVLA`).

  Mirrors `mlx_vlm.models.smolvla.smolvla._remap_checkpoint_weights`: an
  independent reimplementation (ADR-0004, "weights-only cross-runtime
  sharing") that reads the identical real tensor names and produces an
  identically-shaped destination key scheme (kept structurally close to
  the Python side on purpose, per this chunk's brief, so the two can be
  compared side by side).

  All tensors load onto `Emily.Backend` -- `safetensors` itself has no
  notion of `Emily.Backend`, so every tensor is `Nx.backend_transfer/2`'d
  after reading (see component 01.2: "loads the same safetensors weights
  01.1 produces or consumes -- the only artifact shared between the two
  adapters").
  """

  @expert_layer_re ~r/vlm_with_expert\.lm_expert\.layers\.(\d+)\./

  @doc """
  Reads `model.safetensors` at `path` and returns a flat
  `%{String.t() => Nx.Tensor.t()}` map, keys remapped to this port's own
  scheme, values transferred onto `Emily.Backend`.

  Raises (via `Safetensors.read!/1`'s own `File.open!`/pattern-match
  failures) loud and local on a missing or malformed file -- never a
  silent zero-initialized fallback, matching the Python side's
  `from_pretrained` contract.
  """
  @spec load!(Path.t()) :: %{String.t() => Nx.Tensor.t()}
  def load!(path) do
    {weights, _raw_key_map} = load_with_raw_keys!(path)
    weights
  end

  @doc """
  Like `load!/1`, but also returns the `remapped_key -> raw_checkpoint_key`
  map `remap/2` computes along the way -- lets a caller that needs to
  write an UPDATED checkpoint back out (`FineTuneJob`, component 01.4)
  substitute new tensor values back under the checkpoint's own original
  `model.vlm_with_expert.*` naming, so the written file is structurally
  identical (same keys, same shapes) to what a real published SmolVLA
  checkpoint -- and the Python trainer's own safetensors output -- already
  has, rather than inventing a divergent output key scheme.

  The `lm_head` tensor `remap/2` intentionally drops (never read by
  `infer_action/4`, see `remap_rest/2`'s own comment) is NOT present in
  either returned map -- a caller writing a full checkpoint back out that
  wants byte-for-byte structural parity with the ORIGINAL file (including
  the unused `lm_head`) must read it separately; `FineTuneJob` does this
  (see its own moduledoc) since "same safetensors shape as the Python
  trainer's output" is this chunk's own acceptance bar.
  """
  @spec load_with_raw_keys!(Path.t()) ::
          {%{String.t() => Nx.Tensor.t()}, %{String.t() => String.t()}}
  def load_with_raw_keys!(path) do
    unless File.exists?(path) do
      raise File.Error,
        reason: :enoent,
        action: "read (SmolVLA checkpoint)",
        path: IO.chardata_to_string(path)
    end

    raw = Safetensors.read!(path)

    if map_size(raw) == 0 do
      raise ArgumentError, "model.safetensors at #{path} contains no tensors"
    end

    remapped_entries =
      Enum.flat_map(raw, fn {raw_key, tensor} ->
        remap(raw_key, tensor) |> Enum.map(fn {new_key, value} -> {new_key, raw_key, value} end)
      end)

    weights =
      Map.new(remapped_entries, fn {new_key, _raw_key, value} ->
        {new_key, Nx.backend_transfer(value, Emily.Backend)}
      end)

    raw_key_map =
      Map.new(remapped_entries, fn {new_key, raw_key, _value} -> {new_key, raw_key} end)

    {weights, raw_key_map}
  end

  @doc """
  Counts the flow-matching action expert's transformer layers from the
  checkpoint's own weight keys -- mirrors the Python side's
  `_count_expert_layers`, since the LeRobot config.json's
  `num_expert_layers` field is legacy/unused (always 0 on real
  checkpoints; the real count only exists in the weight names).
  """
  @spec count_expert_layers(%{String.t() => Nx.Tensor.t()}) :: non_neg_integer()
  def count_expert_layers(remapped_weights) do
    remapped_weights
    |> Map.keys()
    |> Enum.flat_map(fn key ->
      case Regex.run(~r/^expert_stack\.layers\.(\d+)\./, key) do
        [_, idx] -> [String.to_integer(idx)]
        nil -> []
      end
    end)
    |> Enum.uniq()
    |> length()
  end

  defp remap("model." <> rest, value), do: remap_rest(rest, value)
  defp remap(_other, _value), do: []

  defp remap_rest("vlm_with_expert.vlm.model.vision_model." <> tail, value) do
    value =
      if tail == "embeddings.patch_embedding.weight" do
        # PyTorch conv2d weight (out, in, kH, kW) -> (out, kH, kW, in).
        Nx.transpose(value, axes: [0, 2, 3, 1])
      else
        value
      end

    [{"vision_encoder.vision_model.#{tail}", value}]
  end

  defp remap_rest(
         "vlm_with_expert.vlm.model.connector.modality_projection.proj.weight",
         value
       ) do
    [{"vision_encoder.connector.modality_projection.weight", value}]
  end

  defp remap_rest("vlm_with_expert.vlm.model.text_model.embed_tokens.weight", value) do
    [{"text_embed_tokens.weight", value}]
  end

  defp remap_rest("vlm_with_expert.vlm.model.text_model.norm.weight", value) do
    [{"expert_stack.backbone_norm.weight", value}]
  end

  defp remap_rest("vlm_with_expert.lm_expert.norm.weight", value) do
    [{"expert_stack.expert_norm.weight", value}]
  end

  defp remap_rest(
         "state_proj." <> _ = rest,
         value
       ) do
    [{rest, value}]
  end

  defp remap_rest(
         "action_in_proj." <> _ = rest,
         value
       ) do
    [{rest, value}]
  end

  defp remap_rest(
         "action_out_proj." <> _ = rest,
         value
       ) do
    [{rest, value}]
  end

  defp remap_rest(
         "action_time_mlp_in." <> _ = rest,
         value
       ) do
    [{rest, value}]
  end

  defp remap_rest(
         "action_time_mlp_out." <> _ = rest,
         value
       ) do
    [{rest, value}]
  end

  defp remap_rest(rest, value) do
    cond do
      match = Regex.run(~r/^vlm_with_expert\.vlm\.model\.text_model\.layers\.(\d+)\.(.+)$/, rest) ->
        [_, idx, tail] = match
        [{"expert_stack.layers.#{idx}.backbone.#{tail}", value}]

      match = Regex.run(@expert_layer_re, rest) ->
        [_, idx] = match
        tail = String.replace_prefix(rest, "vlm_with_expert.lm_expert.layers.#{idx}.", "")
        [{"expert_stack.layers.#{idx}.expert.#{tail}", value}]

      true ->
        # vlm_with_expert.vlm.lm_head.* -- the frozen VLM's own
        # next-token-prediction head. Never used by infer_action() (this
        # model never generates text, see the "never tokenized" and
        # "generate() is never implemented" invariants), intentionally
        # dropped here, matching the Python side's own comment.
        []
    end
  end
end
