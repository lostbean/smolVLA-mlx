defmodule SmolVLA do
  @moduledoc """
  SmolVLA: a frozen SmolVLM2 vision-language backbone plus a
  flow-matching action expert that cross-attends into it, running
  entirely through `emily`'s `Nx.Backend` (no Python process in this
  path). Mirrors `mlx_vlm.models.smolvla.smolvla.SmolVLAModel` --
  an independent reimplementation against `Nx`/`emily` (ADR-0004), kept
  structurally close to the Python module's own shape on purpose.

  Per `docs/design/model-runtime/design.md` component 01.2 (exact
  interface):

      SmolVLA.load(checkpoint_path) :: SmolVLA.t()
      SmolVLA.infer_action(model, image, state, instruction) :: ActionChunk.t()

  `load/1` loads a real SmolVLA (LeRobot-native) checkpoint's
  `config.json` + `model.safetensors` (see `SmolVLA.Config`,
  `SmolVLA.Weights`) and the instruction tokenizer (see
  `SmolVLA.Tokenizer`). `infer_action/4` runs the full forward pass:
  vision encoding (`SmolVLA.Vision`) + language-token embedding build the
  VLM "prefix", the robot state compresses to one token appended to the
  prefix, then a flow-matching Euler loop runs the action expert
  ("suffix") through `SmolVLA.Expert`'s joint attention with the frozen
  prefix to produce one action chunk -- a real continuous `Nx.Tensor`,
  never sampled from a vocabulary (the "action expert's output is never
  tokenized" invariant).

  **Weights-only cross-runtime sharing** (ADR-0004): this module and the
  Python `SmolVLAModel` are two independently-implemented forward passes
  sharing no code, only the checkpoint's real weight file.
  """

  alias SmolVLA.Config
  alias SmolVLA.Expert
  alias SmolVLA.Preprocessing
  alias SmolVLA.Tokenizer
  alias SmolVLA.Vision
  alias SmolVLA.Weights

  @enforce_keys [:config, :weights, :tokenizer]
  defstruct [:config, :weights, :tokenizer]

  @type t :: %__MODULE__{
          config: Config.t(),
          weights: %{String.t() => Nx.Tensor.t()},
          tokenizer: Tokenizers.Tokenizer.t()
        }

  @doc """
  Loads a real SmolVLA (LeRobot-native) checkpoint: `config.json` (flat,
  `type`-keyed, see ADR-0006) plus `model.safetensors` (LeRobot's own
  `vlm_with_expert.*` tensor-name prefixes, see `SmolVLA.Weights`), plus
  the VLM backbone's own instruction tokenizer (`tokenizer_path`,
  typically `tokenizer.json` inside the HF cache for
  `config.vlm_model_name` -- see `SmolVLA.Tokenizer`; this port has no HF
  Hub network-fetch logic, so the caller resolves and supplies that path
  directly, unlike the Python side's own network-fetching
  `AutoTokenizer.from_pretrained`).

  Raises loud and local -- never a silent zero-initialized fallback --
  on a missing or malformed checkpoint, matching 01.1's own
  `from_pretrained` contract.
  """
  @spec load(Path.t(), keyword()) :: t()
  def load(checkpoint_dir, opts \\ []) do
    config_path = Path.join(checkpoint_dir, "config.json")

    unless File.exists?(config_path) do
      raise File.Error,
        reason: :enoent,
        action: "read (SmolVLA config.json)",
        path: IO.chardata_to_string(config_path)
    end

    raw_config =
      case Jason.decode(File.read!(config_path)) do
        {:ok, decoded} ->
          decoded

        {:error, reason} ->
          raise ArgumentError, "Malformed config.json at #{config_path}: #{inspect(reason)}"
      end

    unless Map.has_key?(raw_config, "type") or Map.has_key?(raw_config, "model_type") do
      raise ArgumentError,
            "Malformed config.json at #{config_path}: missing the \"type\" " <>
              "(or \"model_type\") discriminator key expected of a SmolVLA " <>
              "(LeRobot-native) checkpoint."
    end

    config = Config.from_map(raw_config)

    input_image_keys =
      raw_config
      |> Map.get("input_features", %{})
      |> Enum.filter(fn {_k, v} -> is_map(v) and Map.get(v, "type") == "VISUAL" end)
      |> Enum.map(fn {k, _v} -> k end)
      |> Enum.sort()

    config = %{
      config
      | input_image_keys:
          if(input_image_keys == [], do: ["observation.image"], else: input_image_keys)
    }

    weights_path = Path.join(checkpoint_dir, "model.safetensors")
    weights = Weights.load!(weights_path)

    tokenizer_path =
      Keyword.get_lazy(opts, :tokenizer_path, fn -> default_tokenizer_path(config) end)

    tokenizer = Tokenizer.load!(tokenizer_path, config.tokenizer_max_length)

    %__MODULE__{config: config, weights: weights, tokenizer: tokenizer}
  end

  # Mirrors the Python reference's own network-fetched tokenizer source
  # (`AutoTokenizer.from_pretrained(config.vlm_model_name)`), resolved
  # here as the equivalent local HF cache path instead (this port has no
  # HF Hub client of its own) -- `~/.cache/huggingface/hub` is the same
  # cache directory `huggingface_hub`/`transformers` populate, so a prior
  # Python-side run (which every prior chunk in this build already
  # required) has typically already fetched it.
  defp default_tokenizer_path(%Config{vlm_model_name: vlm_model_name}) do
    cache_glob =
      Path.join([
        System.user_home!(),
        ".cache",
        "huggingface",
        "hub",
        "models--" <> String.replace(vlm_model_name, "/", "--"),
        "snapshots",
        "*",
        "tokenizer.json"
      ])

    case Path.wildcard(cache_glob) do
      [path | _] -> path
      [] -> raise "No cached tokenizer.json found for #{vlm_model_name} under #{cache_glob}"
    end
  end

  @doc """
  Encodes one observation (image, robot state, instruction) and runs the
  flow-matching action expert to produce one action chunk.

  `image`: `{H, W, 3}` (or `{3, H, W}`) numeric data -- an `Nx.Tensor`,
  or anything `Nx.tensor/1` accepts, values in `[0, 1]` or `[0, 255]`
  (both handled, matching 01.1's own heuristic). Per this component's
  interface contract, `infer_action` takes ONE image; a checkpoint whose
  own `input_features` declare multiple cameras receives that single
  image in its first camera slot and zero-filled, zero-masked images in
  the rest -- mirrors 01.1's own multi-camera padding (see
  `prepare_images/2`).

  `state`: a flat list/tensor of floats, `state_dim <=
  config.max_state_dim`.

  `instruction`: a plain-text language instruction.

  Returns a continuous `Nx.Tensor` of shape `{chunk_size, action_dim}`
  -- never a token sampled from a vocabulary (the "action expert's
  output is never tokenized" invariant).

  **Fails**: a state vector whose dimensionality exceeds
  `config.max_state_dim` raises `ArgumentError` BEFORE dispatching to
  `emily` -- a shorter state is valid and zero-padded, never a silent
  truncation of an oversized one (matches 01.1's own "Fails" note and
  01.2's "same loud/local failure shape as 01.1" invariant).
  """
  @spec infer_action(t(), Nx.Tensor.t() | [[number()]], [number()], String.t()) :: Nx.Tensor.t()
  def infer_action(%__MODULE__{} = model, image, state, instruction) do
    infer_action(model, image, state, instruction, nil)
  end

  @doc false
  # Test/conformance seam -- not part of the design's pinned 4-arity
  # interface. Flow-matching's Euler integration starts from random
  # noise (`mx.random.normal` on the Python side, unseeded, drawn fresh
  # per call on both sides in the pinned 4-arity path above) -- a
  # bit-exact cross-runtime numerical-parity check therefore needs the
  # SAME starting noise fed to both implementations, which the pinned
  # interface has no way to express (by design: it is not part of
  # SmolVLA's real contract, which never takes noise as an input). This
  # 5-arity overload lets a conformance test inject a fixed noise tensor
  # (e.g. read from a fixture the Python reference's own run produced)
  # while leaving `infer_action/4`'s public shape completely unchanged.
  @spec infer_action(
          t(),
          Nx.Tensor.t() | [[number()]],
          [number()],
          String.t(),
          Nx.Tensor.t() | nil
        ) ::
          Nx.Tensor.t()
  def infer_action(%__MODULE__{} = model, image, state, instruction, fixed_noise)
      when is_binary(instruction) do
    state_tensor = Nx.tensor(state, type: :f32)

    if Nx.rank(state_tensor) != 1 do
      raise ArgumentError,
            "infer_action/4 expects a 1D state vector, got shape #{inspect(Nx.shape(state_tensor))}"
    end

    state_dim = Nx.axis_size(state_tensor, 0)

    if state_dim > model.config.max_state_dim do
      raise ArgumentError,
            "infer_action/4 got a state vector of dimensionality #{state_dim}, which " <>
              "exceeds this checkpoint's max_state_dim=#{model.config.max_state_dim}. " <>
              "Wrong action-space dimensionality for the loaded config -- refusing to " <>
              "silently reshape or truncate."
    end

    padded_state =
      Nx.pad(state_tensor, 0.0, [{0, model.config.max_state_dim - state_dim, 0}])
      |> Nx.new_axis(0)
      |> Nx.backend_transfer(Emily.Backend)

    {images, image_masks} = prepare_images(model.config, image)

    {prefix_embeds, prefix_pad_mask, prefix_att_mask} =
      embed_prefix(model, images, image_masks, instruction, padded_state)

    noise_shape = {1, model.config.chunk_size, model.config.max_action_dim}

    noise =
      case fixed_noise do
        nil ->
          key = Nx.Random.key(:erlang.system_time())
          {noise, _key} = Nx.Random.normal(key, shape: noise_shape, type: :f32)
          Nx.backend_transfer(noise, Emily.Backend)

        %Nx.Tensor{} = supplied ->
          supplied |> Nx.as_type(:f32) |> Nx.backend_transfer(Emily.Backend)
      end

    x_t = sample_actions(model, prefix_embeds, prefix_pad_mask, prefix_att_mask, noise)

    action_dim = Config.action_dim(model.config)
    x_t |> Nx.slice_along_axis(0, action_dim, axis: 2) |> Nx.squeeze(axes: [0])
  end

  # ------------------------------------------------------------------
  # Prefix: images + language + state.
  # ------------------------------------------------------------------

  defp embed_prefix(model, images, image_masks, instruction, padded_state) do
    hidden_size = model.config.text.hidden_size
    scale = :math.sqrt(hidden_size)

    {embeds_list, att_mask_values, pad_mask_values} =
      Enum.zip(images, image_masks)
      |> Enum.reduce({[], [], []}, fn {image, is_real}, {embeds, att_vals, pad_vals} ->
        img_emb = Vision.forward(model.weights, model.config, image)
        img_emb = Nx.multiply(img_emb, scale)
        num_tokens = Nx.axis_size(img_emb, 1)

        {
          [img_emb | embeds],
          att_vals ++ List.duplicate(0, num_tokens),
          pad_vals ++ List.duplicate(is_real, num_tokens)
        }
      end)

    token_ids = Tokenizer.encode!(model.tokenizer, instruction)
    lang_ids = Nx.tensor([token_ids], type: :s64) |> Nx.backend_transfer(Emily.Backend)
    lang_emb = embedding_lookup(model.weights["text_embed_tokens.weight"], lang_ids)
    lang_emb = Nx.multiply(lang_emb, scale)

    att_mask_values = att_mask_values ++ List.duplicate(0, length(token_ids))
    pad_mask_values = pad_mask_values ++ List.duplicate(true, length(token_ids))

    state_emb =
      linear_no_bias(padded_state, model.weights["state_proj.weight"])
      |> Nx.add(model.weights["state_proj.bias"])
      |> Nx.new_axis(1)

    att_mask_values = att_mask_values ++ [1]
    pad_mask_values = pad_mask_values ++ [true]

    embeds =
      Nx.concatenate(Enum.reverse(embeds_list) ++ [lang_emb, state_emb], axis: 1)

    pad_mask =
      pad_mask_values
      |> Enum.map(&if(&1, do: 1, else: 0))
      |> Nx.tensor(type: :u8)
      |> Nx.not_equal(0)
      |> Nx.new_axis(0)
      |> Nx.backend_transfer(Emily.Backend)

    att_mask =
      att_mask_values
      |> Nx.tensor(type: :s32)
      |> Nx.new_axis(0)
      |> Nx.backend_transfer(Emily.Backend)

    {embeds, pad_mask, att_mask}
  end

  defp embedding_lookup(embedding_table, ids) do
    Nx.take(embedding_table, ids, axis: 0)
  end

  # ------------------------------------------------------------------
  # Suffix: noisy action + flow-matching timestep.
  # ------------------------------------------------------------------

  defp embed_suffix(model, noisy_actions, timestep) do
    expert_hidden_size = Config.expert_hidden_size(model.config)

    action_emb =
      linear_no_bias(noisy_actions, model.weights["action_in_proj.weight"])
      |> Nx.add(model.weights["action_in_proj.bias"])

    time_emb =
      Expert.sinusoidal_pos_embedding(
        timestep,
        expert_hidden_size,
        model.config.min_period,
        model.config.max_period
      )

    time_emb =
      time_emb
      |> Nx.new_axis(1)
      |> Nx.broadcast(Nx.shape(action_emb))

    action_time_emb = Nx.concatenate([action_emb, time_emb], axis: 2)

    action_time_emb =
      linear_no_bias(action_time_emb, model.weights["action_time_mlp_in.weight"])
      |> Nx.add(model.weights["action_time_mlp_in.bias"])
      |> silu()

    action_time_emb =
      linear_no_bias(action_time_emb, model.weights["action_time_mlp_out.weight"])
      |> Nx.add(model.weights["action_time_mlp_out.bias"])

    chunk_size = Nx.axis_size(action_time_emb, 1)

    pad_mask =
      Nx.broadcast(1, {1, chunk_size}) |> Nx.as_type(:u8) |> Nx.not_equal(0)

    att_mask = Nx.broadcast(1, {1, chunk_size}) |> Nx.as_type(:s32)

    {action_time_emb, pad_mask, att_mask}
  end

  # ------------------------------------------------------------------
  # Flow matching: Euler integration from pure noise to a clean action.
  # ------------------------------------------------------------------

  defp sample_actions(model, prefix_embeds, prefix_pad_mask, prefix_att_mask, noise) do
    num_steps = model.config.num_steps
    dt = -1.0 / num_steps

    Enum.reduce(0..(num_steps - 1), noise, fn step, x_t ->
      t = 1.0 + step * dt
      timestep = Nx.broadcast(t, {1}) |> Nx.as_type(:f32) |> Nx.backend_transfer(Emily.Backend)

      x_t_bf16 = Nx.as_type(x_t, :bf16)

      {suffix_embeds, suffix_pad_mask, suffix_att_mask} =
        embed_suffix(model, x_t_bf16, timestep)

      pad_mask = Nx.concatenate([prefix_pad_mask, suffix_pad_mask], axis: 1)
      att_mask = Nx.concatenate([prefix_att_mask, suffix_att_mask], axis: 1)

      {_backbone_out, expert_out} =
        Expert.forward(
          model.weights,
          model.config,
          prefix_embeds,
          suffix_embeds,
          pad_mask,
          att_mask
        )

      v_t =
        linear_no_bias(Nx.as_type(expert_out, :f32), model.weights["action_out_proj.weight"])
        |> Nx.add(model.weights["action_out_proj.bias"])

      Nx.add(x_t, Nx.multiply(dt, v_t))
    end)
  end

  # ------------------------------------------------------------------
  # Image preparation: resize/pad/range-normalize + multi-camera padding.
  # ------------------------------------------------------------------

  defp prepare_images(config, image) do
    image_size = config.vision.image_size

    arr = to_hwc_f32(image)

    arr =
      if Nx.to_number(Nx.reduce_max(arr)) > 1.5 do
        Nx.divide(arr, 255.0)
      else
        arr
      end

    pixel_values = Preprocessing.resize_with_pad(arr, image_size, image_size, 0.0)
    # SigLIP expects [-1, 1].
    pixel_values = Nx.subtract(Nx.multiply(pixel_values, 2.0), 1.0)
    pixel_values = pixel_values |> Nx.new_axis(0) |> Nx.backend_transfer(Emily.Backend)

    num_cameras = max(1, length(config.input_image_keys || []))
    images = [pixel_values]
    image_masks = [true]

    {extra_images, extra_masks} =
      if num_cameras > 1 do
        zero = Nx.broadcast(0.0, Nx.shape(pixel_values)) |> Nx.backend_transfer(Emily.Backend)
        {List.duplicate(zero, num_cameras - 1), List.duplicate(false, num_cameras - 1)}
      else
        {[], []}
      end

    {images ++ extra_images, image_masks ++ extra_masks}
  end

  defp to_hwc_f32(%Nx.Tensor{} = image) do
    shape = Nx.shape(image)

    if tuple_size(shape) != 3 or (elem(shape, 0) != 3 and elem(shape, 2) != 3) do
      raise ArgumentError,
            "infer_action/4 expects a single (H, W, 3) or (3, H, W) image, got shape #{inspect(shape)}"
    end

    image = Nx.as_type(image, :f32)

    if elem(shape, 0) == 3 and elem(shape, 2) != 3 do
      Nx.transpose(image, axes: [1, 2, 0])
    else
      image
    end
  end

  defp to_hwc_f32(image), do: image |> Nx.tensor(type: :f32) |> to_hwc_f32()

  # ------------------------------------------------------------------
  # Shared numeric helpers.
  # ------------------------------------------------------------------

  defp linear_no_bias(x, w), do: Nx.dot(x, [-1], w, [-1])

  defp silu(x), do: Nx.multiply(x, Nx.sigmoid(x))
end
