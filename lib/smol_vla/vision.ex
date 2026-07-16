defmodule SmolVLA.Vision do
  @moduledoc """
  SigLIP vision tower + pixel-shuffle connector: pixel values in,
  text-backbone-hidden-size tokens out.

  Mirrors `mlx_vlm.models.smolvla.vision.VisionEncoder` (SigLIP tower
  reused from `mlx_vlm.models.idefics3.vision.VisionModel`) and
  `Connector` -- an independent reimplementation against `Nx`/`emily`
  (ADR-0004), kept structurally close to the Python side on purpose (this
  chunk's brief).

  Only the `patch_attention_mask == nil` path is ported: `infer_action`'s
  own image preprocessing (`SmolVLA.Preprocessing.prepare_image/2`)
  always resizes-with-pad to one full `image_size x image_size` frame
  before this module ever sees it, so the Python side's fractional patch
  -bucketing branch (used only for partial/padded patch grids) is never
  exercised in this codepath -- confirmed by grep: `smolvla.py` never
  passes `patch_attention_mask` to `VisionEncoder.__call__`.

  The patch embedding is a `stride == kernel_size`, no-padding Conv2d --
  mathematically a non-overlapping patch extraction followed by a linear
  projection, implemented here as reshape + matmul rather than a general
  convolution (exact, not an approximation of the Python side's
  `mx.conv2d`).

  Shape-dependent orchestration (which layer, how many heads, eps,
  reshape target shapes) uses `deftransform`/`deftransformp` -- Nx.Defn's
  "plain Elixir, callable from a defn body" escape hatch -- since a
  weight map, a config struct, and plain integers/floats are not
  themselves traceable tensors; only the tensor math itself runs through
  `defn`/`Emily.Fast`.

  **Dtype**: the real checkpoint stores every vision-tower weight as
  `bf16`, and MLX's own `nn.Linear`/attention/`mx.fast.layer_norm` run
  their matmuls natively in whatever dtype the inputs already are (the
  fused norm kernels upcast to f32 internally for the reduction only,
  then cast back -- see `Emily.Fast`'s own documented "upcast:
  :normalization" recipe). This module matches that: activations flow
  through in `bf16`, never force-upcast to `f32`, so the accumulated
  rounding behavior across 12 layers matches the Python reference's real
  numerics rather than a strictly-more-precise (and therefore diverging)
  f32 reimplementation. Patch/position embedding is the one exception,
  and matches the Python reference exactly there too: it computes in
  `f32` (the raw pixel input's dtype; MLX/Nx both auto-promote a bf16
  weight matmul against an f32 input to f32), and only the resulting
  embedding tensor is explicitly cast down to `bf16` before entering the
  transformer encoder -- see `forward/3`'s own comment on
  `idefics3.vision.VisionModel.__call__`'s identical explicit downcast.

  Verified against the Python reference two ways (see
  `test/smol_vla/vision_test.exs`): an all-`f32` build of both sides
  (bypassing bf16 entirely, by upcasting every vision weight before the
  forward pass) matches to float32 rounding noise, confirming the
  MECHANISM is correct; the real bf16-native forward pass on a
  realistic random image then diverges from the Python reference by
  ~1.5% mean relative error after 12 transformer layers -- consistent
  with expected bf16 (8-bit mantissa) rounding accumulating differently
  across two independently-implemented op sequences on the same MLX
  backend, not a mechanism bug (confirmed by the matching all-f32
  control). A degenerate same-valued-pixel image (adversarial: every
  patch identical, so every attention/softmax position is a near-exact
  tie, maximally sensitive to rounding-order differences) diverges
  further (up to ~20% mean relative at a few outlier positions) -- an
  edge case around softmax/GELU tie-breaking, not something a real
  camera image exercises.
  """

  alias SmolVLA.Config

  import Nx.Defn

  @doc """
  Runs the SigLIP tower + pixel-shuffle connector on one preprocessed
  image batch.

  `pixel_values`: `{batch, image_size, image_size, 3}`, already
  resized/padded/range-normalized to `[-1, 1]` (see
  `SmolVLA.Preprocessing.prepare_image/2`).

  Returns `{batch, num_tokens, text_hidden_size}`, where `num_tokens =
  (image_size / patch_size)^2 / scale_factor^2`.
  """
  @spec forward(map(), Config.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  deftransform forward(weights, %Config{} = config, pixel_values) do
    forward_traced(weights, pixel_values, config: config)
  end

  # The traced entry point, mirroring `SmolVLA.Expert.forward`'s own
  # pattern: called eagerly (no enclosing defn), it jits the WHOLE
  # 12-layer SigLIP tower + connector into a SINGLE Nx.Defn graph -- one
  # Emily.Compiler native replay per camera instead of hundreds of
  # separate eager NIF dispatches (op-by-op `def`/`defp`). `config` rides
  # in `opts` because defn positional arguments must be tensors or
  # containers, and Config is a plain struct of scalars; as a
  # compile-time option it also keys the jit cache, so repeated calls
  # with the same shapes/config replay one compiled program.
  defn forward_traced(weights, pixel_values, opts \\ []) do
    forward_stack(weights, pixel_values, opts[:config])
  end

  # Plain-Elixir stack construction, run at trace time inside the single
  # traced graph (deftransformp): the 12-layer encoder loop unrolls at
  # compile time and the string-keyed weight lookups resolve against the
  # traced weights container.
  deftransformp forward_stack(weights, pixel_values, %Config{} = config) do
    conv_w = weights["vision_encoder.vision_model.embeddings.patch_embedding.weight"]

    hidden =
      weights
      |> embeddings(config, pixel_values)
      # Matches the Python reference exactly: patch+position embedding
      # computes in f32 (pixel_values arrives f32; MLX auto-promotes
      # bf16-weight matmuls against an f32 input to f32), and only THEN
      # is the whole embedding tensor cast down to the patch embedding
      # weight's native dtype (bf16 on the real checkpoint) before
      # entering the transformer encoder -- see
      # `idefics3.vision.VisionModel.__call__`'s own
      # `x = x.astype(self.embeddings.patch_embedding.weight.dtype)`.
      |> Nx.as_type(Nx.type(conv_w))

    hidden =
      Enum.reduce(0..(config.vision.num_hidden_layers - 1), hidden, fn layer_idx, hidden ->
        encoder_layer(weights, config, layer_idx, hidden)
      end)

    pooled =
      fused_layer_norm(
        hidden,
        weights["vision_encoder.vision_model.post_layernorm.weight"],
        weights["vision_encoder.vision_model.post_layernorm.bias"],
        eps: config.vision.layer_norm_eps
      )

    connector(weights, pooled)
  end

  # ------------------------------------------------------------------
  # Patch + position embeddings.
  # ------------------------------------------------------------------

  deftransformp embeddings(weights, config, pixel_values) do
    patch_size = config.vision.patch_size
    hidden_size = config.vision.hidden_size

    conv_w = weights["vision_encoder.vision_model.embeddings.patch_embedding.weight"]
    conv_b = weights["vision_encoder.vision_model.embeddings.patch_embedding.bias"]
    pos_table = weights["vision_encoder.vision_model.embeddings.position_embedding.weight"]

    {batch, height, width, channels} = Nx.shape(pixel_values)
    patches_h = div(height, patch_size)
    patches_w = div(width, patch_size)
    num_patches = patches_h * patches_w

    patches =
      pixel_values
      |> Nx.reshape({batch, patches_h, patch_size, patches_w, patch_size, channels})
      |> Nx.transpose(axes: [0, 1, 3, 2, 4, 5])
      |> Nx.reshape({batch, num_patches, patch_size * patch_size * channels})

    conv_w_flat = Nx.reshape(conv_w, {hidden_size, patch_size * patch_size * channels})

    # `patches` is f32 (from pixel_values); conv_w/conv_b/pos_table are
    # bf16 (native checkpoint dtype). Left un-cast here so Nx's own
    # type promotion (bf16 + f32 -> f32) reproduces MLX's identical
    # promotion behavior instead of forcing a dtype -- matching the
    # Python reference's real embeddings dtype (see `forward/3`'s own
    # comment on the subsequent explicit downcast).
    patch_embeds =
      patches
      |> then(&linear_no_bias(&1, conv_w_flat))
      |> Nx.add(conv_b)

    position_ids = Nx.tile(Nx.iota({num_patches}, type: :s64), [batch, 1])
    pos_embeds = Nx.take(pos_table, position_ids, axis: 0)

    Nx.add(patch_embeds, pos_embeds)
  end

  # ------------------------------------------------------------------
  # Encoder layer: pre-LN self-attention + pre-LN MLP(GELU-precise).
  # ------------------------------------------------------------------

  deftransformp encoder_layer(weights, config, layer_idx, hidden) do
    prefix = "vision_encoder.vision_model.encoder.layers.#{layer_idx}."
    num_heads = config.vision.num_attention_heads
    head_dim = div(config.vision.hidden_size, num_heads)
    eps = config.vision.layer_norm_eps

    normed =
      fused_layer_norm(
        hidden,
        weights[prefix <> "layer_norm1.weight"],
        weights[prefix <> "layer_norm1.bias"],
        eps: eps
      )

    q =
      normed
      |> linear(
        weights[prefix <> "self_attn.q_proj.weight"],
        weights[prefix <> "self_attn.q_proj.bias"]
      )
      |> split_heads(num_heads, head_dim)

    k =
      normed
      |> linear(
        weights[prefix <> "self_attn.k_proj.weight"],
        weights[prefix <> "self_attn.k_proj.bias"]
      )
      |> split_heads(num_heads, head_dim)

    v =
      normed
      |> linear(
        weights[prefix <> "self_attn.v_proj.weight"],
        weights[prefix <> "self_attn.v_proj.bias"]
      )
      |> split_heads(num_heads, head_dim)

    scale = 1.0 / :math.sqrt(head_dim)
    attn = Emily.Fast.scaled_dot_product_attention(q, k, v, scale: scale)

    attn_out =
      attn
      |> merge_heads()
      |> linear(
        weights[prefix <> "self_attn.out_proj.weight"],
        weights[prefix <> "self_attn.out_proj.bias"]
      )

    hidden = Nx.add(hidden, attn_out)

    normed2 =
      fused_layer_norm(
        hidden,
        weights[prefix <> "layer_norm2.weight"],
        weights[prefix <> "layer_norm2.bias"],
        eps: eps
      )

    mlp_hidden =
      normed2
      |> linear(weights[prefix <> "mlp.fc1.weight"], weights[prefix <> "mlp.fc1.bias"])
      |> gelu_precise()

    mlp_out =
      linear(mlp_hidden, weights[prefix <> "mlp.fc2.weight"], weights[prefix <> "mlp.fc2.bias"])

    Nx.add(hidden, mlp_out)
  end

  # GELU "precise"/"tanh" approximation, matching mlx.nn.GELU(approx="precise"):
  # 0.5*x*(1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3))).
  defnp gelu_precise(x) do
    c = 0.7978845608028654
    inner = c * (x + 0.044715 * x * x * x)
    0.5 * x * (1.0 + Nx.tanh(inner))
  end

  # ------------------------------------------------------------------
  # Pixel-shuffle connector.
  # ------------------------------------------------------------------

  deftransformp connector(weights, image_hidden_states) do
    proj_w = weights["vision_encoder.connector.modality_projection.weight"]
    shuffled = pixel_shuffle(image_hidden_states, 4)
    linear_no_bias(shuffled, proj_w)
  end

  deftransformp pixel_shuffle(x, scale_factor) do
    {bsz, seq, embed_dim} = Nx.shape(x)
    side = trunc(:math.sqrt(seq))

    x
    |> Nx.reshape({bsz, side, side, embed_dim})
    |> Nx.reshape({bsz, side, div(side, scale_factor), embed_dim * scale_factor})
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> Nx.reshape({
      bsz,
      div(side, scale_factor),
      div(side, scale_factor),
      embed_dim * scale_factor * scale_factor
    })
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> Nx.reshape({
      bsz,
      div(seq, scale_factor * scale_factor),
      embed_dim * scale_factor * scale_factor
    })
  end

  # ------------------------------------------------------------------
  # Shared numeric helpers.
  # ------------------------------------------------------------------

  deftransformp linear(x, w, b) do
    y = linear_no_bias(x, w)
    Nx.add(y, b)
  end

  defnp linear_no_bias(x, w) do
    Nx.dot(x, [-1], w, [-1])
  end

  deftransformp fused_layer_norm(x, weight, bias, opts) do
    Emily.Fast.layer_norm(x, weight, bias, opts)
  end

  deftransformp split_heads(x, num_heads, head_dim) do
    {b, l, _} = Nx.shape(x)
    x |> Nx.reshape({b, l, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
  end

  deftransformp merge_heads(x) do
    {b, h, l, d} = Nx.shape(x)
    x |> Nx.transpose(axes: [0, 2, 1, 3]) |> Nx.reshape({b, l, h * d})
  end
end
