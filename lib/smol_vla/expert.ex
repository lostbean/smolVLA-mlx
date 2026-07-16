defmodule SmolVLA.Expert do
  @moduledoc """
  The joint attention mechanism between the frozen SmolLM2 text backbone
  (the VLM "prefix") and the flow-matching action expert (the "suffix").

  Mirrors `mlx_vlm.models.smolvla.expert` structurally -- an independent
  reimplementation against `Nx`/`emily` (ADR-0004), kept close to the
  Python module's own shape (same function/section split) on purpose per
  this chunk's brief, so the two can be read side by side. See that
  module's own extensive docstring (grounded against lerobot's
  `smolvlm_with_expert.py`) for the mechanism's full citations; summary:

    * `num_vlm_layers` (16) aligned (backbone-layer, expert-layer) pairs.
      Every `self_attn_every_n_layers`-th layer is SELF-attention:
      backbone and expert tokens concatenate into one joint sequence,
      RoPE'd and attended together under one shared block-causal mask.
      The remaining layers are CROSS-attention: the backbone
      self-attends as usual, but the expert computes its own query and
      reads the backbone's already-RoPE'd key (and matching value),
      RE-PROJECTED through the expert's OWN (differently-shaped)
      k_proj/v_proj -- never through its own hidden state on those
      layers.
    * The VLM prefix never attends to the action suffix; the suffix
      attends to the entire prefix plus itself.
    * Both branches always compute their own queries from their own
      hidden state at every layer; only the key/value SOURCE differs
      between the two layer kinds.
    * head_dim (64) and query head count (15) are shared between
      backbone and expert at every layer; only hidden_size (960 vs 720)
      and intermediate_size differ.
    * RoPE is the standard split-half ("non-traditional") rotation --
      `Emily.Fast.rope(..., traditional: false)`, matching
      `mlx.nn.RoPE(traditional=False)` bit-for-bit (verified directly
      during development).

  Real-checkpoint corroboration (identical to the Python module's own,
  since both read the same checkpoint):
    lm_expert.layers.0.self_attn.{q,k,v}_proj.weight: (960,720) (320,720) (320,720)  -- self-attn (even)
    lm_expert.layers.1.self_attn.{q,k,v}_proj.weight: (960,720) (320,320) (320,320)  -- cross-attn (odd)
    vlm.text_model.layers.0.self_attn.{q,k,v}_proj.weight: (960,960) (320,960) (320,960)
  """

  alias SmolVLA.Config

  import Nx.Defn

  @doc """
  Sine-cosine embedding of a scalar flow-matching timestep.

  Per lerobot's `create_sinusoidal_pos_embedding`: a geometric sweep of
  `div(dimension, 2)` periods between `min_period` and `max_period`,
  embedded as `[sin(2*pi*t/period), cos(2*pi*t/period)]` concatenated.
  `time`: shape `{batch}`. Returns `{batch, dimension}`.
  """
  @spec sinusoidal_pos_embedding(Nx.Tensor.t(), pos_integer(), float(), float()) ::
          Nx.Tensor.t()
  def sinusoidal_pos_embedding(time, dimension, min_period, max_period)
      when rem(dimension, 2) == 0 do
    half = div(dimension, 2)
    sinusoidal_pos_embedding_kernel(time, min_period, max_period, half: half)
  end

  # `half` (the geometric sweep's element count -- a plain Elixir integer,
  # never itself a tensor) is passed as a `defn` OPTION (`half: half`),
  # not a positional tensor argument -- `deftransformp fraction_of/1`
  # below builds `Nx.linspace(..., n: half)` purely at trace-construction
  # time via `Nx.Defn`'s documented "invoking custom Elixir code" escape
  # hatch, so `half` is never lifted into a traced `Nx.Defn.Expr` (which
  # `Nx.linspace/2`'s `n:` option cannot accept -- it needs a real integer;
  # confirmed directly: passing `half` as an ordinary positional `defn`
  # argument makes `Nx.Defn` trace-lift it into a scalar `Nx.Defn.Expr`,
  # and `Nx.linspace(n: <traced expr>)` then raises). This also fixes a
  # SEPARATE issue a prior version of this function had: eagerly
  # `Nx.backend_transfer`-ing `fraction` before passing it in, which broke
  # under `Nx.Defn.value_and_grad` (`SmolVLA.Train.loss/3` calls
  # `SmolVLA.embed_suffix/4`, which calls this, from inside a traced
  # gradient computation, where `backend_transfer` on an already-symbolic
  # value raises) -- building `fraction` freshly inside the trace (via
  # `deftransformp`, still real Elixir/Nx code, just invoked while the
  # kernel below is being traced) sidesteps both problems at once.
  deftransformp fraction_of(half) do
    Nx.linspace(0.0, 1.0, n: half, type: :f32)
  end

  defnp sinusoidal_pos_embedding_kernel(time, min_period, max_period, opts \\ []) do
    fraction = fraction_of(opts[:half])
    period = min_period * Nx.pow(max_period / min_period, fraction)
    scaling_factor = 1.0 / period * 2 * 3.141592653589793

    sin_input = Nx.multiply(Nx.new_axis(scaling_factor, 0), Nx.new_axis(time, 1))
    Nx.concatenate([Nx.sin(sin_input), Nx.cos(sin_input)], axis: 1)
  end

  @doc """
  Builds the block-causal 2D attention mask (as a boolean, `True` =
  attend): tokens attend to any token whose cumulative `att_mask` is
  `<=` their own, further restricted to non-padding positions.

  `att_masks` values: `0` = "shares the previous token's attention
  scope" (image/language tokens, within one prefix block), `1` =
  "starts a new causal block" (the state token, and every suffix
  token) -- matches lerobot's own `make_att_2d_masks`. `pad_masks`,
  `att_masks`: `{batch, len}`. Returns `{batch, len, len}` boolean.
  """
  @spec make_att_2d_masks(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  defn make_att_2d_masks(pad_masks, att_masks) do
    cumsum = Nx.cumulative_sum(att_masks, axis: 1)
    att_2d = Nx.less_equal(Nx.new_axis(cumsum, 1), Nx.new_axis(cumsum, 2))
    pad_2d = Nx.logical_and(Nx.new_axis(pad_masks, 1), Nx.new_axis(pad_masks, 2))
    Nx.logical_and(att_2d, pad_2d)
  end

  @doc """
  Runs the full stack of `num_vlm_layers` joint (backbone, expert)
  layers, plus final RMSNorms on each branch.

  `backbone_embeds`: `{1, backbone_len, text_hidden_size}`.
  `expert_embeds`: `{1, expert_len, expert_hidden_size}`.
  `pad_mask`/`att_mask`: `{1, backbone_len + expert_len}`, the
  concatenation of the prefix's and suffix's own per-token masks.

  Returns `{backbone_out, expert_out}`.
  """
  @spec forward(map(), Config.t(), Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t()) ::
          {Nx.Tensor.t(), Nx.Tensor.t()}
  deftransform forward(
                 weights,
                 %Config{} = config,
                 backbone_embeds,
                 expert_embeds,
                 pad_mask,
                 att_mask
               ) do
    forward_traced(weights, backbone_embeds, expert_embeds, pad_mask, att_mask, config: config)
  end

  # The traced entry point: called eagerly (no enclosing defn), the defn
  # entrypoint jits the WHOLE 16-layer joint-attention stack into a
  # single Nx.Defn graph -- one Emily.Compiler native replay per Euler
  # step instead of hundreds of separate eager NIF dispatches. Called
  # from inside a caller's own defn (e.g. a future traced training
  # step), it composes into that caller's graph instead. Requires the
  # Emily.Fast kernels to be composable inside defn (ausimian/emily#205,
  # fixed on the pinned fork branch). `config` rides in `opts` because
  # defn positional arguments must be tensors/containers, and Config is
  # a plain struct of scalars -- as a compile-time option it also keys
  # the jit cache, so all ten Euler steps (same shapes, same config)
  # replay one compiled program.
  defn forward_traced(weights, backbone_embeds, expert_embeds, pad_mask, att_mask, opts \\ []) do
    forward_stack(weights, backbone_embeds, expert_embeds, pad_mask, att_mask, opts[:config])
  end

  # Plain-Elixir stack construction, run at trace time inside the single
  # traced graph (deftransformp): the layer loop unrolls over the 16
  # compile-time (backbone, expert) layer pairs, and the string-keyed
  # weight lookups resolve against the traced weights container.
  deftransformp forward_stack(
                  weights,
                  backbone_embeds,
                  expert_embeds,
                  pad_mask,
                  att_mask,
                  config
                ) do
    backbone_len = elem(Nx.shape(backbone_embeds), 1)
    mask_bool = make_att_2d_masks(pad_mask, att_mask)
    # Emily.Fast SDPA takes an ADDITIVE mask, not a boolean gate (unlike
    # mlx.fast.scaled_dot_product_attention, which accepts boolean
    # directly) -- convert once here: True -> 0.0, False -> -inf. The
    # mask's dtype must promote to the attention's own output dtype
    # (bf16 for the real checkpoint's native-dtype activations), so it
    # is built directly in that dtype rather than f32 -- matching the
    # activations it will be added onto.
    activation_dtype = Nx.type(backbone_embeds)
    mask = additive_mask(mask_bool, activation_dtype)

    num_layers = config.num_vlm_layers
    n = config.self_attn_every_n_layers

    {backbone_hidden, expert_hidden} =
      Enum.reduce(0..(num_layers - 1), {backbone_embeds, expert_embeds}, fn layer_idx, {b, e} ->
        is_self_attn_layer = n <= 0 or rem(layer_idx, n) == 0

        layer(weights, config, layer_idx, is_self_attn_layer, b, e, mask, backbone_len)
      end)

    backbone_out =
      fused_rms_norm(backbone_hidden, weights["expert_stack.backbone_norm.weight"],
        eps: config.text.rms_norm_eps
      )

    expert_out =
      fused_rms_norm(expert_hidden, weights["expert_stack.expert_norm.weight"],
        eps: config.text.rms_norm_eps
      )

    {backbone_out, expert_out}
  end

  deftransformp additive_mask(mask_bool, dtype) do
    zeros = Nx.broadcast(0.0, Nx.shape(mask_bool)) |> Nx.as_type(dtype)
    neg_inf = Nx.broadcast(:neg_infinity, Nx.shape(mask_bool)) |> Nx.as_type(dtype)
    mask = Nx.select(mask_bool, zeros, neg_inf)
    # (batch, len, len) -> (batch, 1, len, len), broadcasts across heads.
    Nx.new_axis(mask, 1)
  end

  # ------------------------------------------------------------------
  # Prefill / step split (inference KV cache).
  #
  # The attention mask guarantees the backbone branch never attends the
  # suffix, so across all N Euler steps the backbone's entire 16-layer
  # trajectory is IDENTICAL -- and so, per layer, are the exact key/value
  # tensors the suffix attends: the RoPE'd backbone k/v (`bk`/`bv`) on
  # self-attn layers, and the expert-reprojected `ek`/`ev` (which derive
  # only from `bk`/`bv`) on cross-attn layers. `prefill` runs the
  # backbone once and caches those per-layer k/v tensors plus the two
  # precomputed additive masks the step pass needs; `step` (run once per
  # Euler step) advances ONLY the expert branch, its suffix queries
  # attending the cached keys. This mirrors lerobot's `fill_kv_cache`.
  #
  # Numerically identical to the joint `forward/6`: on a self-attn layer
  # the backbone-only attention over the prefix mask equals the joint
  # attention sliced to the backbone rows (the expert-key logits are
  # -inf, contributing exactly zero to both the softmax numerator and its
  # normalizer), and the expert-branch math is the same ops in the same
  # order -- only the backbone k/v it concatenates are now read from the
  # cache instead of recomputed.
  # ------------------------------------------------------------------

  @doc """
  Prefill pass: run the frozen prefix (backbone) through all
  `num_vlm_layers` layers once, returning the per-layer key/value cache
  the suffix will attend plus the precomputed step masks.

  Returns `{cache, self_mask, cross_mask}`:
    * `cache` -- a tuple of `{k, v}` tuples, one per layer (self-attn
      layers cache the RoPE'd backbone k/v; cross-attn layers cache the
      expert-reprojected k/v); a tuple, not a list, so it threads through
      the defn boundary as an `Nx.Container`;
    * `self_mask` -- `{1, 1, expert_len, backbone_len + expert_len}`, the
      additive mask for a self-attn step (expert rows attending all keys);
    * `cross_mask` -- `{1, 1, expert_len, backbone_len}`, the additive
      mask for a cross-attn step (expert rows attending prefix keys only).

  `expert_len` is fixed across steps, so both masks are step-invariant.
  """
  @spec prefill(map(), Config.t(), Nx.Tensor.t(), non_neg_integer(), Nx.Tensor.t(), Nx.Tensor.t()) ::
          {tuple(), Nx.Tensor.t(), Nx.Tensor.t()}
  deftransform prefill(
                 weights,
                 %Config{} = config,
                 backbone_embeds,
                 expert_len,
                 pad_mask,
                 att_mask
               )
               when is_integer(expert_len) do
    prefill_traced(weights, backbone_embeds, pad_mask, att_mask,
      config: config,
      expert_len: expert_len
    )
  end

  defn prefill_traced(weights, backbone_embeds, pad_mask, att_mask, opts \\ []) do
    prefill_stack(weights, backbone_embeds, pad_mask, att_mask, opts[:config], opts[:expert_len])
  end

  deftransformp prefill_stack(weights, backbone_embeds, pad_mask, att_mask, config, expert_len) do
    backbone_len = elem(Nx.shape(backbone_embeds), 1)
    mask_bool = make_att_2d_masks(pad_mask, att_mask)
    activation_dtype = Nx.type(backbone_embeds)
    mask = additive_mask(mask_bool, activation_dtype)

    # The expert query rows [backbone_len, backbone_len+expert_len).
    self_mask = Nx.slice_along_axis(mask, backbone_len, expert_len, axis: 2)
    cross_mask = Nx.slice_along_axis(self_mask, 0, backbone_len, axis: 3)

    num_heads = config.text.num_attention_heads
    num_kv_heads = config.text.num_key_value_heads
    head_dim = div(config.text.hidden_size, num_heads)
    rope_theta = config.text.rope_theta
    eps = config.text.rms_norm_eps
    prefix_mask = Nx.slice_along_axis(mask, 0, backbone_len, axis: 2)
    prefix_mask = Nx.slice_along_axis(prefix_mask, 0, backbone_len, axis: 3)

    num_layers = config.num_vlm_layers
    n = config.self_attn_every_n_layers

    {_backbone_hidden, cache_rev} =
      Enum.reduce(0..(num_layers - 1), {backbone_embeds, []}, fn layer_idx, {b, cache} ->
        is_self_attn_layer = n <= 0 or rem(layer_idx, n) == 0
        backbone_prefix = "expert_stack.layers.#{layer_idx}.backbone."
        expert_prefix = "expert_stack.layers.#{layer_idx}.expert."

        {bq, bk, bv} =
          project_qkv(weights, backbone_prefix, b, num_heads, num_kv_heads, head_dim,
            rope_theta: rope_theta,
            eps: eps,
            offset: 0
          )

        scale = 1.0 / :math.sqrt(head_dim)

        backbone_att =
          Emily.Fast.scaled_dot_product_attention_with_mask(bq, bk, bv, prefix_mask, scale: scale)

        backbone_out =
          merge_heads(backbone_att)
          |> linear_no_bias(weights[backbone_prefix <> "self_attn.o_proj.weight"])

        b_next = mlp_residual(weights, backbone_prefix, b, backbone_out, eps)

        # The (k, v) the suffix attends on THIS layer:
        #   self-attn -> the RoPE'd backbone k/v directly;
        #   cross-attn -> the expert's re-projection of the backbone k/v.
        kv =
          if is_self_attn_layer do
            {bk, bv}
          else
            bk_flat = merge_heads_kv(bk)
            bv_flat = merge_heads_kv(bv)

            ek =
              bk_flat
              |> linear_no_bias(weights[expert_prefix <> "self_attn.k_proj.weight"])
              |> split_heads(num_kv_heads, head_dim)

            ev =
              bv_flat
              |> linear_no_bias(weights[expert_prefix <> "self_attn.v_proj.weight"])
              |> split_heads(num_kv_heads, head_dim)

            {ek, ev}
          end

        {b_next, [kv | cache]}
      end)

    # The cache is returned as a TUPLE of `{k, v}` tuples (not a list):
    # `Nx.Container` is implemented for tuples but NOT for lists, so a
    # list passed back through a defn boundary would not be traversed as
    # a container -- its tensor leaves would not be recognized as jit
    # inputs. A fixed-size tuple (16 layers, compile-time known) threads
    # each cached tensor through correctly.
    cache = cache_rev |> Enum.reverse() |> List.to_tuple()
    {cache, self_mask, cross_mask}
  end

  @doc """
  Step pass: advance ONLY the expert (suffix) branch through all
  `num_vlm_layers` layers, its queries attending the prefilled backbone
  key/value cache. Returns `expert_out` (final expert RMSNorm applied),
  ready for the action projection.

  `cache`, `self_mask`, `cross_mask` come from `prefill/6`;
  `backbone_len` is the prefix length the self-attn RoPE offset uses.
  """
  @spec step(
          map(),
          Config.t(),
          tuple(),
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          non_neg_integer()
        ) ::
          Nx.Tensor.t()
  deftransform step(
                 weights,
                 %Config{} = config,
                 cache,
                 expert_embeds,
                 self_mask,
                 cross_mask,
                 backbone_len
               )
               when is_integer(backbone_len) do
    step_traced(weights, cache, expert_embeds, self_mask, cross_mask,
      config: config,
      backbone_len: backbone_len
    )
  end

  defn step_traced(weights, cache, expert_embeds, self_mask, cross_mask, opts \\ []) do
    step_stack(
      weights,
      cache,
      expert_embeds,
      self_mask,
      cross_mask,
      opts[:config],
      opts[:backbone_len]
    )
  end

  deftransformp step_stack(
                  weights,
                  cache,
                  expert_embeds,
                  self_mask,
                  cross_mask,
                  config,
                  backbone_len
                ) do
    num_heads = config.text.num_attention_heads
    num_kv_heads = config.text.num_key_value_heads
    head_dim = div(config.text.hidden_size, num_heads)
    rope_theta = config.text.rope_theta
    eps = config.text.rms_norm_eps
    scale = 1.0 / :math.sqrt(head_dim)

    num_layers = config.num_vlm_layers
    n = config.self_attn_every_n_layers

    expert_hidden =
      Enum.reduce(0..(num_layers - 1), expert_embeds, fn layer_idx, e ->
        is_self_attn_layer = n <= 0 or rem(layer_idx, n) == 0
        expert_prefix = "expert_stack.layers.#{layer_idx}.expert."
        {ck, cv} = elem(cache, layer_idx)

        expert_att =
          if is_self_attn_layer do
            # Expert projects its OWN q/k/v (offset = backbone_len), then
            # attends the cached backbone k/v CONCATENATED with its own.
            {eq, ek, ev} =
              project_qkv(weights, expert_prefix, e, num_heads, num_kv_heads, head_dim,
                rope_theta: rope_theta,
                eps: eps,
                offset: backbone_len
              )

            k = Nx.concatenate([ck, ek], axis: 2)
            v = Nx.concatenate([cv, ev], axis: 2)
            Emily.Fast.scaled_dot_product_attention_with_mask(eq, k, v, self_mask, scale: scale)
          else
            # Cross-attn: expert query (offset 0) attends the cached
            # expert-reprojected backbone k/v only.
            normed_expert =
              fused_rms_norm(e, weights[expert_prefix <> "input_layernorm.weight"], eps: eps)

            eq =
              normed_expert
              |> linear_no_bias(weights[expert_prefix <> "self_attn.q_proj.weight"])
              |> split_heads(num_heads, head_dim)
              |> then(&Emily.Fast.rope(&1, Nx.tensor(0), dims: head_dim, base: rope_theta))

            Emily.Fast.scaled_dot_product_attention_with_mask(eq, ck, cv, cross_mask,
              scale: scale
            )
          end

        expert_out =
          merge_heads(expert_att)
          |> linear_no_bias(weights[expert_prefix <> "self_attn.o_proj.weight"])

        mlp_residual(weights, expert_prefix, e, expert_out, eps)
      end)

    fused_rms_norm(expert_hidden, weights["expert_stack.expert_norm.weight"],
      eps: config.text.rms_norm_eps
    )
  end

  # ------------------------------------------------------------------
  # One aligned (backbone-layer, expert-layer) pair.
  # ------------------------------------------------------------------

  deftransformp layer(
                  weights,
                  %Config{} = config,
                  layer_idx,
                  is_self_attn_layer,
                  backbone_hidden,
                  expert_hidden,
                  mask,
                  backbone_len
                ) do
    num_heads = config.text.num_attention_heads
    num_kv_heads = config.text.num_key_value_heads
    head_dim = div(config.text.hidden_size, num_heads)
    rope_theta = config.text.rope_theta
    eps = config.text.rms_norm_eps

    backbone_prefix = "expert_stack.layers.#{layer_idx}.backbone."
    expert_prefix = "expert_stack.layers.#{layer_idx}.expert."

    if is_self_attn_layer do
      self_attn_layer(
        weights,
        backbone_prefix,
        expert_prefix,
        backbone_hidden,
        expert_hidden,
        mask,
        backbone_len,
        num_heads,
        num_kv_heads,
        head_dim,
        rope_theta,
        eps
      )
    else
      cross_attn_layer(
        weights,
        backbone_prefix,
        expert_prefix,
        backbone_hidden,
        expert_hidden,
        mask,
        backbone_len,
        num_heads,
        num_kv_heads,
        head_dim,
        rope_theta,
        eps
      )
    end
  end

  # Concatenate backbone + expert hidden states into one joint sequence,
  # project q/k/v per branch (each branch normalizes and projects its
  # OWN hidden state), RoPE (single monotonic position sequence across
  # the concatenation: backbone 0..len-1, expert len..len+expert_len-1),
  # attend jointly under the shared mask, split the output back per
  # branch, each branch's own o_proj + residual + MLP.
  deftransformp self_attn_layer(
                  weights,
                  backbone_prefix,
                  expert_prefix,
                  backbone_hidden,
                  expert_hidden,
                  mask,
                  backbone_len,
                  num_heads,
                  num_kv_heads,
                  head_dim,
                  rope_theta,
                  eps
                ) do
    {bq, bk, bv} =
      project_qkv(weights, backbone_prefix, backbone_hidden, num_heads, num_kv_heads, head_dim,
        rope_theta: rope_theta,
        eps: eps,
        offset: 0
      )

    {eq, ek, ev} =
      project_qkv(weights, expert_prefix, expert_hidden, num_heads, num_kv_heads, head_dim,
        rope_theta: rope_theta,
        eps: eps,
        offset: backbone_len
      )

    q = Nx.concatenate([bq, eq], axis: 2)
    k = Nx.concatenate([bk, ek], axis: 2)
    v = Nx.concatenate([bv, ev], axis: 2)

    scale = 1.0 / :math.sqrt(head_dim)
    att_out = Emily.Fast.scaled_dot_product_attention_with_mask(q, k, v, mask, scale: scale)

    expert_len = elem(Nx.shape(expert_hidden), 1)
    backbone_att = Nx.slice_along_axis(att_out, 0, backbone_len, axis: 2)
    expert_att = Nx.slice_along_axis(att_out, backbone_len, expert_len, axis: 2)

    backbone_out =
      merge_heads(backbone_att)
      |> linear_no_bias(weights[backbone_prefix <> "self_attn.o_proj.weight"])

    expert_out =
      merge_heads(expert_att)
      |> linear_no_bias(weights[expert_prefix <> "self_attn.o_proj.weight"])

    backbone_hidden = mlp_residual(weights, backbone_prefix, backbone_hidden, backbone_out, eps)
    expert_hidden = mlp_residual(weights, expert_prefix, expert_hidden, expert_out, eps)

    {backbone_hidden, expert_hidden}
  end

  # Backbone: ordinary self-attention over the prefix only. Expert: query
  # from its own hidden state, key/value RE-PROJECTED from the
  # backbone's ALREADY-ROPE'd key/value (not RoPE'd again), through the
  # expert's OWN k_proj/v_proj (kv_dim -> kv_dim). The expert's query
  # RoPE position is renormalized to start at 0 on cross-attn layers
  # (unlike the continuing-offset self-attn layers above).
  deftransformp cross_attn_layer(
                  weights,
                  backbone_prefix,
                  expert_prefix,
                  backbone_hidden,
                  expert_hidden,
                  mask,
                  backbone_len,
                  num_heads,
                  num_kv_heads,
                  head_dim,
                  rope_theta,
                  eps
                ) do
    {bq, bk, bv} =
      project_qkv(weights, backbone_prefix, backbone_hidden, num_heads, num_kv_heads, head_dim,
        rope_theta: rope_theta,
        eps: eps,
        offset: 0
      )

    prefix_mask = Nx.slice_along_axis(mask, 0, backbone_len, axis: 2)
    prefix_mask = Nx.slice_along_axis(prefix_mask, 0, backbone_len, axis: 3)

    scale = 1.0 / :math.sqrt(head_dim)

    backbone_att =
      Emily.Fast.scaled_dot_product_attention_with_mask(bq, bk, bv, prefix_mask, scale: scale)

    backbone_out =
      merge_heads(backbone_att)
      |> linear_no_bias(weights[backbone_prefix <> "self_attn.o_proj.weight"])

    backbone_hidden = mlp_residual(weights, backbone_prefix, backbone_hidden, backbone_out, eps)

    normed_expert =
      fused_rms_norm(expert_hidden, weights[expert_prefix <> "input_layernorm.weight"], eps: eps)

    eq =
      normed_expert
      |> linear_no_bias(weights[expert_prefix <> "self_attn.q_proj.weight"])
      |> split_heads(num_heads, head_dim)
      |> then(&Emily.Fast.rope(&1, Nx.tensor(0), dims: head_dim, base: rope_theta))

    bk_flat = merge_heads_kv(bk)
    bv_flat = merge_heads_kv(bv)

    ek =
      bk_flat
      |> linear_no_bias(weights[expert_prefix <> "self_attn.k_proj.weight"])
      |> split_heads(num_kv_heads, head_dim)

    ev =
      bv_flat
      |> linear_no_bias(weights[expert_prefix <> "self_attn.v_proj.weight"])
      |> split_heads(num_kv_heads, head_dim)

    expert_len = elem(Nx.shape(expert_hidden), 1)
    expert_mask = Nx.slice_along_axis(mask, backbone_len, expert_len, axis: 2)
    expert_mask = Nx.slice_along_axis(expert_mask, 0, backbone_len, axis: 3)

    expert_att =
      Emily.Fast.scaled_dot_product_attention_with_mask(eq, ek, ev, expert_mask, scale: scale)

    expert_out =
      merge_heads(expert_att)
      |> linear_no_bias(weights[expert_prefix <> "self_attn.o_proj.weight"])

    expert_hidden = mlp_residual(weights, expert_prefix, expert_hidden, expert_out, eps)

    {backbone_hidden, expert_hidden}
  end

  # ------------------------------------------------------------------
  # Shared per-branch helpers.
  # ------------------------------------------------------------------

  deftransformp project_qkv(weights, prefix, hidden, num_heads, num_kv_heads, head_dim, opts) do
    rope_theta = Keyword.fetch!(opts, :rope_theta)
    eps = Keyword.fetch!(opts, :eps)
    offset = Keyword.fetch!(opts, :offset)

    normed = fused_rms_norm(hidden, weights[prefix <> "input_layernorm.weight"], eps: eps)

    q =
      normed
      |> linear_no_bias(weights[prefix <> "self_attn.q_proj.weight"])
      |> split_heads(num_heads, head_dim)
      |> then(&Emily.Fast.rope(&1, Nx.tensor(offset), dims: head_dim, base: rope_theta))

    k =
      normed
      |> linear_no_bias(weights[prefix <> "self_attn.k_proj.weight"])
      |> split_heads(num_kv_heads, head_dim)
      |> then(&Emily.Fast.rope(&1, Nx.tensor(offset), dims: head_dim, base: rope_theta))

    v =
      normed
      |> linear_no_bias(weights[prefix <> "self_attn.v_proj.weight"])
      |> split_heads(num_kv_heads, head_dim)

    {q, k, v}
  end

  deftransformp mlp_residual(weights, prefix, hidden, att_out, eps) do
    hidden = Nx.add(hidden, att_out)

    normed =
      fused_rms_norm(hidden, weights[prefix <> "post_attention_layernorm.weight"], eps: eps)

    gate = linear_no_bias(normed, weights[prefix <> "mlp.gate_proj.weight"])
    up = linear_no_bias(normed, weights[prefix <> "mlp.up_proj.weight"])

    mlp_out =
      silu(gate) |> Nx.multiply(up) |> linear_no_bias(weights[prefix <> "mlp.down_proj.weight"])

    Nx.add(hidden, mlp_out)
  end

  defnp(silu(x), do: x * Nx.sigmoid(x))

  defnp linear_no_bias(x, w) do
    Nx.dot(x, [-1], w, [-1])
  end

  deftransformp fused_rms_norm(x, weight, opts) do
    Emily.Fast.rms_norm(x, weight, opts)
  end

  deftransformp split_heads(x, heads, head_dim) do
    {b, l, _} = Nx.shape(x)
    x |> Nx.reshape({b, l, heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
  end

  deftransformp merge_heads(x) do
    {b, h, l, d} = Nx.shape(x)
    x |> Nx.transpose(axes: [0, 2, 1, 3]) |> Nx.reshape({b, l, h * d})
  end

  # k/v arrive as (B, num_kv_heads, L, head_dim); flatten back to
  # (B, L, num_kv_heads*head_dim) so the expert's k_proj/v_proj
  # (kv_dim -> kv_dim) can re-project them on cross-attn layers.
  deftransformp(merge_heads_kv(x), do: merge_heads(x))
end
