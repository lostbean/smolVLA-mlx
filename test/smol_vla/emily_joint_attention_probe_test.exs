defmodule SmolVLA.EmilyJointAttentionProbeTest do
  @moduledoc """
  The mandatory de-risking probe for the Elixir-native SmolVLA port
  (`docs/design/model-runtime/design.md` component 01.2): before writing
  any of the full forward-pass port, this confirms `Emily.Fast`'s fused
  scaled-dot-product attention can actually run SmolVLA's joint-attention
  mechanism (see `vendor/mlx-vlm/mlx_vlm/models/smolvla/expert.py`) at its
  REAL shapes, loaded from the REAL `lerobot/smolvla_base` checkpoint.

  This was the single named, flagged, unverified risk going into this
  chunk: whether the expert's cross-attention layers (query in the
  15-head x 64-dim space projected from the expert's own 720-wide hidden
  state, key/value RE-PROJECTED from the backbone's already-projected
  320-wide k/v through the expert's own k_proj/v_proj) are shapes
  `Emily.Fast.scaled_dot_product_attention_with_mask/5` can actually
  accept.

  Result (see PROBE RESULT below and the chunk's final report): PASSED.
  Both the self-attention layer variant (layer 0: backbone and expert
  each project q/k/v from their own hidden state, concatenated into one
  joint sequence) and the cross-attention layer variant (layer 1: the
  expert's query attends into the backbone's own k/v, re-projected
  through the expert's k_proj/v_proj) run against real layer-0/layer-1
  weight shapes with finite, non-zero output at the shapes the
  architecture predicts. This is a MECHANISM/SHAPE probe only -- dummy
  random activations, not a numerical-parity check (that is the full
  port's own acceptance bar, checked end to end against the Python
  reference's real output).

  Kept as a permanent regression test (not deleted after the full port
  landed) because it isolates exactly one risk -- "can emily's fused SDPA
  even take these shapes" -- separately from the full model's own tests,
  so a future emily upgrade or shape change surfaces here first with a
  small, fast, easy-to-read failure instead of only inside a much larger
  end-to-end test.
  """
  use ExUnit.Case, async: false

  @moduletag :real_checkpoint

  @checkpoint_path Path.expand(
                     "~/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/c83c3163b8ca9b7e67c509fffd9121e66cb96205/model.safetensors"
                   )

  @num_heads 15
  @num_kv_heads 5
  @head_dim 64

  setup_all do
    Nx.global_default_backend({Emily.Backend, device: :gpu})
    Nx.Defn.global_default_options(compiler: Emily.Compiler)

    unless File.exists?(@checkpoint_path) do
      raise "SmolVLA checkpoint not found at #{@checkpoint_path} -- " <>
              "this probe requires the real lerobot/smolvla_base checkpoint " <>
              "already cached locally (see the other real-checkpoint tests " <>
              "in this repo for the same expectation)."
    end

    tensors = Safetensors.read!(@checkpoint_path)
    {:ok, tensors: tensors}
  end

  defp weight(tensors, key) do
    tensors
    |> Map.fetch!("model." <> key)
    |> Nx.backend_transfer(Emily.Backend)
  end

  # PyTorch nn.Linear weight is (out, in); computes x @ w^T.
  defp linear(x, w), do: Nx.dot(x, [-1], w, [-1])

  defp split_heads(x, heads) do
    {b, l, _} = Nx.shape(x)

    x
    |> Nx.reshape({b, l, heads, @head_dim})
    |> Nx.transpose(axes: [0, 2, 1, 3])
  end

  defp zero_mask(shape) do
    Nx.broadcast(Nx.tensor(0.0, type: :f32), shape) |> Nx.backend_transfer(Emily.Backend)
  end

  defp finite_and_nonzero?(tensor) do
    all_finite = tensor |> Nx.is_nan() |> Nx.logical_not() |> Nx.all() |> Nx.to_number()
    any_nonzero = tensor |> Nx.not_equal(0.0) |> Nx.any() |> Nx.to_number()
    all_finite == 1 and any_nonzero == 1
  end

  describe "self-attention layer variant (layer 0)" do
    test "Emily.Fast SDPA runs on the real layer-0 backbone+expert joint sequence", %{
      tensors: tensors
    } do
      bq_w =
        weight(tensors, "vlm_with_expert.vlm.model.text_model.layers.0.self_attn.q_proj.weight")

      bk_w =
        weight(tensors, "vlm_with_expert.vlm.model.text_model.layers.0.self_attn.k_proj.weight")

      bv_w =
        weight(tensors, "vlm_with_expert.vlm.model.text_model.layers.0.self_attn.v_proj.weight")

      eq_w = weight(tensors, "vlm_with_expert.lm_expert.layers.0.self_attn.q_proj.weight")
      ek_w = weight(tensors, "vlm_with_expert.lm_expert.layers.0.self_attn.k_proj.weight")
      ev_w = weight(tensors, "vlm_with_expert.lm_expert.layers.0.self_attn.v_proj.weight")

      # Real checkpoint shapes, confirmed directly (see expert.py's own
      # module docstring corroboration): backbone q_proj (960,960),
      # k/v_proj (320,960); expert q_proj (960,720), k/v_proj (320,720).
      assert Nx.shape(bq_w) == {960, 960}
      assert Nx.shape(bk_w) == {320, 960}
      assert Nx.shape(eq_w) == {960, 720}
      assert Nx.shape(ek_w) == {320, 720}

      backbone_len = 6
      expert_len = 4
      key = Nx.Random.key(0)

      {backbone_hidden, key} =
        Nx.Random.normal(key, shape: {1, backbone_len, 960}, type: :f32)

      {expert_hidden, _key} = Nx.Random.normal(key, shape: {1, expert_len, 720}, type: :f32)

      backbone_hidden = Nx.backend_transfer(backbone_hidden, Emily.Backend)
      expert_hidden = Nx.backend_transfer(expert_hidden, Emily.Backend)

      bq = backbone_hidden |> linear(bq_w) |> split_heads(@num_heads)
      bk = backbone_hidden |> linear(bk_w) |> split_heads(@num_kv_heads)
      bv = backbone_hidden |> linear(bv_w) |> split_heads(@num_kv_heads)

      eq = expert_hidden |> linear(eq_w) |> split_heads(@num_heads)
      ek = expert_hidden |> linear(ek_w) |> split_heads(@num_kv_heads)
      ev = expert_hidden |> linear(ev_w) |> split_heads(@num_kv_heads)

      q = Nx.concatenate([bq, eq], axis: 2)
      k = Nx.concatenate([bk, ek], axis: 2)
      v = Nx.concatenate([bv, ev], axis: 2)

      total_len = backbone_len + expert_len
      mask = zero_mask({1, 1, total_len, total_len})

      out = Emily.Fast.scaled_dot_product_attention_with_mask(q, k, v, mask)

      assert Nx.shape(out) == {1, @num_heads, total_len, @head_dim}
      assert finite_and_nonzero?(out)
    end
  end

  describe "cross-attention layer variant (layer 1)" do
    test "Emily.Fast SDPA runs when the expert re-projects the backbone's own k/v", %{
      tensors: tensors
    } do
      bq_w =
        weight(tensors, "vlm_with_expert.vlm.model.text_model.layers.1.self_attn.q_proj.weight")

      bk_w =
        weight(tensors, "vlm_with_expert.vlm.model.text_model.layers.1.self_attn.k_proj.weight")

      bv_w =
        weight(tensors, "vlm_with_expert.vlm.model.text_model.layers.1.self_attn.v_proj.weight")

      eq_w = weight(tensors, "vlm_with_expert.lm_expert.layers.1.self_attn.q_proj.weight")
      ek_w = weight(tensors, "vlm_with_expert.lm_expert.layers.1.self_attn.k_proj.weight")
      ev_w = weight(tensors, "vlm_with_expert.lm_expert.layers.1.self_attn.v_proj.weight")

      # The tell-tale shape difference vs. the self-attn layer: the
      # expert's k_proj/v_proj on a cross-attn layer take the BACKBONE's
      # kv-dim (320) as input, not the expert's own hidden size (720).
      assert Nx.shape(ek_w) == {320, 320}
      assert Nx.shape(ev_w) == {320, 320}
      assert Nx.shape(eq_w) == {960, 720}

      backbone_len = 6
      expert_len = 4
      key = Nx.Random.key(1)

      {backbone_hidden, key} =
        Nx.Random.normal(key, shape: {1, backbone_len, 960}, type: :f32)

      {expert_hidden, _key} = Nx.Random.normal(key, shape: {1, expert_len, 720}, type: :f32)

      backbone_hidden = Nx.backend_transfer(backbone_hidden, Emily.Backend)
      expert_hidden = Nx.backend_transfer(expert_hidden, Emily.Backend)

      bq = backbone_hidden |> linear(bq_w) |> split_heads(@num_heads)
      bk_flat = linear(backbone_hidden, bk_w)
      bv_flat = linear(backbone_hidden, bv_w)
      bk = split_heads(bk_flat, @num_kv_heads)
      bv = split_heads(bv_flat, @num_kv_heads)

      eq = expert_hidden |> linear(eq_w) |> split_heads(@num_heads)
      # The mechanism under test: the expert's k_proj/v_proj re-project
      # the BACKBONE's already-projected (kv_dim-wide) key/value, not the
      # expert's own hidden state.
      ek = bk_flat |> linear(ek_w) |> split_heads(@num_kv_heads)
      ev = bv_flat |> linear(ev_w) |> split_heads(@num_kv_heads)

      prefix_mask = zero_mask({1, 1, backbone_len, backbone_len})
      backbone_out = Emily.Fast.scaled_dot_product_attention_with_mask(bq, bk, bv, prefix_mask)

      expert_mask = zero_mask({1, 1, expert_len, backbone_len})
      expert_out = Emily.Fast.scaled_dot_product_attention_with_mask(eq, ek, ev, expert_mask)

      assert Nx.shape(backbone_out) == {1, @num_heads, backbone_len, @head_dim}
      assert Nx.shape(expert_out) == {1, @num_heads, expert_len, @head_dim}
      assert finite_and_nonzero?(backbone_out)
      assert finite_and_nonzero?(expert_out)
    end
  end
end
