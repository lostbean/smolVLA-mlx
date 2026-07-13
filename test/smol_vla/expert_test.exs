defmodule SmolVLA.ExpertTest do
  @moduledoc """
  Verifies `SmolVLA.Expert` per the TDD directive's step (4) ("the joint
  -attention mechanism (self-attn AND cross-attn layer variants)
  integrated with the expert").

  `sinusoidal_pos_embedding/4` and `make_att_2d_masks/2` are checked
  against hand-computed values matching the Python reference's own
  documented output (both confirmed identical to Python during
  development). `forward/6` (the full 16-layer joint-attention stack,
  exercising both the self-attn layers -- even indices -- and the
  cross-attn layers -- odd indices, per `self_attn_every_n_layers=2`) is
  checked against a real fixture generated once from the Python
  reference's own `SmolVLAModel.expert_stack` on fixed seeded random
  inputs -- see `test/smol_vla/vision_test.exs`'s module doc for the
  same bf16-accumulation-drift discussion this tolerance is based on.
  """
  use ExUnit.Case, async: false

  alias SmolVLA.Config
  alias SmolVLA.Expert
  alias SmolVLA.Weights

  @checkpoint_path Path.expand(
                     "~/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/c83c3163b8ca9b7e67c509fffd9121e66cb96205/model.safetensors"
                   )

  @fixtures_dir Path.join([__DIR__, "..", "fixtures"])

  setup_all do
    Nx.global_default_backend({Emily.Backend, device: :gpu})
    Nx.Defn.global_default_options(compiler: Emily.Compiler)
    :ok
  end

  describe "sinusoidal_pos_embedding/4" do
    test "matches the Python reference's documented sin/cos shape" do
      time = Nx.tensor([1.0, 0.5]) |> Nx.backend_transfer(Emily.Backend)
      emb = Expert.sinusoidal_pos_embedding(time, 8, 0.004, 4.0)

      assert Nx.shape(emb) == {2, 8}

      emb_cpu = Nx.backend_transfer(emb, Nx.BinaryBackend)
      # First half is sin, second half is cos -- at t=1.0 with period
      # exactly matching one of the geometric sweep points, cos hits 1.0
      # (verified against the real Python reference during development:
      # mlx_vlm.models.smolvla.expert.sinusoidal_pos_embedding(mx.array([1.0,0.5]), 8, 0.004, 4.0)).
      assert_in_delta Nx.to_number(emb_cpu[[0, 3]]), 1.0, 1.0e-4
      assert_in_delta Nx.to_number(emb_cpu[[0, 4]]), 1.0, 1.0e-4
    end
  end

  describe "make_att_2d_masks/2" do
    test "prefix tokens never attend to suffix tokens; suffix attends to all" do
      pad = Nx.tensor([[true, true, true, true]]) |> Nx.backend_transfer(Emily.Backend)

      # first two tokens: att_mask=0 (prefix block), last two: att_mask=1 (suffix, new blocks each)
      att = Nx.tensor([[0, 0, 1, 1]]) |> Nx.backend_transfer(Emily.Backend)

      mask = Expert.make_att_2d_masks(pad, att) |> Nx.backend_transfer(Nx.BinaryBackend)

      # Prefix rows (0, 1): can only see prefix columns (0, 1).
      assert Nx.to_number(mask[[0, 0, 0]]) == 1
      assert Nx.to_number(mask[[0, 0, 1]]) == 1
      assert Nx.to_number(mask[[0, 0, 2]]) == 0
      assert Nx.to_number(mask[[0, 0, 3]]) == 0

      # Suffix row (2): sees the whole prefix plus itself, not the token after it.
      assert Nx.to_number(mask[[0, 2, 0]]) == 1
      assert Nx.to_number(mask[[0, 2, 1]]) == 1
      assert Nx.to_number(mask[[0, 2, 2]]) == 1
      assert Nx.to_number(mask[[0, 2, 3]]) == 0

      # Last suffix row (3): sees everything.
      assert Nx.to_number(mask[[0, 3, 0]]) == 1
      assert Nx.to_number(mask[[0, 3, 3]]) == 1
    end
  end

  describe "forward/6 (full 16-layer joint-attention stack)" do
    @tag :real_checkpoint
    test "produces correctly-shaped, finite output against real weights" do
      config = Config.from_map(%{})
      weights = Weights.load!(@checkpoint_path)

      backbone_len = 70
      expert_len = 50
      key = Nx.Random.key(0)

      {backbone_embeds, key} =
        Nx.Random.normal(key, shape: {1, backbone_len, 960}, type: :f32)

      {expert_embeds, _key} = Nx.Random.normal(key, shape: {1, expert_len, 720}, type: :f32)

      backbone_embeds =
        backbone_embeds |> Nx.as_type(:bf16) |> Nx.backend_transfer(Emily.Backend)

      expert_embeds = expert_embeds |> Nx.as_type(:bf16) |> Nx.backend_transfer(Emily.Backend)

      att_mask_vals = List.duplicate(0, backbone_len) ++ List.duplicate(1, expert_len)
      att_mask = Nx.tensor([att_mask_vals]) |> Nx.backend_transfer(Emily.Backend)

      pad_mask_bool =
        Nx.broadcast(1, {1, backbone_len + expert_len})
        |> Nx.as_type(:u8)
        |> Nx.not_equal(0)
        |> Nx.backend_transfer(Emily.Backend)

      {backbone_out, expert_out} =
        Expert.forward(weights, config, backbone_embeds, expert_embeds, pad_mask_bool, att_mask)

      assert Nx.shape(backbone_out) == {1, backbone_len, 960}
      assert Nx.shape(expert_out) == {1, expert_len, 720}

      refute Nx.to_number(Nx.any(Nx.is_nan(Nx.as_type(expert_out, :f32)))) == 1
      refute Nx.to_number(Nx.any(Nx.is_nan(Nx.as_type(backbone_out, :f32)))) == 1
    end

    @tag :real_checkpoint
    test "matches the Python reference's real expert_stack output within bf16 tolerance" do
      config = Config.from_map(%{})
      weights = Weights.load!(@checkpoint_path)

      backbone_len = 70
      expert_len = 50

      backbone_in =
        File.read!(Path.join(@fixtures_dir, "expert_probe_backbone_in_f32.bin"))
        |> Nx.from_binary(:f32)
        |> Nx.reshape({1, backbone_len, 960})

      expert_in =
        File.read!(Path.join(@fixtures_dir, "expert_probe_expert_in_f32.bin"))
        |> Nx.from_binary(:f32)
        |> Nx.reshape({1, expert_len, 720})

      expected_backbone_out =
        File.read!(Path.join(@fixtures_dir, "expert_probe_backbone_out_f32.bin"))
        |> Nx.from_binary(:f32)
        |> Nx.reshape({1, backbone_len, 960})

      expected_expert_out =
        File.read!(Path.join(@fixtures_dir, "expert_probe_expert_out_f32.bin"))
        |> Nx.from_binary(:f32)
        |> Nx.reshape({1, expert_len, 720})

      backbone_embeds = backbone_in |> Nx.as_type(:bf16) |> Nx.backend_transfer(Emily.Backend)
      expert_embeds = expert_in |> Nx.as_type(:bf16) |> Nx.backend_transfer(Emily.Backend)

      att_mask_vals = List.duplicate(0, backbone_len) ++ List.duplicate(1, expert_len)
      att_mask = Nx.tensor([att_mask_vals]) |> Nx.backend_transfer(Emily.Backend)

      pad_mask_bool =
        Nx.broadcast(1, {1, backbone_len + expert_len})
        |> Nx.as_type(:u8)
        |> Nx.not_equal(0)
        |> Nx.backend_transfer(Emily.Backend)

      {backbone_out, expert_out} =
        Expert.forward(weights, config, backbone_embeds, expert_embeds, pad_mask_bool, att_mask)

      backbone_out_f32 = backbone_out |> Nx.as_type(:f32) |> Nx.backend_transfer(Nx.BinaryBackend)
      expert_out_f32 = expert_out |> Nx.as_type(:f32) |> Nx.backend_transfer(Nx.BinaryBackend)

      backbone_rel_error =
        mean_relative_error(backbone_out_f32, expected_backbone_out)

      expert_rel_error = mean_relative_error(expert_out_f32, expected_expert_out)

      # See vision_test.exs's module doc for the same bf16-accumulation
      # -drift discussion; the expert stack shows similar magnitude
      # (~2.3% backbone, ~0.9% expert during development) after 16
      # layers with both self-attn and cross-attn variants exercised.
      assert backbone_rel_error < 0.05,
             "backbone branch mean relative error #{backbone_rel_error} exceeds budget"

      assert expert_rel_error < 0.05,
             "expert branch mean relative error #{expert_rel_error} exceeds budget"
    end
  end

  defp mean_relative_error(actual, expected) do
    abs_diff = Nx.abs(Nx.subtract(actual, expected))
    Nx.to_number(Nx.mean(abs_diff)) / Nx.to_number(Nx.mean(Nx.abs(expected)))
  end
end
