defmodule SmolVLA.WeightsTest do
  use ExUnit.Case, async: false

  alias SmolVLA.Weights

  @checkpoint_path Path.expand(
                     "~/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/c83c3163b8ca9b7e67c509fffd9121e66cb96205/model.safetensors"
                   )

  setup_all do
    Nx.global_default_backend({Emily.Backend, device: :gpu})
    Nx.Defn.global_default_options(compiler: Emily.Compiler)
    :ok
  end

  describe "load!/1" do
    test "raises loud and local on a missing checkpoint file, never a silent fallback" do
      assert_raise File.Error, fn ->
        Weights.load!("/nonexistent/path/model.safetensors")
      end
    end

    @tag :real_checkpoint
    test "loads and remaps the real checkpoint's keys" do
      weights = Weights.load!(@checkpoint_path)

      assert Nx.shape(weights["expert_stack.layers.0.backbone.self_attn.q_proj.weight"]) ==
               {960, 960}

      assert Nx.shape(weights["expert_stack.layers.0.expert.self_attn.q_proj.weight"]) ==
               {960, 720}

      assert Nx.shape(weights["expert_stack.layers.1.expert.self_attn.k_proj.weight"]) ==
               {320, 320}

      assert Nx.shape(weights["vision_encoder.vision_model.embeddings.patch_embedding.weight"]) ==
               {768, 16, 16, 3}

      assert Nx.shape(weights["vision_encoder.connector.modality_projection.weight"]) ==
               {960, 12288}

      assert Nx.shape(weights["text_embed_tokens.weight"]) == {49280, 960}
      assert Nx.shape(weights["state_proj.weight"]) == {960, 32}
      assert Nx.shape(weights["action_in_proj.weight"]) == {720, 32}
      assert Nx.shape(weights["action_out_proj.weight"]) == {32, 720}
      assert Nx.shape(weights["action_time_mlp_in.weight"]) == {720, 1440}

      refute Enum.any?(Map.keys(weights), &String.contains?(&1, "lm_head"))

      assert Weights.count_expert_layers(weights) == 16
    end
  end
end
