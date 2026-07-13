defmodule SmolVLA.ConfigTest do
  use ExUnit.Case, async: true

  alias SmolVLA.Config

  describe "from_map/1" do
    test "parses the real checkpoint's flat, type-keyed shape (ADR-0006)" do
      raw = %{
        "type" => "smolvla",
        "chunk_size" => 50,
        "n_action_steps" => 50,
        "max_state_dim" => 32,
        "max_action_dim" => 32,
        "num_expert_layers" => 0,
        "self_attn_every_n_layers" => 2,
        "expert_width_multiplier" => 0.75,
        "min_period" => 0.004,
        "max_period" => 4.0,
        "num_steps" => 10,
        "tokenizer_max_length" => 48,
        "num_vlm_layers" => 16,
        "vlm_model_name" => "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
      }

      config = Config.from_map(raw)

      assert config.model_type == "smolvla"
      assert config.chunk_size == 50
      assert config.max_state_dim == 32
      assert config.max_action_dim == 32
      assert config.self_attn_every_n_layers == 2
      assert config.expert_width_multiplier == 0.75
      assert config.num_vlm_layers == 16
      assert config.num_steps == 10
    end

    test "defaults when the config map is empty" do
      config = Config.from_map(%{})
      assert config.model_type == "smolvla"
      assert config.chunk_size == 50
    end

    test "accepts model_type as a fallback discriminator" do
      config = Config.from_map(%{"model_type" => "smolvla"})
      assert config.model_type == "smolvla"
    end

    test "raises loud and local on a non-smolvla type, never silently coerces" do
      assert_raise ArgumentError, ~r/expected "smolvla"/, fn ->
        Config.from_map(%{"type" => "idefics3"})
      end
    end
  end

  describe "action_dim/1" do
    test "aliases max_action_dim" do
      config = Config.from_map(%{"max_action_dim" => 32})
      assert Config.action_dim(config) == 32
    end
  end

  describe "expert sizing (grounded against the real checkpoint's weight shapes)" do
    test "expert_hidden_size/1 is 720 for the real checkpoint's 960-wide backbone at 0.75x" do
      config = Config.from_map(%{})
      assert Config.expert_hidden_size(config) == 720
    end

    test "expert_intermediate_size/1 is 2048, confirmed against lm_expert.layers.*.mlp.gate_proj" do
      config = Config.from_map(%{})
      assert Config.expert_intermediate_size(config) == 2048
    end
  end
end
