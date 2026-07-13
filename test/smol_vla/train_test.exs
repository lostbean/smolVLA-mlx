defmodule SmolVLA.TrainTest do
  @moduledoc """
  Fast, synthetic-data tests for `SmolVLA.Train` -- TDD directive step (2):
  "a minimal single-step training update on a small parameter slice
  (verify loss decreases or at least gradients apply sensibly, on
  synthetic/toy data first -- fast, no real checkpoint needed)".

  Builds a tiny (2-layer, narrow-hidden) toy weights map matching
  `SmolVLA.Weights`' own remapped key scheme -- not the real checkpoint --
  so this suite runs in milliseconds and belongs in the default (non
  -`:real_checkpoint`) test gate.
  """
  use ExUnit.Case, async: false

  alias SmolVLA.Config
  alias SmolVLA.Train

  setup_all do
    Nx.global_default_backend({Emily.Backend, device: :gpu})
    Nx.Defn.global_default_options(compiler: Emily.Compiler)
    :ok
  end

  # A tiny but architecturally faithful config: 2 joint-attention layers
  # (one self-attn, one cross-attn -- self_attn_every_n_layers: 2), narrow
  # hidden sizes, small chunk_size -- fast to run, same key scheme/shape
  # relationships as the real checkpoint.
  defp toy_config do
    %Config{
      chunk_size: 4,
      max_state_dim: 6,
      max_action_dim: 6,
      self_attn_every_n_layers: 2,
      expert_width_multiplier: 0.75,
      min_period: 0.004,
      max_period: 4.0,
      num_vlm_layers: 2,
      text: %Config.Text{
        hidden_size: 16,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        rms_norm_eps: 1.0e-5,
        rope_theta: 100_000.0
      }
    }
  end

  defp rand(key, shape) do
    {t, key} = Nx.Random.normal(key, shape: shape, type: :f32)
    {Nx.multiply(t, 0.02), key}
  end

  # Builds a toy weights map with SmolVLA.Weights' own remapped key scheme
  # (expert_stack.layers.<n>.{backbone,expert}.*, action_in_proj, etc.) at
  # `toy_config/0`'s tiny shapes -- enough for SmolVLA.Expert.forward/6 and
  # SmolVLA.embed_suffix/4 to run end to end.
  defp toy_weights(config) do
    key = Nx.Random.key(42)
    hidden = config.text.hidden_size
    expert_hidden = Config.expert_hidden_size(config)
    num_heads = config.text.num_attention_heads
    num_kv_heads = config.text.num_key_value_heads
    head_dim = div(hidden, num_heads)
    kv_dim = num_kv_heads * head_dim
    intermediate = Config.expert_intermediate_size(config)
    backbone_intermediate = hidden * 3

    {weights, _key} =
      Enum.reduce(0..(config.num_vlm_layers - 1), {%{}, key}, fn layer_idx, {acc, key} ->
        {backbone_layer, key} =
          branch_weights(key, "expert_stack.layers.#{layer_idx}.backbone.", hidden, hidden,
            kv_in: hidden,
            intermediate: backbone_intermediate
          )

        expert_kv_in =
          if config.self_attn_every_n_layers > 0 and
               rem(layer_idx, config.self_attn_every_n_layers) == 0,
             do: expert_hidden,
             else: kv_dim

        {expert_layer, key} =
          branch_weights(key, "expert_stack.layers.#{layer_idx}.expert.", expert_hidden, hidden,
            kv_in: expert_kv_in,
            intermediate: intermediate
          )

        {Map.merge(acc, Map.merge(backbone_layer, expert_layer)), key}
      end)

    {backbone_norm, key} = rand(key, {hidden})
    {expert_norm, key} = rand(key, {expert_hidden})

    {state_proj_w, key} = rand(key, {expert_hidden, config.max_state_dim})
    {state_proj_b, key} = rand(key, {expert_hidden})

    action_time_in = expert_hidden * 2

    {action_in_proj_w, key} = rand(key, {expert_hidden, config.max_action_dim})
    {action_in_proj_b, key} = rand(key, {expert_hidden})
    {action_out_proj_w, key} = rand(key, {config.max_action_dim, expert_hidden})
    {action_out_proj_b, key} = rand(key, {config.max_action_dim})
    {mlp_in_w, key} = rand(key, {expert_hidden, action_time_in})
    {mlp_in_b, key} = rand(key, {expert_hidden})
    {mlp_out_w, key} = rand(key, {expert_hidden, expert_hidden})
    {mlp_out_b, _key} = rand(key, {expert_hidden})

    weights
    |> Map.merge(%{
      "expert_stack.backbone_norm.weight" => backbone_norm,
      "expert_stack.expert_norm.weight" => expert_norm,
      "state_proj.weight" => state_proj_w,
      "state_proj.bias" => state_proj_b,
      "action_in_proj.weight" => action_in_proj_w,
      "action_in_proj.bias" => action_in_proj_b,
      "action_out_proj.weight" => action_out_proj_w,
      "action_out_proj.bias" => action_out_proj_b,
      "action_time_mlp_in.weight" => mlp_in_w,
      "action_time_mlp_in.bias" => mlp_in_b,
      "action_time_mlp_out.weight" => mlp_out_w,
      "action_time_mlp_out.bias" => mlp_out_b
    })
    |> Map.new(fn {k, v} -> {k, Nx.backend_transfer(v, Emily.Backend)} end)
  end

  defp branch_weights(key, prefix, hidden_in, q_hidden, opts) do
    kv_in = Keyword.fetch!(opts, :kv_in)
    intermediate = Keyword.fetch!(opts, :intermediate)
    num_heads = 2
    head_dim = div(q_hidden, num_heads)
    q_dim = num_heads * head_dim
    kv_dim = div(q_hidden, num_heads) * 1

    {input_ln, key} = rand(key, {hidden_in})
    {q_w, key} = rand(key, {q_dim, hidden_in})
    {k_w, key} = rand(key, {kv_dim, kv_in})
    {v_w, key} = rand(key, {kv_dim, kv_in})
    {o_w, key} = rand(key, {hidden_in, q_dim})
    {post_ln, key} = rand(key, {hidden_in})
    {gate_w, key} = rand(key, {intermediate, hidden_in})
    {up_w, key} = rand(key, {intermediate, hidden_in})
    {down_w, key} = rand(key, {hidden_in, intermediate})

    weights = %{
      (prefix <> "input_layernorm.weight") => input_ln,
      (prefix <> "self_attn.q_proj.weight") => q_w,
      (prefix <> "self_attn.k_proj.weight") => k_w,
      (prefix <> "self_attn.v_proj.weight") => v_w,
      (prefix <> "self_attn.o_proj.weight") => o_w,
      (prefix <> "post_attention_layernorm.weight") => post_ln,
      (prefix <> "mlp.gate_proj.weight") => gate_w,
      (prefix <> "mlp.up_proj.weight") => up_w,
      (prefix <> "mlp.down_proj.weight") => down_w
    }

    {weights, key}
  end

  defp toy_batch(config, batch_size) do
    key = Nx.Random.key(7)
    prefix_len = 5

    {prefix_embeds, key} = rand(key, {batch_size, prefix_len, config.text.hidden_size})
    prefix_embeds = Nx.as_type(prefix_embeds, :bf16) |> Nx.backend_transfer(Emily.Backend)

    prefix_pad_mask =
      Nx.broadcast(1, {batch_size, prefix_len})
      |> Nx.as_type(:u8)
      |> Nx.not_equal(0)
      |> Nx.backend_transfer(Emily.Backend)

    prefix_att_mask =
      Nx.broadcast(0, {batch_size, prefix_len})
      |> Nx.as_type(:s32)
      |> Nx.backend_transfer(Emily.Backend)

    {actions, key} = rand(key, {batch_size, config.chunk_size, config.max_action_dim})
    {noise, key} = Train.sample_noise(key, {batch_size, config.chunk_size, config.max_action_dim})
    {time, _key} = Train.sample_time(key, batch_size)

    action_is_pad =
      Nx.broadcast(0, {batch_size, config.chunk_size})
      |> Nx.as_type(:u8)
      |> Nx.not_equal(0)
      |> Nx.backend_transfer(Emily.Backend)

    %{
      prefix_embeds: prefix_embeds,
      prefix_pad_mask: prefix_pad_mask,
      prefix_att_mask: prefix_att_mask,
      actions: Nx.backend_transfer(actions, Emily.Backend),
      action_is_pad: action_is_pad,
      noise: Nx.backend_transfer(noise, Emily.Backend),
      time: Nx.backend_transfer(time, Emily.Backend)
    }
  end

  describe "trainable_keys/2" do
    test "default split: only the action-expert namespace is trainable" do
      config = toy_config()
      weights = toy_weights(config)

      {trainable, frozen} = Train.trainable_keys(weights)

      assert Enum.all?(trainable, fn k ->
               String.contains?(k, ".expert.") or
                 String.starts_with?(k, "expert_stack.expert_norm.") or
                 String.starts_with?(k, "state_proj.") or
                 String.starts_with?(k, "action_")
             end)

      assert Enum.any?(frozen, &String.contains?(&1, ".backbone."))
      assert Enum.any?(frozen, &String.starts_with?(&1, "expert_stack.backbone_norm."))
      assert MapSet.disjoint?(MapSet.new(trainable), MapSet.new(frozen))

      assert MapSet.union(MapSet.new(trainable), MapSet.new(frozen)) ==
               MapSet.new(Map.keys(weights))
    end

    test "full_finetune: true makes every key trainable" do
      config = toy_config()
      weights = toy_weights(config)

      {trainable, frozen} = Train.trainable_keys(weights, full_finetune: true)

      assert Enum.empty?(frozen)
      assert MapSet.new(trainable) == MapSet.new(Map.keys(weights))
    end
  end

  describe "loss/2" do
    test "produces a finite scalar loss and finite, correctly-shaped, correctly-frozen gradients" do
      config = toy_config()
      weights = toy_weights(config)
      {trainable_keys, _frozen} = Train.trainable_keys(weights)
      trainable_set = MapSet.new(trainable_keys)
      batch = toy_batch(config, 2)

      pytree = %{
        weights: Map.new(weights, fn {k, v} -> {k, Nx.as_type(v, :f32)} end),
        batch: batch
      }

      {loss, %{weights: grads}} =
        Nx.Defn.value_and_grad(pytree, &Train.loss(&1, trainable_set, config))

      loss_value = Nx.to_number(loss)
      assert is_number(loss_value)
      assert loss_value == loss_value

      Enum.each(trainable_keys, fn k ->
        g = Map.fetch!(grads, k)
        assert Nx.shape(g) == Nx.shape(weights[k])
        finite = g |> Nx.is_nan() |> Nx.logical_not() |> Nx.all() |> Nx.to_number()
        assert finite == 1, "grad for #{k} contains NaN"
      end)

      {_trainable, frozen_keys} = Train.trainable_keys(weights)

      Enum.each(frozen_keys, fn k ->
        g = Map.fetch!(grads, k)
        all_zero = g |> Nx.equal(0.0) |> Nx.all() |> Nx.to_number()
        assert all_zero == 1, "frozen param #{k} received a nonzero gradient"
      end)
    end

    test "a few Adam steps on fixed synthetic data drive the loss down" do
      config = toy_config()
      weights = toy_weights(config)
      {trainable_keys, _frozen} = Train.trainable_keys(weights)
      trainable_set = MapSet.new(trainable_keys)
      batch = toy_batch(config, 2)

      all_weights_f32 = Map.new(weights, fn {k, v} -> {k, Nx.as_type(v, :f32)} end)

      {init_fn, update_fn} = Polaris.Optimizers.adam(learning_rate: 1.0e-2)

      trainable_params = Map.take(all_weights_f32, trainable_keys)
      opt_state = init_fn.(trainable_params)

      {final_weights, final_loss, first_loss} =
        Enum.reduce(1..8, {all_weights_f32, nil, nil}, fn step, {w, _last, first} ->
          pytree = %{weights: w, batch: batch}

          {loss, %{weights: grads}} =
            Nx.Defn.value_and_grad(pytree, &Train.loss(&1, trainable_set, config))

          trainable_grads = Map.take(grads, trainable_keys)
          trainable_params = Map.take(w, trainable_keys)

          {updates, opt_state} = update_fn.(trainable_grads, opt_state, trainable_params)

          new_trainable =
            Nx.Defn.jit_apply(&Polaris.Updates.apply_updates(&1, &2, %{}), [
              trainable_params,
              updates
            ])

          new_w = Map.merge(w, new_trainable)

          loss_value = Nx.to_number(loss)
          first = first || loss_value

          send(self(), {:opt_state, opt_state})

          if step == 8, do: {new_w, loss_value, first}, else: {new_w, loss_value, first}
        end)

      assert final_loss < first_loss,
             "loss did not decrease: first=#{first_loss} final=#{final_loss}"

      # Frozen weights must be bit-for-bit unchanged across the whole run.
      {_trainable, frozen_keys} = Train.trainable_keys(weights)

      Enum.each(frozen_keys, fn k ->
        assert Nx.to_flat_list(final_weights[k]) == Nx.to_flat_list(all_weights_f32[k])
      end)

      # At least one trainable weight actually moved.
      assert Enum.any?(trainable_keys, fn k ->
               Nx.to_flat_list(final_weights[k]) != Nx.to_flat_list(all_weights_f32[k])
             end)
    end
  end
end
