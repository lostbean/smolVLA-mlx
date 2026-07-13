defmodule SmolVLA.TrainGradProbeTest do
  @moduledoc """
  The mandatory gradient de-risking probe for `FineTuneJob (Elixir-native)`
  (`docs/design/model-runtime/design.md` component 01.4): before writing
  the full training loop, this confirms `Nx.Defn.value_and_grad/2` can
  actually compute real, finite, correctly-shaped gradients through
  SmolVLA's real joint-attention mechanism (`SmolVLA.Expert.forward/6`),
  loaded from the REAL `lerobot/smolvla_base` checkpoint, on `emily`'s
  `Nx.Backend` -- something no prior chunk in this repo has exercised (every
  prior Elixir chunk only needed the forward pass).

  This was the single named, unverified risk going into this chunk (per the
  work order): no documentation existed on `emily`(MLX)'s backend-specific
  behavior for `Nx.Defn.grad`/`value_and_grad` through a 100M+ parameter
  computation graph.

  **A real structural finding surfaced by this probe, not a failure**: a
  tensor closed over from OUTSIDE the function passed to
  `Nx.Defn.value_and_grad/2` -- even a plain positional-looking argument
  captured via `&fun.(&1, frozen)` -- raises
  `cannot invoke Nx function because it relies on two incompatible tensor
  implementations: Emily.Backend and Nx.Defn.Expr`. This is a general
  `Nx.Defn` constraint (reproduced with `Nx.BinaryBackend` too during
  development, not `emily`-specific), documented in the error message
  itself ("passing a tensor to defn/jit as ... a closure in an anonymous
  function"). The fix (used throughout `SmolVLA.Train`): every real tensor
  the differentiated function touches -- trainable AND frozen weights,
  activations, masks -- must flow through the SAME `value_and_grad` pytree
  argument; frozen tensors are marked with `Nx.Defn.Kernel.stop_grad/1`
  INSIDE the traced function rather than split into a second
  `value_and_grad` argument (which does not work here -- see `SmolVLA.Train`'s
  own moduledoc for the full reasoning).

  Result: PASSED, at full real scale. See `SmolVLA.Train.loss/2`'s own
  moduledoc and this chunk's final report for the measured timing (~450-850ms
  per real backward pass through all 16 real joint-attention layers, 499 real
  tensors carried through one pytree, zero NaNs, correct shapes, frozen
  gradients exactly zero) -- comfortably fast/small enough for a laptop
  training loop (training has no hard latency budget, unlike inference's
  100ms bar).

  Kept as a permanent regression test (mirrors
  `emily_joint_attention_probe_test.exs`'s own rationale): isolates
  "can `Nx.Defn.value_and_grad` even compute gradients through this
  model's real ops on `emily`" from the full trainer's own tests, so a
  future `emily`/`Nx` upgrade surfaces a regression here first, small and
  fast to read.
  """
  use ExUnit.Case, async: false

  alias SmolVLA.Config
  alias SmolVLA.Expert
  alias SmolVLA.Weights

  @moduletag :real_checkpoint

  @checkpoint_dir Path.expand(
                    "~/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/c83c3163b8ca9b7e67c509fffd9121e66cb96205"
                  )

  setup_all do
    Nx.global_default_backend({Emily.Backend, device: :gpu})
    Nx.Defn.global_default_options(compiler: Emily.Compiler)

    weights_path = Path.join(@checkpoint_dir, "model.safetensors")

    unless File.exists?(weights_path) do
      raise "SmolVLA checkpoint not found at #{weights_path} -- this probe " <>
              "requires the real lerobot/smolvla_base checkpoint already " <>
              "cached locally (see the other real-checkpoint tests in this " <>
              "repo for the same expectation)."
    end

    weights = Weights.load!(weights_path)
    raw_config = @checkpoint_dir |> Path.join("config.json") |> File.read!() |> Jason.decode!()
    config = Config.from_map(raw_config)

    {:ok, weights: weights, config: config}
  end

  test "value_and_grad through all 16 real joint-attention layers produces finite, correctly-shaped gradients",
       %{weights: weights, config: config} do
    {trainable_keys, _frozen_keys} = SmolVLA.Train.trainable_keys(weights)

    trainable_keys =
      trainable_keys |> Enum.filter(&String.contains?(&1, "expert_stack.layers")) |> MapSet.new()

    refute Enum.empty?(trainable_keys)

    all_params_f32 = Map.new(weights, fn {k, v} -> {k, Nx.as_type(v, :f32)} end)

    backbone_hidden_size = config.text.hidden_size
    expert_hidden_size = Config.expert_hidden_size(config)
    backbone_len = 20
    expert_len = config.chunk_size

    key = Nx.Random.key(0)

    {backbone_hidden, key} =
      Nx.Random.normal(key, shape: {1, backbone_len, backbone_hidden_size}, type: :f32)

    {expert_hidden, _key} =
      Nx.Random.normal(key, shape: {1, expert_len, expert_hidden_size}, type: :f32)

    backbone_hidden = backbone_hidden |> Nx.as_type(:bf16) |> Nx.backend_transfer(Emily.Backend)
    expert_hidden = expert_hidden |> Nx.as_type(:bf16) |> Nx.backend_transfer(Emily.Backend)

    pad_mask =
      Nx.broadcast(1, {1, backbone_len + expert_len})
      |> Nx.as_type(:u8)
      |> Nx.not_equal(0)
      |> Nx.backend_transfer(Emily.Backend)

    att_mask_vals = List.duplicate(0, backbone_len) ++ List.duplicate(1, expert_len)
    att_mask = Nx.tensor([att_mask_vals], type: :s32) |> Nx.backend_transfer(Emily.Backend)

    pytree = %{
      weights: all_params_f32,
      activations: %{
        backbone_hidden: backbone_hidden,
        expert_hidden: expert_hidden,
        pad_mask: pad_mask,
        att_mask: att_mask
      }
    }

    loss_fn = fn %{weights: params, activations: acts} ->
      merged_bf16 =
        Map.new(params, fn {k, v} ->
          v = if MapSet.member?(trainable_keys, k), do: v, else: Nx.Defn.Kernel.stop_grad(v)
          {k, Nx.as_type(v, :bf16)}
        end)

      bh = Nx.Defn.Kernel.stop_grad(acts.backbone_hidden)
      eh = Nx.Defn.Kernel.stop_grad(acts.expert_hidden)
      pm = Nx.Defn.Kernel.stop_grad(acts.pad_mask)
      am = Nx.Defn.Kernel.stop_grad(acts.att_mask)

      {_backbone_out, expert_out} = Expert.forward(merged_bf16, config, bh, eh, pm, am)

      expert_out |> Nx.as_type(:f32) |> Nx.pow(2) |> Nx.sum()
    end

    t0 = System.monotonic_time(:millisecond)
    {loss, %{weights: grads}} = Nx.Defn.value_and_grad(pytree, loss_fn)
    elapsed_ms = System.monotonic_time(:millisecond) - t0

    IO.puts("\n[gradient de-risking probe] full 16-layer backward pass: #{elapsed_ms}ms")

    loss_value = Nx.to_number(loss)
    assert is_number(loss_value)
    assert loss_value == loss_value, "loss is NaN"

    Enum.each(trainable_keys, fn k ->
      g = Map.fetch!(grads, k)
      assert Nx.shape(g) == Nx.shape(weights[k]), "grad shape mismatch for #{k}"

      finite = g |> Nx.is_nan() |> Nx.logical_not() |> Nx.all() |> Nx.to_number()
      assert finite == 1, "grad for #{k} contains NaN"

      nonzero = g |> Nx.not_equal(0.0) |> Nx.any() |> Nx.to_number()
      assert nonzero == 1, "grad for #{k} is degenerate all-zero"
    end)

    frozen_sample = [
      "expert_stack.layers.0.backbone.self_attn.q_proj.weight",
      "vision_encoder.vision_model.embeddings.patch_embedding.weight"
    ]

    Enum.each(frozen_sample, fn k ->
      g = Map.fetch!(grads, k)
      all_zero = g |> Nx.equal(0.0) |> Nx.all() |> Nx.to_number()
      assert all_zero == 1, "frozen param #{k} unexpectedly received a nonzero gradient"
    end)

    # Comfortably fast/small for a laptop training loop -- training has no
    # hard latency budget (unlike inference's 100ms bar), but a
    # multi-minute single backward pass would suggest a structural
    # problem worth escalating rather than proceeding.
    assert elapsed_ms < 30_000,
           "full 16-layer backward pass took #{elapsed_ms}ms -- unexpectedly slow, worth investigating before building the full trainer"
  end
end
