defmodule SmolVLA.Train do
  @moduledoc """
  The differentiable core of the Elixir-native `FineTuneJob` (component
  01.4): a single-timestep flow-matching regression loss, and the
  frozen-backbone/trainable-action-expert parameter split, both built
  directly on top of `SmolVLA`'s already-accepted forward-pass functions
  (`SmolVLA.embed_suffix/4`, `SmolVLA.Expert.forward/6`) -- no forward-pass
  logic is duplicated here, only composed into a differentiable loss.

  **Training objective, confirmed against LeRobot's own real training code**
  (`lerobot.policies.smolvla.modeling_smolvla.VLAFlowMatching.forward`, read
  in full on 2026-07-13 from this repo's own vendored `.venv` install) --
  NOT backprop through the full multi-step Euler integration loop
  `SmolVLA.infer_action/4` runs at inference time:

    1. Sample one timestep per batch element from `Beta(1.5, 1.0)`, rescaled
       into `[0.001, 1.0]` (`sample_time/2`) -- matches LeRobot's own
       `sample_time`, not a uniform distribution.
    2. Sample noise `~ N(0, 1)`, same shape as the target actions
       (`sample_noise/2`).
    3. Linearly interpolate: `x_t = t * noise + (1 - t) * actions` (the
       "corrupted" input at that timestep) and the constant target velocity
       `u_t = noise - actions` -- both closed-form, no iteration.
    4. Run exactly ONE forward pass: `embed_prefix` (frozen, precomputed
       once per batch since the prefix never depends on the sampled
       timestep) + `SmolVLA.embed_suffix/4` at that single `x_t`/`t` +
       `SmolVLA.Expert.forward/6`'s joint-attention stack + `action_out_proj`
       to get the predicted velocity `v_t`.
    5. `loss = mean((u_t - v_t)^2)` over the real (non-padded) action
       dimensions -- MSE, matching LeRobot's `F.mse_loss`.

  This is the standard flow-matching/rectified-flow training objective
  (a single denoising-step regression), not the iterative sampling
  procedure -- iterating is only how a TRAINED model is later run at
  inference, never how it is trained.

  **Frozen backbone / trainable action expert**: per
  `docs/design/model-runtime/design.md` component 01.4 ("same frozen
  -backbone default as 01.3"), only the action expert's own parameters
  (`expert_stack.layers.*.expert.*`, `expert_stack.expert_norm`,
  `state_proj`, `action_in_proj`, `action_out_proj`,
  `action_time_mlp_{in,out}`) receive gradients by default; the VLM
  backbone (vision tower, embeddings, `expert_stack.layers.*.backbone.*`,
  `expert_stack.backbone_norm`) stays frozen. `trainable_keys/2` computes
  this split directly from a real checkpoint's own weight-key namespace
  (mirrors `SmolVLA.Weights`' own remapped key scheme) rather than
  hardcoding a layer count, so it holds for any real checkpoint's actual
  `num_vlm_layers`. Implemented via `Nx.Defn.Kernel.stop_grad/1` applied
  to every non-trainable tensor INSIDE the single differentiated pytree
  (see `loss/2`'s own comment) -- the cleaner
  parameter-subset-as-a-second-`value_and_grad`-argument approach was
  tried first and does not work under `emily`/`Nx.Defn` here: any tensor
  closed over from outside the differentiated function raises
  (`cannot invoke Nx function because it relies on two incompatible
  tensor implementations: Emily.Backend and Nx.Defn.Expr`) -- confirmed
  directly during this chunk's mandatory de-risking probe (see
  `test/smol_vla/train_grad_probe_test.exs`). Every tensor the loss touches
  must flow through the SAME `value_and_grad` pytree argument, frozen ones
  wrapped in `stop_grad` inside the traced function.
  """

  alias SmolVLA.Config
  alias SmolVLA.Expert

  @doc """
  Splits a real (remapped, per `SmolVLA.Weights`) weights map's keys into
  `{trainable, frozen}` key lists.

  `full_finetune: true` makes every key trainable (component 01.4's "a
  config option for full fine-tuning ... never silently desyncing a run
  from its recorded mode"). The default (`full_finetune: false`) selects
  exactly the flow-matching action-expert's own parameters -- every
  `expert_stack.layers.<n>.expert.*` tensor (any `n`, not a hardcoded
  layer count), `expert_stack.expert_norm.weight`, and the state/action
  projection and time-MLP tensors -- leaving the VLM backbone (vision
  tower, text embeddings, `expert_stack.layers.<n>.backbone.*`,
  `expert_stack.backbone_norm.weight`) frozen.
  """
  @spec trainable_keys(%{String.t() => Nx.Tensor.t()}, keyword()) ::
          {[String.t()], [String.t()]}
  def trainable_keys(weights, opts \\ []) do
    full_finetune = Keyword.get(opts, :full_finetune, false)
    all_keys = Map.keys(weights)

    if full_finetune do
      {all_keys, []}
    else
      Enum.split_with(all_keys, &trainable_key?/1)
    end
  end

  defp trainable_key?(key) do
    String.contains?(key, ".expert.") or
      String.starts_with?(key, "expert_stack.expert_norm.") or
      String.starts_with?(key, "state_proj.") or
      String.starts_with?(key, "action_in_proj.") or
      String.starts_with?(key, "action_out_proj.") or
      String.starts_with?(key, "action_time_mlp_in.") or
      String.starts_with?(key, "action_time_mlp_out.")
  end

  @doc """
  Samples one flow-matching timestep per batch element from LeRobot's own
  `Beta(1.5, 1.0)` distribution, rescaled into `[0.001, 1.0]` -- matches
  `VLAFlowMatching.sample_time` exactly (read from the vendored `lerobot`
  install on 2026-07-13), not a uniform `[0, 1]` sample. `Nx.Random` has no
  built-in Beta sampler, so this uses the standard
  Beta(a,1)-via-inverse-transform identity (`U^(1/a)` for `Beta(a, 1)`,
  since `concentration0: 1.0` degenerates the general two-parameter Beta to
  this closed form) rather than a Gamma-ratio construction, which needs no
  extra machinery and is exact for this specific `(1.5, 1.0)` case.
  """
  @spec sample_time(Nx.Tensor.t(), pos_integer()) :: {Nx.Tensor.t(), Nx.Tensor.t()}
  def sample_time(key, batch_size) do
    {u, key} = Nx.Random.uniform(key, shape: {batch_size}, type: :f32)
    beta_sample = Nx.pow(u, 1.0 / 1.5)
    time = Nx.add(Nx.multiply(beta_sample, 0.999), 0.001)
    {time, key}
  end

  @doc """
  Samples standard-normal noise, same shape as the target actions --
  matches `VLAFlowMatching.sample_noise`.
  """
  @spec sample_noise(Nx.Tensor.t(), tuple()) :: {Nx.Tensor.t(), Nx.Tensor.t()}
  def sample_noise(key, shape) do
    Nx.Random.normal(key, shape: shape, type: :f32)
  end

  @doc """
  The single-timestep flow-matching regression loss for one batch,
  differentiable via `Nx.Defn.value_and_grad/2` against `pytree.weights`.

  `pytree` is a single map (the `value_and_grad` differentiation variable):

      %{
        weights: %{String.t() => Nx.Tensor.t()},   # ALL weights, f32
        batch: %{
          prefix_embeds: Nx.Tensor.t(),   # {batch, prefix_len, text_hidden}
          prefix_pad_mask: Nx.Tensor.t(), # {batch, prefix_len}
          prefix_att_mask: Nx.Tensor.t(), # {batch, prefix_len}
          actions: Nx.Tensor.t(),         # {batch, chunk_size, max_action_dim}
          action_is_pad: Nx.Tensor.t(),   # {batch, chunk_size} boolean
          noise: Nx.Tensor.t(),           # {batch, chunk_size, max_action_dim}
          time: Nx.Tensor.t()             # {batch}
        }
      }

  `trainable_keys` (a plain `MapSet.t(String.t())`, from `trainable_keys/2`)
  is a REGULAR function argument, not part of the `value_and_grad` pytree --
  it holds no tensors, so unlike every actual tensor this function touches
  it is safe to close over normally (`Nx.Defn`'s closure restriction, per
  this module's own moduledoc, applies to TENSORS resolving to a foreign
  backend, not plain Elixir terms).

  `prefix_embeds`/`prefix_pad_mask`/`prefix_att_mask` are precomputed
  ONCE per batch by the caller (via `SmolVLA`'s own `embed_prefix`
  machinery, run through the frozen backbone at inference-equivalent
  precision) since the prefix does not depend on the sampled timestep --
  recomputing it once per denoising step would be correct too but wasteful
  (LeRobot's own training loop makes the same choice: `embed_prefix` runs
  once per training step, not once per timestep, since there is only ONE
  timestep per training step here, unlike inference's multi-step loop).

  `weights` MUST be f32 (grad accumulation dtype); this function casts to
  each real op's own native dtype internally (bf16 for the transformer
  ops, matching `SmolVLA`'s own forward pass) via `Nx.as_type/2` on the
  already-`stop_grad`-marked frozen tensors and the still-differentiable
  trainable ones alike -- casting after (not before) `stop_grad` is safe
  here since `stop_grad` only marks the gradient tape, it does not change
  values.

  Returns a scalar `f32` loss.
  """
  @spec loss(map(), MapSet.t(String.t()), Config.t()) :: Nx.Tensor.t()
  def loss(%{weights: weights, batch: batch}, trainable_keys, %Config{} = config) do
    merged_bf16 =
      Map.new(weights, fn {k, v} ->
        v = if MapSet.member?(trainable_keys, k), do: v, else: Nx.Defn.Kernel.stop_grad(v)
        {k, Nx.as_type(v, :bf16)}
      end)

    prefix_embeds = Nx.Defn.Kernel.stop_grad(batch.prefix_embeds) |> Nx.as_type(:bf16)
    prefix_pad_mask = Nx.Defn.Kernel.stop_grad(batch.prefix_pad_mask)
    prefix_att_mask = Nx.Defn.Kernel.stop_grad(batch.prefix_att_mask)
    actions = Nx.Defn.Kernel.stop_grad(batch.actions) |> Nx.as_type(:f32)
    noise = Nx.Defn.Kernel.stop_grad(batch.noise) |> Nx.as_type(:f32)
    time = Nx.Defn.Kernel.stop_grad(batch.time) |> Nx.as_type(:f32)
    action_is_pad = Nx.Defn.Kernel.stop_grad(batch.action_is_pad)

    time_expanded = time |> Nx.new_axis(1) |> Nx.new_axis(2)

    x_t =
      Nx.add(
        Nx.multiply(time_expanded, noise),
        Nx.multiply(Nx.subtract(1.0, time_expanded), actions)
      )

    u_t = Nx.subtract(noise, actions)

    {suffix_embeds, suffix_pad_mask, suffix_att_mask} =
      SmolVLA.embed_suffix(merged_bf16, config, Nx.as_type(x_t, :bf16), time)

    pad_mask = Nx.concatenate([prefix_pad_mask, suffix_pad_mask], axis: 1)
    att_mask = Nx.concatenate([prefix_att_mask, suffix_att_mask], axis: 1)

    {_backbone_out, expert_out} =
      Expert.forward(merged_bf16, config, prefix_embeds, suffix_embeds, pad_mask, att_mask)

    v_t =
      expert_out
      |> Nx.as_type(:f32)
      |> linear_no_bias(merged_bf16["action_out_proj.weight"] |> Nx.as_type(:f32))
      |> Nx.add(merged_bf16["action_out_proj.bias"] |> Nx.as_type(:f32))

    squared_error = Nx.pow(Nx.subtract(u_t, v_t), 2)

    valid = Nx.subtract(1, Nx.as_type(action_is_pad, :f32)) |> Nx.new_axis(2)
    masked = Nx.multiply(squared_error, valid)

    num_valid =
      valid
      |> Nx.sum()
      |> Nx.multiply(Nx.axis_size(squared_error, 2))
      |> Nx.max(1.0)

    Nx.sum(masked) |> Nx.divide(num_valid)
  end

  defp linear_no_bias(x, w), do: Nx.dot(x, [-1], w, [-1])
end
