defmodule FineTuneJob do
  @moduledoc """
  The Elixir-native `FineTuneJob` adapter (`docs/design/model-runtime/design.md`
  component 01.4): fine-tunes SmolVLA's flow-matching action expert against
  real LeRobotDataset-format episodes, via `Nx.Defn.value_and_grad` (see
  `SmolVLA.Train`) and `Polaris`'s Adam optimizer -- no Axon graph-building
  DSL, no code shared with the Python `finetune_job.job.FineTuneJob`
  (component 01.3, ADR-0004 "weights-only cross-runtime sharing").

  **Cutover gate** (component 01.4's own note, ADR-0005): this module
  produces an EVALUATED CANDIDATE, not the production trainer -- promoted
  only once a task-performance-parity check (issue 08, not this chunk)
  shows it is not meaningfully worse than the Python trainer. Nothing here
  asserts or assumes this trainer is "the real one".

  **Interface** (component 01.4, exact):

      FineTuneJob.run(checkpoint_path, episodes, output_path) :: FineTuneJob.t()
      FineTuneJob.resume(checkpoint_path) :: FineTuneJob.t()

  **Episodes**: `%FineTuneJob.Episodes{root: path}` -- a real LeRobotDataset
  v3.0 directory, read via `SmolVLA.Dataset` (this module's own independent
  episode-loading, see that module's moduledoc for the real on-disk format
  and the video-decoding dependency this chunk had to resolve). Carries no
  provenance field (real robot vs. simulation), matching component 01.3's
  own `Episodes` and the CONTEXT term "Episode": nothing here branches on
  where an episode came from.

  **Default training path**: VLM backbone frozen, only the action expert's
  own parameters updated (`SmolVLA.Train.trainable_keys/2`'s default
  split) -- matches 01.3's own default. `full_finetune: true` trains every
  parameter instead, recorded in this run's own metadata sidecar so a run
  and its checkpoint are never silently inconsistent about which mode
  produced it (component 01.4's "Invariants held").

  **Checkpointing** (own format -- component 01.4 does not need to match
  01.3's training-STATE format, only the final safetensors WEIGHTS shape,
  per ADR-0004): `<output_path>/checkpoints/<step>/` holds
  `model.safetensors` (the FULL checkpoint -- frozen + trainable tensors
  alike, under the checkpoint's own ORIGINAL `model.vlm_with_expert.*`
  key scheme, structurally identical to `lerobot/smolvla_base`'s own
  file and to the Python trainer's own output -- see `write_checkpoint!/2`),
  `config.json` (copied from the source checkpoint, unchanged -- the
  architecture doesn't change during fine-tuning), and
  `training_state.json` (this run's own step count, optimizer state,
  and a SHA-256 checksum of the weights file, for corruption detection on
  resume). `<output_path>/checkpoints/last` symlinks the most recent step
  dir. `<output_path>/finetune_job_meta.json` records this run's identity
  (mirrors the Python trainer's own sidecar of the same name and
  intent).

  **Fails**: `resume/1` structurally validates a checkpoint
  (`validate_checkpoint!/1` -- required files present, safetensors header
  parses, checksum matches) BEFORE ever using it, raising
  `CorruptCheckpointError` rather than silently continuing from or
  restarting over a corrupt checkpoint -- same loud/local failure shape as
  component 01.3.
  """

  alias SmolVLA.Dataset
  alias SmolVLA.Train
  alias SmolVLA.Weights

  defmodule Episodes do
    @moduledoc """
    A real LeRobotDataset v3.0 directory to fine-tune against. Mirrors the
    Python trainer's own `Episodes` value (component 01.3): carries no
    provenance field (real robot vs. simulation) -- see that module's own
    doc for why there is nothing here to distinguish.
    """
    @enforce_keys [:root]
    defstruct [:root]

    @type t :: %__MODULE__{root: Path.t()}
  end

  defmodule CorruptCheckpointError do
    @moduledoc """
    Raised by `FineTuneJob.resume/1` when a checkpoint directory fails
    structural validation -- missing required files, a safetensors header
    that won't parse, or a checksum mismatch. Never silently continued
    from (component 01.4's "Fails" requirement).
    """
    defexception [:message]
  end

  @enforce_keys [:run_id, :output_path, :full_finetune, :checkpoint_path, :dataset_root, :step]
  defstruct [:run_id, :output_path, :full_finetune, :checkpoint_path, :dataset_root, :step]

  @type t :: %__MODULE__{
          run_id: String.t(),
          output_path: Path.t(),
          full_finetune: boolean(),
          checkpoint_path: Path.t(),
          dataset_root: Path.t(),
          step: non_neg_integer()
        }

  @metadata_filename "finetune_job_meta.json"

  # ------------------------------------------------------------------
  # Public interface (component 01.4, exact per the work order):
  #   FineTuneJob.run(checkpoint_path, episodes, output_path) -> FineTuneJob
  #   FineTuneJob.resume(checkpoint_path) -> FineTuneJob
  # ------------------------------------------------------------------

  @doc """
  Fine-tunes `checkpoint_path` (a real SmolVLA/LeRobot-native checkpoint
  directory, loadable via `SmolVLA.load/2`) against `episodes` (a real
  LeRobotDataset v3.0 directory), writing checkpoints under `output_path`.

  Options:

    * `:steps` (default `20`) -- number of training steps. Deliberately a
      much smaller default than the Python trainer's own 20,000 (component
      01.3's default) -- this is a laptop-scale candidate trainer, not
      tuned for a full production run (out of this chunk's scope; the
      parity gate, issue 08, decides tuning).
    * `:batch_size` (default `2`)
    * `:learning_rate` (default `1.0e-4`)
    * `:full_finetune` (default `false`) -- train every parameter instead
      of only the action expert.
    * `:save_every` (default: every step) -- checkpoint-write frequency.
    * `:seed` (default derived from `System.system_time/0`) -- the
      `Nx.Random` key seed driving batch sampling, noise, and timestep
      sampling. Exposed for reproducible tests, not part of the pinned
      3-arity interface's own callers.

  Returns a `t()` whose identity (`run_id`) persists across a later
  `resume/1` of one of this run's own checkpoints (component 01.4's
  "Invariants held").
  """
  @spec run(Path.t(), Episodes.t(), Path.t(), keyword()) :: t()
  def run(checkpoint_path, %Episodes{} = episodes, output_path, opts \\ []) do
    Nx.global_default_backend({Emily.Backend, device: :gpu})
    Nx.Defn.global_default_options(compiler: Emily.Compiler)

    steps = Keyword.get(opts, :steps, 20)
    batch_size = Keyword.get(opts, :batch_size, 2)
    learning_rate = Keyword.get(opts, :learning_rate, 1.0e-4)
    full_finetune = Keyword.get(opts, :full_finetune, false)
    save_every = Keyword.get(opts, :save_every, 1)
    seed = Keyword.get(opts, :seed, System.system_time())

    run_id = random_run_id()

    model = SmolVLA.load(checkpoint_path)

    {_weights, raw_key_map} =
      Weights.load_with_raw_keys!(Path.join(checkpoint_path, "model.safetensors"))

    raw_tensors = Safetensors.read!(Path.join(checkpoint_path, "model.safetensors"))

    weights_f32 = Map.new(model.weights, fn {k, v} -> {k, Nx.as_type(v, :f32)} end)
    {trainable_keys, _frozen} = Train.trainable_keys(model.weights, full_finetune: full_finetune)
    trainable_set = MapSet.new(trainable_keys)

    dataset = Dataset.open(episodes.root)

    {optimizer_init_fn, optimizer_update_fn} =
      Polaris.Optimizers.adam(learning_rate: learning_rate)

    trainable_params0 = Map.take(weights_f32, trainable_keys)
    opt_state0 = optimizer_init_fn.(trainable_params0)

    job = %__MODULE__{
      run_id: run_id,
      output_path: Path.expand(output_path),
      full_finetune: full_finetune,
      checkpoint_path: Path.expand(checkpoint_path),
      dataset_root: Path.expand(episodes.root),
      step: 0
    }

    write_run_metadata(job)

    key = Nx.Random.key(seed)

    {final_weights, _opt_state, _key, final_step} =
      Enum.reduce(1..steps, {weights_f32, opt_state0, key, 0}, fn step,
                                                                  {weights, opt_state, key, _} ->
        {batch, key} = sample_batch(model, dataset, batch_size, key)

        pytree = %{weights: weights, batch: batch}

        {loss, %{weights: grads}} =
          Nx.Defn.value_and_grad(pytree, &Train.loss(&1, trainable_set, model.config))

        trainable_grads = Map.take(grads, trainable_keys)
        trainable_params = Map.take(weights, trainable_keys)

        {updates, new_opt_state} =
          optimizer_update_fn.(trainable_grads, opt_state, trainable_params)

        new_trainable =
          Nx.Defn.jit_apply(&Polaris.Updates.apply_updates(&1, &2, %{}), [
            trainable_params,
            updates
          ])

        new_weights = Map.merge(weights, new_trainable)

        require Logger
        Logger.info("FineTuneJob #{run_id}: step #{step}/#{steps} loss=#{Nx.to_number(loss)}")

        if rem(step, save_every) == 0 or step == steps do
          write_checkpoint!(
            job,
            step,
            new_weights,
            raw_key_map,
            raw_tensors,
            new_opt_state,
            trainable_keys
          )
        end

        {new_weights, new_opt_state, key, step}
      end)

    _ = final_weights

    %{job | step: final_step}
  end

  @doc """
  Resumes an interrupted run from its last checkpoint (a step-checkpoint
  directory under some `<output_path>/checkpoints/<step>/`, or that
  directory's `last` symlink sibling).

  Validates the checkpoint structurally (`validate_checkpoint!/1`) BEFORE
  ever using it -- a missing file, an unparseable safetensors header, or a
  checksum mismatch raises `CorruptCheckpointError` rather than silently
  continuing from or restarting over corruption (component 01.4's "Fails"
  requirement). Reads this run's own `finetune_job_meta.json` sidecar
  (walking up from the checkpoint dir, mirroring the Python trainer's own
  `_read_metadata`) to recover the original run's identity
  (`run_id`, dataset root, training mode) so `resume/1`'s returned `t()`
  represents "the same fine-tuning job" (component 01.4's "Invariants
  held").

  Continuing training from a resumed checkpoint (as opposed to merely
  validating and re-identifying it) is driven by calling `run/4` again
  with `checkpoint_path` pointed at this resumed step-checkpoint's own
  `pretrained_model`-equivalent weights and a bumped `:steps` -- `resume/1`
  itself only re-establishes identity and validates, matching component
  01.3's own `resume/1` shape (which likewise re-invokes the underlying
  trainer rather than manually splicing back into a live training loop).
  """
  @spec resume(Path.t()) :: t()
  def resume(checkpoint_path) do
    checkpoint_dir = Path.expand(checkpoint_path)
    validate_checkpoint!(checkpoint_dir)

    training_state =
      checkpoint_dir |> Path.join("training_state.json") |> File.read!() |> Jason.decode!()

    meta = read_run_metadata(checkpoint_dir)

    %__MODULE__{
      run_id: meta["run_id"] || random_run_id(),
      output_path: Path.expand(meta["output_path"] || checkpoint_dir),
      full_finetune: Map.get(meta, "full_finetune", false),
      checkpoint_path: meta["checkpoint_path"] || checkpoint_dir,
      dataset_root: meta["dataset_root"] || "",
      step: training_state["step"]
    }
  end

  # ------------------------------------------------------------------
  # Batch construction.
  # ------------------------------------------------------------------

  # Samples `batch_size` real (image, state, action, instruction) frames
  # from the real dataset (a fresh random episode + in-episode frame index
  # per element, i.i.d. -- simple random sampling rather than sequential
  # per-episode batching, adequate for this candidate trainer's own
  # laptop-scale step counts; a smarter sampler is not required by this
  # chunk's own acceptance bar), builds each one's prefix embedding via
  # `SmolVLA.embed_prefix/7` (run through the frozen backbone at its own
  # native precision, OUTSIDE the differentiable region -- see
  # `SmolVLA.Train.loss/3`'s own moduledoc on why), pads chunk_size (real
  # per-frame actions, not real multi-frame action CHUNKS -- see this
  # function's own note below), and samples this step's noise/timestep.
  defp sample_batch(model, dataset, batch_size, key) do
    total_episodes = Dataset.total_episodes(dataset)

    {episode_indices, key} = random_ints(key, batch_size, total_episodes)

    samples =
      Enum.map(episode_indices, fn episode_index ->
        frames = Dataset.frames(dataset, episode_index)
        frame = Enum.random(frames)
        {frames, frame}
      end)

    prefix_entries =
      Enum.map(samples, fn {_frames, frame} ->
        padded_state =
          Nx.tensor(frame.state, type: :f32)
          |> then(fn s ->
            state_dim = Nx.axis_size(s, 0)
            Nx.pad(s, 0.0, [{0, model.config.max_state_dim - state_dim, 0}])
          end)
          |> Nx.new_axis(0)
          |> Nx.backend_transfer(Emily.Backend)

        # `SmolVLA.prepare_images/2` zero-pads to the checkpoint's full
        # declared camera count (3 for `lerobot/smolvla_base`) with
        # `is_real: false` dummy images, matching `infer_action/4`'s own
        # single-image-in/multi-camera-slot-checkpoint contract. Training
        # takes only the REAL first entry, deliberately dropping the
        # dummy camera slots here -- a real, structural finding from this
        # chunk's own real-data integration run: those dummy slots
        # produce prefix tokens whose `pad_mask`/`att_mask` entries make
        # their OWN query rows fully masked-out (attending to nothing,
        # since a masked-out image's tokens are excluded as both query
        # AND key), and backpropagating through `Emily.Fast`'s fused
        # attention kernel with a fully `-inf` row produces NaN gradients
        # -- reproduced directly in isolation (a minimal
        # `Emily.Fast.scaled_dot_product_attention_with_mask` probe with
        # one deliberately all-masked row NaNs on `value_and_grad`, while
        # every row having at least one attendable position does not).
        # `infer_action/4` never hits this because it never backprops.
        # Dropping the dummy slots (never differentiating through a
        # padding-only block at all) is the same fix flow-matching
        # training conceptually needs anyway -- LeRobot's own reference
        # training loop only ever supplies real camera frames per its
        # `input_features`, never a call with fewer real cameras than the
        # checkpoint declares.
        {all_images, _all_masks} = SmolVLA.prepare_images(model.config, frame.image)
        real_image = hd(all_images)

        SmolVLA.embed_prefix(
          model.weights,
          model.config,
          model.tokenizer,
          [real_image],
          [true],
          frame.task,
          padded_state
        )
      end)

    prefix_embeds = prefix_entries |> Enum.map(&elem(&1, 0)) |> Nx.concatenate(axis: 0)
    prefix_pad_mask = prefix_entries |> Enum.map(&elem(&1, 1)) |> Nx.concatenate(axis: 0)
    prefix_att_mask = prefix_entries |> Enum.map(&elem(&1, 2)) |> Nx.concatenate(axis: 0)

    # `SmolVLA.Dataset.Frame` carries one real per-FRAME action (the
    # dataset's own `action` column, shape `{action_dim}`), not a
    # multi-step action CHUNK -- the real per-episode action-chunk
    # construction LeRobot's own dataloader does (gathering the next
    # `chunk_size` consecutive frames' actions, padding at an episode's
    # tail) is exactly the kind of "exact loss function implementation
    # detail" the work order leaves to this chunk's own judgment; this
    # trainer instead broadcasts the single sampled frame's action across
    # `chunk_size` (a real, if simplified, training target -- the flow
    # -matching objective still regresses a real velocity field against a
    # real target, just not a temporally-extended one). Every frame is
    # therefore "valid" (`action_is_pad` all-false) -- there is no
    # padding to mask since there is no multi-step gather.
    chunk_size = model.config.chunk_size
    max_action_dim = model.config.max_action_dim

    actions =
      samples
      |> Enum.map(fn {_frames, frame} ->
        action = pad_action(frame.action, max_action_dim)
        action |> Nx.tensor(type: :f32) |> Nx.new_axis(0) |> Nx.tile([1, chunk_size, 1])
      end)
      |> Nx.concatenate(axis: 0)
      |> Nx.backend_transfer(Emily.Backend)

    action_is_pad =
      Nx.broadcast(0, {batch_size, chunk_size})
      |> Nx.as_type(:u8)
      |> Nx.not_equal(0)
      |> Nx.backend_transfer(Emily.Backend)

    {noise, key} = Train.sample_noise(key, {batch_size, chunk_size, max_action_dim})
    {time, key} = Train.sample_time(key, batch_size)

    batch = %{
      prefix_embeds: prefix_embeds,
      prefix_pad_mask: prefix_pad_mask,
      prefix_att_mask: prefix_att_mask,
      actions: actions,
      action_is_pad: action_is_pad,
      noise: Nx.backend_transfer(noise, Emily.Backend),
      time: Nx.backend_transfer(time, Emily.Backend)
    }

    {batch, key}
  end

  defp pad_action(action, max_action_dim) do
    dim = length(action)

    if dim >= max_action_dim do
      Enum.take(action, max_action_dim)
    else
      action ++ List.duplicate(0.0, max_action_dim - dim)
    end
  end

  defp random_ints(key, count, max_exclusive) do
    {values, key} = Nx.Random.randint(key, 0, max_exclusive, shape: {count}, type: :s64)
    {Nx.to_flat_list(values), key}
  end

  # ------------------------------------------------------------------
  # Checkpoint writing.
  # ------------------------------------------------------------------

  defp write_checkpoint!(
         job,
         step,
         weights_f32,
         raw_key_map,
         raw_tensors,
         opt_state,
         trainable_keys
       ) do
    step_dir = Path.join([job.output_path, "checkpoints", Integer.to_string(step)])
    File.mkdir_p!(step_dir)

    out_tensors =
      Map.new(raw_tensors, fn {raw_key, original_tensor} ->
        remapped_key = Enum.find(Map.keys(raw_key_map), &(raw_key_map[&1] == raw_key))

        value =
          case remapped_key && weights_f32[remapped_key] do
            nil ->
              original_tensor

            new_value ->
              new_value
              |> unremap_patch_embedding(raw_key)
              |> Nx.as_type(Nx.type(original_tensor))
              |> Nx.backend_transfer(Nx.BinaryBackend)
          end

        {raw_key, value}
      end)

    weights_path = Path.join(step_dir, "model.safetensors")
    Safetensors.write!(weights_path, out_tensors)

    config_src = Path.join(job.checkpoint_path, "config.json")
    if File.exists?(config_src), do: File.cp!(config_src, Path.join(step_dir, "config.json"))

    optimizer_tensors =
      Map.new(trainable_keys, fn k ->
        %{mu: mu, nu: nu} = fetch_adam_state(opt_state, k)
        {k, {mu, nu}}
      end)
      |> Enum.flat_map(fn {k, {mu, nu}} -> [{"mu.#{k}", mu}, {"nu.#{k}", nu}] end)
      |> Map.new(fn {k, v} -> {k, Nx.backend_transfer(v, Nx.BinaryBackend)} end)

    optimizer_path = Path.join(step_dir, "optimizer_state.safetensors")
    Safetensors.write!(optimizer_path, optimizer_tensors)

    checksum = file_sha256(weights_path)

    training_state = %{
      "step" => step,
      "run_id" => job.run_id,
      "full_finetune" => job.full_finetune,
      "weights_checksum_sha256" => checksum
    }

    Path.join(step_dir, "training_state.json")
    |> File.write!(Jason.encode!(training_state, pretty: true))

    last_link = Path.join([job.output_path, "checkpoints", "last"])
    File.rm(last_link)
    File.ln_s!(Path.absname(step_dir), last_link)

    :ok
  end

  # `SmolVLA.Weights.remap_rest/2`'s vision patch-embedding case is the
  # ONE remapping that is not a pure rename -- it also transposes the
  # PyTorch conv2d weight layout `(out, in, kH, kW)` into this port's own
  # `(out, kH, kW, in)` layout on the way IN. Writing an updated tensor
  # back under that same raw key must reverse the SAME transpose, or the
  # checkpoint's own patch-embedding weight comes back out with a
  # different shape than the source file's -- confirmed directly: this
  # chunk's real integration run caught exactly this shape mismatch
  # before the reverse-transpose existed (`bf16[768][3][16][16]` instead
  # of the source's `bf16[768][16][16][3]`).
  defp unremap_patch_embedding(
         value,
         "model.vlm_with_expert.vlm.model.vision_model.embeddings.patch_embedding.weight"
       ) do
    Nx.transpose(value, axes: [0, 3, 1, 2])
  end

  defp unremap_patch_embedding(value, _raw_key), do: value

  # `Polaris.Optimizers.adam/1` composes `scale_by_adam |> scale_by_learning_rate`
  # -- `scale_by_learning_rate` is the OUTERMOST/last-applied combinator in
  # that pipe, and `Polaris.Updates.stateful/3` PREPENDS (`Tuple.insert_at
  # (state, 0, ...)`) each combinator's own local state ahead of its
  # parent's, so the state tuple's element order is innermost-first: index
  # 0 is `scale_by_learning_rate`'s own (empty, `%{scale: ...}`) state,
  # index 1 is `scale_by_adam`'s real `%{mu:, nu:, count:}` state --
  # confirmed directly by inspecting a real `init_fn.(params)` result
  # rather than assumed. Polaris exposes no public accessor for "the adam
  # moment estimates for key k", so this reads the tuple shape directly.
  defp fetch_adam_state(opt_state, key) do
    {_learning_rate_state, adam_state} = opt_state
    %{mu: mu_map, nu: nu_map} = adam_state
    %{mu: Map.fetch!(mu_map, key), nu: Map.fetch!(nu_map, key)}
  end

  defp file_sha256(path) do
    path
    |> File.stream!(2048)
    |> Enum.reduce(:crypto.hash_init(:sha256), &:crypto.hash_update(&2, &1))
    |> :crypto.hash_final()
    |> Base.encode16(case: :lower)
  end

  # ------------------------------------------------------------------
  # Checkpoint validation (resume/1's own "Fails" requirement).
  # ------------------------------------------------------------------

  @doc """
  Structurally validates a step-checkpoint directory before `resume/1`
  ever uses it: required files present, `model.safetensors`' header
  parses and declares at least one tensor, `training_state.json` parses
  with an integer `step`, and the recorded SHA-256 checksum matches the
  real on-disk weights file -- raises `CorruptCheckpointError` otherwise.
  Public so a caller (or a future issue) can pre-flight a checkpoint
  without going through the full `resume/1` (matches the Python trainer's
  own `validate_checkpoint` being a public module function, not just an
  internal step of `resume`).
  """
  @spec validate_checkpoint!(Path.t()) :: :ok
  def validate_checkpoint!(checkpoint_dir) do
    unless File.dir?(checkpoint_dir) do
      raise CorruptCheckpointError,
        message: "checkpoint directory does not exist: #{checkpoint_dir}"
    end

    required = [
      Path.join(checkpoint_dir, "model.safetensors"),
      Path.join(checkpoint_dir, "training_state.json")
    ]

    missing = Enum.reject(required, &File.regular?/1)

    unless missing == [] do
      raise CorruptCheckpointError,
        message:
          "checkpoint at #{checkpoint_dir} is missing required file(s): #{inspect(missing)}"
    end

    weights_path = Path.join(checkpoint_dir, "model.safetensors")
    validate_safetensors_header!(weights_path)

    training_state_path = Path.join(checkpoint_dir, "training_state.json")
    training_state = validate_training_state!(training_state_path)

    expected_checksum = training_state["weights_checksum_sha256"]

    if expected_checksum do
      actual_checksum = file_sha256(weights_path)

      unless actual_checksum == expected_checksum do
        raise CorruptCheckpointError,
          message:
            "checkpoint weight file #{weights_path} failed checksum validation " <>
              "(expected #{expected_checksum}, got #{actual_checksum}) -- refusing to " <>
              "silently resume from what may be a corrupted or truncated file"
      end
    end

    :ok
  end

  defp validate_safetensors_header!(path) do
    case File.open(path, [:read, :raw]) do
      {:ok, file} ->
        result =
          with {:ok, <<header_size::unsigned-64-integer-little>>} <- :file.read(file, 8),
               {:ok, header_json} <- :file.read(file, header_size),
               {:ok, decoded} <- Jason.decode(header_json) do
            keys = decoded |> Map.drop(["__metadata__"]) |> Map.keys()

            if keys == [] do
              {:error, "declares zero tensors"}
            else
              :ok
            end
          else
            _ -> {:error, "header did not parse as a valid safetensors header"}
          end

        File.close(file)

        case result do
          :ok ->
            :ok

          {:error, reason} ->
            raise CorruptCheckpointError,
              message: "checkpoint weight file #{path} failed to parse as safetensors: #{reason}"
        end

      {:error, reason} ->
        raise CorruptCheckpointError,
          message: "checkpoint weight file #{path} could not be opened: #{inspect(reason)}"
    end
  end

  defp validate_training_state!(path) do
    case path |> File.read!() |> Jason.decode() do
      {:ok, %{"step" => step} = decoded} when is_integer(step) ->
        decoded

      {:ok, decoded} ->
        raise CorruptCheckpointError,
          message:
            "training state file #{path} has a missing/non-integer 'step' field: #{inspect(decoded)}"

      {:error, reason} ->
        raise CorruptCheckpointError,
          message: "training state file #{path} is not valid JSON: #{inspect(reason)}"
    end
  end

  # ------------------------------------------------------------------
  # Run identity / metadata sidecar.
  # ------------------------------------------------------------------

  defp random_run_id, do: :crypto.strong_rand_bytes(16) |> Base.encode16(case: :lower)

  defp write_run_metadata(job) do
    File.mkdir_p!(job.output_path)

    metadata = %{
      "run_id" => job.run_id,
      "full_finetune" => job.full_finetune,
      "checkpoint_path" => job.checkpoint_path,
      "dataset_root" => job.dataset_root,
      "output_path" => job.output_path
    }

    Path.join(job.output_path, @metadata_filename)
    |> File.write!(Jason.encode!(metadata, pretty: true))
  end

  # Walks up from a step-checkpoint dir looking for this module's own
  # `finetune_job_meta.json` sidecar, written at `output_path` by `run/4`
  # -- `output_path` is an ancestor of every step-checkpoint dir this
  # module writes under it (`output_path/checkpoints/<step>/`), so this
  # looks in the checkpoint dir itself, its parent, and its grandparent.
  # Mirrors the Python trainer's own `_read_metadata` walk-up shape.
  defp read_run_metadata(checkpoint_dir) do
    candidates = [
      Path.join(checkpoint_dir, @metadata_filename),
      checkpoint_dir |> Path.dirname() |> Path.join(@metadata_filename),
      checkpoint_dir |> Path.dirname() |> Path.dirname() |> Path.join(@metadata_filename)
    ]

    Enum.find_value(candidates, %{}, fn path ->
      if File.regular?(path) do
        case path |> File.read!() |> Jason.decode() do
          {:ok, decoded} -> decoded
          _ -> nil
        end
      end
    end)
  end
end
