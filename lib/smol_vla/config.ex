defmodule SmolVLA.Config do
  @moduledoc """
  Mirrors `mlx_vlm.models.smolvla.config.SmolVLAConfig` /
  `VisionConfig` / `TextConfig`: parses a real SmolVLA (LeRobot-native)
  `config.json` -- a flat, `"type"`-keyed config, not mlx-vlm's usual
  HF-nested `vision_config`/`text_config` shape (see ADR-0006, which
  applies identically here since it's the same checkpoint file).

  Field names mirror the checkpoint's own LeRobot policy-config keys
  (`chunk_size`, `max_action_dim`, `num_vlm_layers`, ...), matching the
  Python side's own naming choice for the same reason: this is the real
  shape a published SmolVLA checkpoint ships with.

  Per ADR-0004 ("weights-only cross-runtime sharing"): this module is an
  independent reimplementation of the Python `SmolVLAConfig`'s parsing
  logic, sharing no code with it -- only the checkpoint's `config.json`
  file itself crosses the boundary.
  """

  alias SmolVLA.Config.Text
  alias SmolVLA.Config.Vision

  defstruct model_type: "smolvla",
            chunk_size: 50,
            n_action_steps: 50,
            max_state_dim: 32,
            max_action_dim: 32,
            num_expert_layers: nil,
            self_attn_every_n_layers: 2,
            expert_width_multiplier: 0.75,
            min_period: 0.004,
            max_period: 4.0,
            num_steps: 10,
            tokenizer_max_length: 48,
            num_vlm_layers: 16,
            vlm_model_name: "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
            vision: %Vision{},
            text: %Text{},
            input_image_keys: nil

  @type t :: %__MODULE__{
          model_type: String.t(),
          chunk_size: pos_integer(),
          n_action_steps: pos_integer(),
          max_state_dim: pos_integer(),
          max_action_dim: pos_integer(),
          num_expert_layers: non_neg_integer() | nil,
          self_attn_every_n_layers: pos_integer(),
          expert_width_multiplier: float(),
          min_period: float(),
          max_period: float(),
          num_steps: pos_integer(),
          tokenizer_max_length: pos_integer(),
          num_vlm_layers: pos_integer(),
          vlm_model_name: String.t(),
          vision: Vision.t(),
          text: Text.t(),
          input_image_keys: [String.t()] | nil
        }

  @doc """
  Action dimensionality. Aliases `max_action_dim`, the real checkpoint's
  field name, under this port's `action_dim` contract name -- mirrors the
  Python side's `SmolVLAConfig.action_dim` property.
  """
  @spec action_dim(t()) :: pos_integer()
  def action_dim(%__MODULE__{max_action_dim: max_action_dim}), do: max_action_dim

  @doc """
  Parses a raw, already JSON-decoded `config.json` map into a `t()`.

  The real checkpoint's discriminator key is `"type"`, not mlx-vlm's usual
  `"model_type"` (ADR-0006) -- accepts either so a real config.json and a
  hand-built toy config both work, matching the Python side's
  `from_dict`.

  Raises `ArgumentError` on a `"type"`/`"model_type"` other than
  `"smolvla"` -- loud and local, matching the Python side's own
  `ValueError`.
  """
  @spec from_map(map()) :: t()
  def from_map(params) when is_map(params) do
    raw_type = Map.get(params, "type") || Map.get(params, "model_type")
    model_type = raw_type || "smolvla"

    if model_type != "smolvla" do
      raise ArgumentError,
            "SmolVLA.Config received a config with type=#{inspect(model_type)}, expected \"smolvla\"."
    end

    %__MODULE__{
      model_type: model_type,
      chunk_size: Map.get(params, "chunk_size", 50),
      n_action_steps: Map.get(params, "n_action_steps", 50),
      max_state_dim: Map.get(params, "max_state_dim", 32),
      max_action_dim: Map.get(params, "max_action_dim", 32),
      num_expert_layers: Map.get(params, "num_expert_layers"),
      self_attn_every_n_layers: Map.get(params, "self_attn_every_n_layers", 2),
      expert_width_multiplier: Map.get(params, "expert_width_multiplier", 0.75),
      min_period: Map.get(params, "min_period", 0.004),
      max_period: Map.get(params, "max_period", 4.0),
      num_steps: Map.get(params, "num_steps", 10),
      tokenizer_max_length: Map.get(params, "tokenizer_max_length", 48),
      num_vlm_layers: Map.get(params, "num_vlm_layers", 16),
      vlm_model_name:
        Map.get(params, "vlm_model_name", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"),
      vision: %Vision{},
      text: %Text{},
      input_image_keys: nil
    }
  end

  @doc """
  The flow-matching action expert's hidden size: `text.hidden_size *
  expert_width_multiplier`, rounded to the nearest integer -- matches the
  checkpoint's own `lm_expert.layers.*` weight shapes (720 for the real
  checkpoint: `960 * 0.75`).
  """
  @spec expert_hidden_size(t()) :: pos_integer()
  def expert_hidden_size(%__MODULE__{text: text, expert_width_multiplier: mult}) do
    round(text.hidden_size * mult)
  end

  @doc """
  The flow-matching action expert's SwiGLU MLP intermediate size, derived
  from `expert_hidden_size/1` via lerobot's own `get_intermediate_size`
  sizing rule (2/3-scaled, 4x-multiplied, rounded up to a multiple of
  256) -- confirmed directly against the checkpoint's own
  `lm_expert.layers.*.mlp.gate_proj` shape `(2048, 720)`.
  """
  @spec expert_intermediate_size(t()) :: pos_integer()
  def expert_intermediate_size(%__MODULE__{} = config) do
    hidden = expert_hidden_size(config)
    swiglu_intermediate_size(hidden)
  end

  @doc false
  @spec swiglu_intermediate_size(pos_integer(), pos_integer(), pos_integer()) :: pos_integer()
  def swiglu_intermediate_size(hidden_dim, ffn_dim_multiplier \\ 4, multiple_of \\ 256) do
    hidden_dim = trunc(2 * hidden_dim / 3)
    hidden_dim = ffn_dim_multiplier * hidden_dim
    multiple_of * div(hidden_dim + multiple_of - 1, multiple_of)
  end
end
