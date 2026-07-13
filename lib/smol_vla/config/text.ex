defmodule SmolVLA.Config.Text do
  @moduledoc "The frozen SmolVLM2 backbone's language tower."

  defstruct hidden_size: 960,
            intermediate_size: 2560,
            num_attention_heads: 15,
            num_key_value_heads: 5,
            num_hidden_layers: 16,
            rms_norm_eps: 1.0e-5,
            vocab_size: 49280,
            rope_theta: 100_000.0

  @type t :: %__MODULE__{
          hidden_size: pos_integer(),
          intermediate_size: pos_integer(),
          num_attention_heads: pos_integer(),
          num_key_value_heads: pos_integer(),
          num_hidden_layers: pos_integer(),
          rms_norm_eps: float(),
          vocab_size: pos_integer(),
          rope_theta: float()
        }
end
