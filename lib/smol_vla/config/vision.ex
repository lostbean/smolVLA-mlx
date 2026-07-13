defmodule SmolVLA.Config.Vision do
  @moduledoc "The frozen SmolVLM2 backbone's SigLIP vision tower."

  defstruct hidden_size: 768,
            num_attention_heads: 12,
            num_hidden_layers: 12,
            intermediate_size: 3072,
            patch_size: 16,
            image_size: 512,
            num_channels: 3,
            layer_norm_eps: 1.0e-6

  @type t :: %__MODULE__{
          hidden_size: pos_integer(),
          num_attention_heads: pos_integer(),
          num_hidden_layers: pos_integer(),
          intermediate_size: pos_integer(),
          patch_size: pos_integer(),
          image_size: pos_integer(),
          num_channels: pos_integer(),
          layer_norm_eps: float()
        }
end
