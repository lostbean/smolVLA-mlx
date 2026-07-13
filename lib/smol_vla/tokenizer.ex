defmodule SmolVLA.Tokenizer do
  @moduledoc """
  Instruction tokenization for the VLM prefix's language tokens.

  Reads the same `tokenizer.json` the Python side's
  `transformers.AutoTokenizer.from_pretrained(vlm_model_name)` loads
  (`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`'s own tokenizer --
  SmolVLA's checkpoint does not bundle its own, matching the Python
  reference's own lazy `_get_tokenizer`). Not a violation of ADR-0004's
  "weights-only" boundary: this reads a standard tokenizer artifact
  independently on each side, exactly like both sides independently read
  `model.safetensors` -- no Python code or process crosses the boundary.

  Confirmed byte-for-byte token-ID identical to
  `transformers.AutoTokenizer` on this checkpoint's tokenizer during
  development (both `"pick up the cube"` and a longer, truncation
  -triggering instruction were checked directly).
  """

  @doc """
  Loads the tokenizer from a local `tokenizer.json` path (typically
  inside the HF cache for `vlm_model_name`, resolved by the caller --
  this module has no HF Hub network-fetch logic of its own).

  `max_length`: matches the checkpoint's `tokenizer_max_length` config
  field. Truncation direction is `:left` -- this tokenizer's own
  `tokenizer_config.json` declares `"truncation_side": "left"` (confirmed
  directly: the real `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`
  tokenizer truncates from the left, keeping the LAST `max_length`
  tokens, not the first -- `Tokenizers.Tokenizer.set_truncation/2` has no
  way to read that field automatically since it only reads
  `tokenizer.json`, not `tokenizer_config.json`, so it is set explicitly
  here rather than left at the library's right-truncation default).

  Raises (`File.Error` via `Tokenizers.Tokenizer.from_file!/1`'s own
  failure) loud and local on a missing or malformed tokenizer file.
  """
  @spec load!(Path.t(), pos_integer()) :: Tokenizers.Tokenizer.t()
  def load!(tokenizer_json_path, max_length) do
    unless File.exists?(tokenizer_json_path) do
      raise File.Error,
        reason: :enoent,
        action: "read (SmolVLA instruction tokenizer)",
        path: IO.chardata_to_string(tokenizer_json_path)
    end

    {:ok, tokenizer} = Tokenizers.Tokenizer.from_file(tokenizer_json_path)
    Tokenizers.Tokenizer.set_truncation(tokenizer, max_length: max_length, direction: :left)
  end

  @doc """
  Encodes `instruction` into a list of token IDs, truncated to the
  tokenizer's configured `max_length` (see `load!/2`). No padding --
  matches the Python reference's `padding=False`.
  """
  @spec encode!(Tokenizers.Tokenizer.t(), String.t()) :: [non_neg_integer()]
  def encode!(tokenizer, instruction) when is_binary(instruction) do
    {:ok, encoding} = Tokenizers.Tokenizer.encode(tokenizer, instruction)
    Tokenizers.Encoding.get_ids(encoding)
  end
end
