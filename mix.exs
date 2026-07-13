defmodule ControlLoop.MixProject do
  use Mix.Project

  def project do
    [
      app: :control_loop,
      version: "0.1.0",
      elixir: "~> 1.18",
      start_permanent: Mix.env() == :prod,
      elixirc_paths: elixirc_paths(Mix.env()),
      deps: deps()
    ]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger],
      mod: {ControlLoop.Application, []}
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      # Pure-Erlang ZMTP client for the ZeroMQ fallback adapter (ADR-0008) --
      # no NIF, no libzmq system dependency.
      {:chumak, "~> 1.5"},
      # MessagePack wire format for the ZeroMQ fallback adapter (ADR-0007).
      {:msgpax, "~> 2.4"},
      # Elixir bindings + Nx.Backend for Apple MLX -- the emily-native
      # infer_action adapter (ADR-0003, model-runtime design component 01.2).
      # Precompiled NIF, no C++ build step.
      {:emily, "~> 1.0"},
      # Reads the real checkpoint's model.safetensors into Nx tensors.
      # emily itself has no safetensors loader (see component 01.2).
      {:safetensors, "~> 0.1.3"},
      # Instruction tokenization: reads the same tokenizer.json the Python
      # side's transformers.AutoTokenizer.from_pretrained(vlm_model_name)
      # loads (HuggingFaceTB/SmolVLM2-500M-Video-Instruct's own tokenizer
      # -- SmolVLA's checkpoint does not bundle its own). Not a violation
      # of ADR-0004's "weights-only" boundary: this reads a standard
      # tokenizer artifact independently on each side, exactly like both
      # sides independently read model.safetensors -- no Python code or
      # process crosses the boundary.
      {:tokenizers, "~> 0.5"},
      # Reads LeRobotDataset v3.0 frame-data parquet files (episodes'
      # action/state/index columns) -- the Nx ecosystem's own DataFrame
      # library (Rust NIF over Polars, no Python), same "own the format,
      # not the training logic" boundary as safetensors/tokenizers above.
      # Used by SmolVLA.Dataset (see finetune_job's episode-loading).
      {:explorer, "~> 0.12"},
      # Optimizer implementations for the Nx ecosystem (Adam), extracted
      # from Axon's graph-building DSL into a standalone package with no
      # dependency on Axon itself -- see FineTuneJob's own moduledoc for
      # why Axon's DSL doesn't fit this already-hand-written Nx.Defn model.
      {:polaris, "~> 0.1"}
    ]
  end
end
