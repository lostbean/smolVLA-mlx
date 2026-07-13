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
      {:msgpax, "~> 2.4"}
    ]
  end
end
