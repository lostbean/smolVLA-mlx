defmodule SmolVLATest do
  @moduledoc """
  Verifies the top-level `SmolVLA.infer_action/4` per the TDD directive's
  step (6) ("the full `infer_action` pipeline end to end against the
  real checkpoint, checked for NUMERICAL PARITY against the Python
  implementation's real output on the same real input").

  **Noise**: flow-matching's Euler integration starts from random noise
  neither side's pinned `infer_action/4` interface exposes as an input
  -- a bit-exact comparison therefore needs the SAME starting noise fed
  to both sides. `test/fixtures/e2e_probe_noise_f32.bin` was captured
  directly from the Python reference's own real `mx.random.normal` draw
  during a real `infer_action` call (via a temporary monkeypatch,
  generation-only, not part of the shipped Python source) and is fed to
  the Elixir side through `SmolVLA.infer_action/5`'s test-only fixed
  -noise seam (see that function's own `@doc false`).

  **Tolerance**: measured during development at 0.65% mean relative
  error / 0.008 max absolute difference on real 6-dimensional state,
  real random-content image, real instruction, through the FULL pipeline
  (vision, tokenization, joint attention x16 layers x10 Euler steps,
  action projection) -- tighter than the per-component ~1.5-2.3% bf16
  drift measured in `vision_test.exs`/`expert_test.exs` alone (errors
  partially cancel rather than compound across the full pipeline, which
  is plausible but not something to rely on -- the actual number is what
  matters). A 2% mean-relative-error budget is used below: comfortably
  above the measured 0.65% (headroom for hardware/MLX-version variance)
  while still an order of magnitude tighter than what a real mechanism
  bug would produce (a wrong RoPE offset, a transposed weight, or a
  wrong mask produces divergence in the tens-of-percent range or
  outright NaN/garbage, not a few tenths of a percent -- confirmed
  during development by deliberately observing what those bug classes
  looked like before they were fixed).
  """
  use ExUnit.Case, async: false

  alias SmolVLA.Config

  @checkpoint_dir Path.expand(
                    "~/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/c83c3163b8ca9b7e67c509fffd9121e66cb96205"
                  )

  @fixtures_dir Path.join([__DIR__, "fixtures"])

  setup_all do
    Nx.global_default_backend({Emily.Backend, device: :gpu})
    # `fuse: true` evals each traced forward in Emily.Compiler's
    # `mx::compile`'d mode, fusing the elementwise runs the plain native
    # replay leaves separate -- a small but free win on this
    # GPU-launch-bound forward (~8% warm, measured), reassociating f32
    # within a few ULP (parity MRE moves 0.6458% -> 0.6462%, far inside
    # the 2% budget). This is the shipped inference-path config; the
    # end-to-end parity test below exercises it exactly as a real
    # `infer_action` caller would.
    Nx.Defn.global_default_options(compiler: Emily.Compiler, fuse: true)
    :ok
  end

  describe "load/2" do
    @tag :real_checkpoint
    test "loads the real checkpoint's config, weights, and tokenizer" do
      model = SmolVLA.load(@checkpoint_dir)

      assert %SmolVLA{config: %Config{}, weights: weights, tokenizer: tokenizer} = model
      assert map_size(weights) > 0
      assert %Tokenizers.Tokenizer{} = tokenizer
      assert model.config.chunk_size == 50
      assert model.config.max_action_dim == 32
    end

    test "raises loud and local on a missing checkpoint directory" do
      assert_raise File.Error, fn ->
        SmolVLA.load("/nonexistent/checkpoint/dir")
      end
    end
  end

  describe "infer_action/4 -- shape validation (Fails invariant)" do
    @tag :real_checkpoint
    test "raises before dispatching to emily on an oversized state vector" do
      model = SmolVLA.load(@checkpoint_dir)

      oversized_state = List.duplicate(0.0, model.config.max_state_dim + 1)
      image = Nx.broadcast(0.5, {224, 224, 3})

      assert_raise ArgumentError, ~r/max_state_dim/, fn ->
        SmolVLA.infer_action(model, image, oversized_state, "pick up the cube")
      end
    end

    @tag :real_checkpoint
    test "raises on a non-1D state vector rather than silently reshaping" do
      model = SmolVLA.load(@checkpoint_dir)
      image = Nx.broadcast(0.5, {224, 224, 3})

      assert_raise ArgumentError, ~r/1D state vector/, fn ->
        SmolVLA.infer_action(model, image, [[0.0, 1.0], [2.0, 3.0]], "pick up the cube")
      end
    end
  end

  describe "infer_action/4 -- end-to-end real-checkpoint pipeline" do
    @tag :real_checkpoint
    test "produces a correctly-shaped, finite action chunk against the real checkpoint" do
      model = SmolVLA.load(@checkpoint_dir)

      image = Nx.broadcast(0.5, {224, 224, 3})
      state = List.duplicate(0.0, 6)

      action_chunk = SmolVLA.infer_action(model, image, state, "pick up the cube")

      assert Nx.shape(action_chunk) == {50, 32}
      refute Nx.to_number(Nx.any(Nx.is_nan(Nx.as_type(action_chunk, :f32)))) == 1
    end

    @tag :real_checkpoint
    test "matches the Python reference's real infer_action output within tolerance, given the same noise" do
      model = SmolVLA.load(@checkpoint_dir)

      image =
        File.read!(Path.join(@fixtures_dir, "e2e_probe_image_f32.bin"))
        |> Nx.from_binary(:f32)
        |> Nx.reshape({224, 224, 3})

      state =
        File.read!(Path.join(@fixtures_dir, "e2e_probe_state_f32.bin"))
        |> Nx.from_binary(:f32)
        |> Nx.to_flat_list()

      instruction = File.read!(Path.join(@fixtures_dir, "e2e_probe_instruction.txt"))

      noise =
        File.read!(Path.join(@fixtures_dir, "e2e_probe_noise_f32.bin"))
        |> Nx.from_binary(:f32)
        |> Nx.reshape({1, 50, 32})

      expected =
        File.read!(Path.join(@fixtures_dir, "e2e_probe_action_chunk_f32.bin"))
        |> Nx.from_binary(:f32)
        |> Nx.reshape({50, 32})

      actual =
        SmolVLA.infer_action(model, image, state, instruction, noise)
        |> Nx.as_type(:f32)
        |> Nx.backend_transfer(Nx.BinaryBackend)

      assert Nx.shape(actual) == {50, 32}

      abs_diff = Nx.abs(Nx.subtract(actual, expected))
      max_abs_diff = Nx.to_number(Nx.reduce_max(abs_diff))

      mean_relative_error =
        Nx.to_number(Nx.mean(abs_diff)) / Nx.to_number(Nx.mean(Nx.abs(expected)))

      assert mean_relative_error < 0.02,
             "end-to-end mean relative error #{mean_relative_error} exceeds the 2% budget " <>
               "(max abs diff #{max_abs_diff}) -- see this module's own doc for the tolerance rationale"
    end

    @tag :real_checkpoint
    test "reports real measured wall-clock latency against the 100ms budget" do
      model = SmolVLA.load(@checkpoint_dir)
      image = Nx.broadcast(0.5, {224, 224, 3})
      state = List.duplicate(0.0, 6)

      # warm up (first call includes JIT trace overhead, not representative
      # of steady-state latency)
      SmolVLA.infer_action(model, image, state, "pick up the cube")

      {elapsed_us, _action_chunk} =
        :timer.tc(fn -> SmolVLA.infer_action(model, image, state, "pick up the cube") end)

      elapsed_ms = elapsed_us / 1000

      # Real number, reported regardless of outcome (per this chunk's own
      # acceptance criterion) -- NOT asserted against the 100ms budget,
      # since the measured latency during development (~1.2-1.4s warm)
      # is over budget; see the final chunk report for the full analysis
      # (this is a real, structural finding, not a flaky-test situation).
      IO.puts(
        "\n  [SmolVLA latency] warm infer_action/4: #{Float.round(elapsed_ms, 1)}ms (100ms budget)"
      )

      assert elapsed_ms > 0
    end
  end
end
