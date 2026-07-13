defmodule SmolVLA.VisionTest do
  @moduledoc """
  Verifies `SmolVLA.Vision` two ways per the TDD directive's step (2)
  ("vision tower alone (verify output shape/sanity against a toy or the
  real checkpoint)"):

    * shape/sanity against the real checkpoint with a synthetic input
      (no fixture needed);
    * numerical parity against a real, fixed Python reference output
      (`test/fixtures/vision_probe_*`, generated once from the
      already-accepted Python `SmolVLAModel.vision_encoder` on a fixed
      seeded random image -- see the module doc's own discussion of the
      ~1.5% mean relative error this comparison expects from bf16
      accumulation drift between two independent implementations on the
      same MLX backend, confirmed NOT to be a mechanism bug by a
      separate all-f32 control comparison recorded there).
  """
  use ExUnit.Case, async: false

  alias SmolVLA.Config
  alias SmolVLA.Weights

  @checkpoint_path Path.expand(
                     "~/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/c83c3163b8ca9b7e67c509fffd9121e66cb96205/model.safetensors"
                   )

  @fixtures_dir Path.join([__DIR__, "..", "fixtures"])

  setup_all do
    Nx.global_default_backend({Emily.Backend, device: :gpu})
    Nx.Defn.global_default_options(compiler: Emily.Compiler)
    :ok
  end

  @tag :real_checkpoint
  test "forward/3 produces the expected shape against the real checkpoint" do
    config = Config.from_map(%{})
    weights = Weights.load!(@checkpoint_path)

    pixel_values =
      Nx.broadcast(0.0, {1, 512, 512, 3})
      |> Nx.as_type(:f32)
      |> Nx.backend_transfer(Emily.Backend)

    out = SmolVLA.Vision.forward(weights, config, pixel_values)

    # (image_size / patch_size)^2 / scale_factor^2 = (512/16)^2 / 16 = 64
    assert Nx.shape(out) == {1, 64, 960}
    refute Nx.to_number(Nx.any(Nx.is_nan(Nx.as_type(out, :f32)))) == 1
  end

  @tag :real_checkpoint
  test "forward/3 matches the Python reference's real output within bf16 accumulation tolerance" do
    config = Config.from_map(%{})
    weights = Weights.load!(@checkpoint_path)

    image_bin = File.read!(Path.join(@fixtures_dir, "vision_probe_image_f32.bin"))
    expected_bin = File.read!(Path.join(@fixtures_dir, "vision_probe_output_bf16_f32.bin"))

    pixel_values =
      image_bin
      |> Nx.from_binary(:f32)
      |> Nx.reshape({1, 512, 512, 3})
      |> Nx.backend_transfer(Emily.Backend)

    expected =
      expected_bin
      |> Nx.from_binary(:f32)
      |> Nx.reshape({1, 64, 960})

    actual =
      weights
      |> SmolVLA.Vision.forward(config, pixel_values)
      |> Nx.as_type(:f32)
      |> Nx.backend_transfer(Nx.BinaryBackend)

    assert Nx.shape(actual) == Nx.shape(expected)

    abs_diff = Nx.abs(Nx.subtract(actual, expected))
    mean_abs_diff = Nx.to_number(Nx.mean(abs_diff))
    mean_abs_expected = Nx.to_number(Nx.mean(Nx.abs(expected)))
    mean_relative_error = mean_abs_diff / mean_abs_expected

    # ~1.5% mean relative error observed during development on this same
    # fixture (bf16 accumulation drift across 12 independently-implemented
    # transformer layers on the same MLX backend -- see the module doc).
    # 5% gives headroom for platform/MLX-version variance while still
    # catching a real mechanism regression, which produces order-of-
    # magnitude-larger divergence (an all-f32 control comparison during
    # development matched to float32 rounding noise, ruling out a
    # mechanism bug at this tolerance).
    assert mean_relative_error < 0.05,
           "vision tower mean relative error #{mean_relative_error} exceeds the 5% bf16-drift budget"
  end
end
