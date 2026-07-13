defmodule SmolVLA.PreprocessingTest do
  @moduledoc """
  Verifies `SmolVLA.Preprocessing.resize_with_pad/4` against real,
  fixed Python-reference fixtures (`test/fixtures/resize_*`, generated
  once from `mlx_vlm.models.smolvla.smolvla.resize_with_pad`) covering
  both cases the real function branches on:

    * upsampling (224x224 -> 512x512, antialiasing never triggers --
      `ControlLoop`'s own placeholder observation is this shape);
    * downsampling (640x560 -> 512x512, antialiasing DOES trigger on
      both axes -- exercises the Gaussian-blur branch).
  """
  use ExUnit.Case, async: true

  alias SmolVLA.Preprocessing

  @fixtures_dir Path.join([__DIR__, "..", "fixtures"])

  defp read_fixture(name, shape) do
    Path.join(@fixtures_dir, name)
    |> File.read!()
    |> Nx.from_binary(:f32)
    |> Nx.reshape(shape)
  end

  test "upsampling (224x224 -> 512x512) matches the Python reference bit-exactly" do
    image = read_fixture("resize_up_in_f32.bin", {224, 224, 3})
    expected = read_fixture("resize_up_out_f32.bin", {512, 512, 3})

    actual = Preprocessing.resize_with_pad(image, 512, 512, 0.0)

    assert Nx.shape(actual) == {512, 512, 3}
    max_diff = Nx.to_number(Nx.reduce_max(Nx.abs(Nx.subtract(actual, expected))))
    assert max_diff < 1.0e-5
  end

  test "downsampling (640x560 -> 512x512, antialiasing) matches the Python reference" do
    image = read_fixture("resize_down_in_f32.bin", {640, 560, 3})
    expected = read_fixture("resize_down_out_f32.bin", {512, 512, 3})

    actual = Preprocessing.resize_with_pad(image, 512, 512, 0.0)

    assert Nx.shape(actual) == {512, 512, 3}
    max_diff = Nx.to_number(Nx.reduce_max(Nx.abs(Nx.subtract(actual, expected))))
    assert max_diff < 1.0e-5
  end
end
