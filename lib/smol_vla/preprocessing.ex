defmodule SmolVLA.Preprocessing do
  @moduledoc """
  Image preprocessing: resize-with-pad (antialiased bilinear resize, then
  top/left-only padding) plus `[-1, 1]` range normalization -- mirrors
  `mlx_vlm.models.smolvla.smolvla.resize_with_pad` and the relevant
  `(H, W, C)` branch of `mlx_vlm.models.interpolate.resize_bilinear`
  (Gaussian-blur antialiasing + bilinear interpolation), an independent
  reimplementation against `Nx`/`emily` (ADR-0004).

  Antialiasing (`gaussian_blur_axis`) only triggers when the resize is a
  DOWNSAMPLE on that axis (`new_size < old_size`) -- ported faithfully
  here since a real observation's image is not guaranteed to be smaller
  than `image_size` (512) in every deployment; `ControlLoop`'s own
  placeholder observation is 224x224 (an upsample, so antialiasing is a
  no-op there), but a higher-resolution camera would downsample and
  exercise this path for real.
  """

  @doc """
  Resizes `image` (`{H, W, C}`, any numeric dtype) preserving aspect
  ratio (matching the larger of the two required scale factors), then
  pads on the TOP/LEFT only with `pad_value` to reach exactly `{height,
  width, C}` -- mirrors lerobot's own `resize_with_pad`
  (`F.pad(..., (pad_width, 0, pad_height, 0))`).
  """
  @spec resize_with_pad(Nx.Tensor.t(), pos_integer(), pos_integer(), float()) :: Nx.Tensor.t()
  def resize_with_pad(image, height, width, pad_value \\ 0.0) do
    {cur_height, cur_width, _channels} = Nx.shape(image)
    ratio = max(cur_width / width, cur_height / height)
    resized_height = trunc(cur_height / ratio)
    resized_width = trunc(cur_width / ratio)

    resized = resize_bilinear(image, resized_height, resized_width, antialias: true)

    pad_height = max(0, height - resized_height)
    pad_width = max(0, width - resized_width)

    Nx.pad(resized, pad_value, [{pad_height, 0, 0}, {pad_width, 0, 0}, {0, 0, 0}])
  end

  @doc """
  Bilinear-interpolates `image` (`{H, W, C}`) to `{new_height, new_width,
  C}`. When `antialias: true` (default) and an axis is being
  downsampled, applies a Gaussian blur on that axis first (heuristic
  sigma `(1/scale - 1) / 2`), matching
  `mlx_vlm.models.interpolate.resize_bilinear`'s antialias branch.
  """
  @spec resize_bilinear(Nx.Tensor.t(), pos_integer(), pos_integer(), keyword()) :: Nx.Tensor.t()
  def resize_bilinear(image, new_height, new_width, opts \\ []) do
    antialias = Keyword.get(opts, :antialias, true)
    {h_in, w_in, _channels} = Nx.shape(image)

    resized = image

    resized =
      if antialias and new_height < h_in do
        sigma_y = (h_in / new_height - 1) / 2.0
        if sigma_y > 0, do: gaussian_blur_axis(resized, sigma_y, 0), else: resized
      else
        resized
      end

    resized =
      if antialias and new_width < w_in do
        sigma_x = (w_in / new_width - 1) / 2.0
        if sigma_x > 0, do: gaussian_blur_axis(resized, sigma_x, 1), else: resized
      else
        resized
      end

    bilinear_interpolate(resized, new_height, new_width)
  end

  # 1D Gaussian blur along `axis` (0 or 1) of a {H, W, C} tensor, edge
  # -padded, sliding-window-summed -- mirrors `interpolate.py`'s
  # `gaussian_blur_axis`.
  defp gaussian_blur_axis(image, sigma, axis) do
    radius = trunc(3 * sigma)

    if radius < 1 do
      image
    else
      xs = Enum.to_list(-radius..radius)
      kernel_vals = Enum.map(xs, fn x -> :math.exp(-(x * x) / (2 * sigma * sigma)) end)
      kernel_sum = Enum.sum(kernel_vals)
      kernel = Enum.map(kernel_vals, &(&1 / kernel_sum))

      padded = pad_edge(image, axis, radius)

      axis_size = elem(Nx.shape(image), axis)

      kernel
      |> Enum.with_index()
      |> Enum.reduce(Nx.broadcast(0.0, Nx.shape(image)), fn {k_val, i}, acc ->
        window = Nx.slice_along_axis(padded, i, axis_size, axis: axis)
        Nx.add(acc, Nx.multiply(window, k_val))
      end)
    end
  end

  # Nx.pad has no `mode: :edge` (constant only) -- replicate it by
  # concatenating repeated edge slices, matching mx.pad(..., mode="edge").
  defp pad_edge(image, axis, radius) do
    axis_size = elem(Nx.shape(image), axis)
    first = Nx.slice_along_axis(image, 0, 1, axis: axis)
    last = Nx.slice_along_axis(image, axis_size - 1, 1, axis: axis)

    left_pad =
      if radius > 0, do: Nx.concatenate(List.duplicate(first, radius), axis: axis), else: nil

    right_pad =
      if radius > 0, do: Nx.concatenate(List.duplicate(last, radius), axis: axis), else: nil

    pieces = Enum.reject([left_pad, image, right_pad], &is_nil/1)
    Nx.concatenate(pieces, axis: axis)
  end

  # Bilinear interpolation, align_corners=false (`half_pixel_centers`),
  # matching `interpolate.py`'s `bilinear_interpolate` for the `(H, W,
  # C)` case (rank 3, `extra_dims = 1`).
  #
  # The per-axis sample geometry (floor/ceil indices + interpolation
  # weights) depends ONLY on the {in, out} size pair, not on pixel
  # values, so it is built once per pair as small 1-D tensors (length
  # new_h / new_w, not the new_h*new_w product) and memoized in the
  # process dictionary. The gather itself is separable -- two sequential
  # `Nx.take`s (row axis, then column axis) reproduce the outer-product
  # `image[row_floor][:, col_floor]` grid exactly -- so no giant paired
  # index list is ever materialized. This replaces the previous version,
  # which built four 262,144-element Elixir lists per call and paid a
  # full host->tensor conversion for each (~130ms warm).
  defp bilinear_interpolate(image, new_height, new_width) do
    {h_in, w_in, _channels} = Nx.shape(image)

    # Place the small index/weight tensors on the SAME backend as the
    # image (they are memoized as backend-agnostic host tensors), so the
    # separable-gather / blend ops below never straddle two backends --
    # `prepare_images` runs on Emily, but the preprocessing parity test
    # runs on the default (Binary) backend.
    backend = image_backend(image)

    %{floor: row_floor, ceil: row_ceil, weight: row_weight} =
      axis_sample_geometry(new_height, h_in, backend)

    %{floor: col_floor, ceil: col_ceil, weight: col_weight} =
      axis_sample_geometry(new_width, w_in, backend)

    # Separable gather: `Nx.take` on axis 0 then axis 1 selects
    # `image[row_idx][:, col_idx]`, matching the old paired-index gather.
    gather = fn row_idx, col_idx ->
      image
      |> Nx.take(row_idx, axis: 0)
      |> Nx.take(col_idx, axis: 1)
    end

    top_left = gather.(row_floor, col_floor)
    top_right = gather.(row_floor, col_ceil)
    bottom_left = gather.(row_ceil, col_floor)
    bottom_right = gather.(row_ceil, col_ceil)

    r_w = Nx.reshape(row_weight, {new_height, 1, 1})
    c_w = Nx.reshape(col_weight, {1, new_width, 1})

    one_minus_r = Nx.subtract(1.0, r_w)
    one_minus_c = Nx.subtract(1.0, c_w)

    top_left
    |> Nx.multiply(Nx.multiply(one_minus_r, one_minus_c))
    |> Nx.add(Nx.multiply(top_right, Nx.multiply(one_minus_r, c_w)))
    |> Nx.add(Nx.multiply(bottom_left, Nx.multiply(r_w, one_minus_c)))
    |> Nx.add(Nx.multiply(bottom_right, Nx.multiply(r_w, c_w)))
  end

  # Floor/ceil gather indices (s64) and interpolation weights (f32) for
  # one resize axis, memoized per {new_size, in_size} pair.
  #
  # The sample positions and weights are computed on Nx.BinaryBackend in
  # f64 -- NOT on the device (Metal has no f64) and NOT lazily as a
  # device graph -- specifically so the arithmetic is bit-identical to
  # the previous plain-Elixir float64 version: `(i+0.5)*in/out - 0.5`,
  # `floor`, clamp, and `pos - clamped_floor` all in float64, so the same
  # floor()/clamp decisions and the same f32-rounded weights come out.
  # Only after the f64 math are the small (length-new_size) index and
  # weight tensors transferred to the device.
  defp axis_sample_geometry(new_size, in_size, backend) do
    key = {__MODULE__, :axis_geometry, new_size, in_size}

    geometry =
      case Process.get(key) do
        nil ->
          computed = compute_axis_sample_geometry(new_size, in_size)
          Process.put(key, computed)
          computed

        cached ->
          cached
      end

    %{
      floor: Nx.backend_transfer(geometry.floor, backend),
      ceil: Nx.backend_transfer(geometry.ceil, backend),
      weight: Nx.backend_transfer(geometry.weight, backend)
    }
  end

  # The memoized geometry is held as backend-agnostic host (Binary)
  # tensors; each use transfers its own small copy onto the caller's
  # backend (`Nx.backend_transfer` on a Binary tensor copies rather than
  # consuming the cached one).
  defp compute_axis_sample_geometry(new_size, in_size) do
    Nx.with_default_backend(Nx.BinaryBackend, fn ->
      positions =
        if new_size == 1 do
          Nx.tensor([0.0], type: :f64)
        else
          # (i + 0.5) * in_size / new_size - 0.5, in f64.
          Nx.iota({new_size}, type: :f64)
          |> Nx.add(0.5)
          |> Nx.multiply(in_size / new_size)
          |> Nx.subtract(0.5)
        end

      raw_floor = Nx.floor(positions)
      floor_idx = raw_floor |> Nx.clip(0, in_size - 1) |> Nx.as_type(:s64)
      ceil_idx = raw_floor |> Nx.add(1.0) |> Nx.clip(0, in_size - 1) |> Nx.as_type(:s64)

      # Weight uses the CLAMPED floor (`pos - clamped_floor`), matching
      # the previous version exactly. Cast to f32 last, so the stored
      # weight is the same f32 value the old code produced.
      clamped_floor_f = Nx.as_type(floor_idx, :f64)
      weight = Nx.subtract(positions, clamped_floor_f) |> Nx.as_type(:f32)

      %{floor: floor_idx, ceil: ceil_idx, weight: weight}
    end)
  end

  # The backend a tensor currently lives on, so freshly-built index /
  # weight tensors can be placed alongside it.
  defp image_backend(%Nx.Tensor{data: %backend_mod{}}), do: backend_mod
end
