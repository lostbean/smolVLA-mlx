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
  defp bilinear_interpolate(image, new_height, new_width) do
    {h_in, w_in, _channels} = Nx.shape(image)

    row_positions = sample_positions(new_height, h_in)
    col_positions = sample_positions(new_width, w_in)

    row_floor = Enum.map(row_positions, &max(min(floor(&1), h_in - 1), 0))
    row_ceil = Enum.map(row_positions, &max(min(floor(&1) + 1, h_in - 1), 0))
    col_floor = Enum.map(col_positions, &max(min(floor(&1), w_in - 1), 0))
    col_ceil = Enum.map(col_positions, &max(min(floor(&1) + 1, w_in - 1), 0))

    row_weight = Enum.zip_with(row_positions, row_floor, fn p, f -> p - f end)
    col_weight = Enum.zip_with(col_positions, col_floor, fn p, f -> p - f end)

    gather = fn row_indices, col_indices ->
      gather_pixels(image, row_indices, col_indices, new_height, new_width)
    end

    top_left = gather.(row_floor, col_floor)
    top_right = gather.(row_floor, col_ceil)
    bottom_left = gather.(row_ceil, col_floor)
    bottom_right = gather.(row_ceil, col_ceil)

    r_w = row_weight |> Nx.tensor(type: :f32) |> Nx.reshape({new_height, 1, 1})
    c_w = col_weight |> Nx.tensor(type: :f32) |> Nx.reshape({1, new_width, 1})

    one_minus_r = Nx.subtract(1.0, r_w)
    one_minus_c = Nx.subtract(1.0, c_w)

    top_left
    |> Nx.multiply(Nx.multiply(one_minus_r, one_minus_c))
    |> Nx.add(Nx.multiply(top_right, Nx.multiply(one_minus_r, c_w)))
    |> Nx.add(Nx.multiply(bottom_left, Nx.multiply(r_w, one_minus_c)))
    |> Nx.add(Nx.multiply(bottom_right, Nx.multiply(r_w, c_w)))
  end

  defp sample_positions(1, _in_size), do: [0.0]

  defp sample_positions(new_size, in_size) do
    for i <- 0..(new_size - 1) do
      (i + 0.5) * in_size / new_size - 0.5
    end
  end

  defp gather_pixels(image, row_indices, col_indices, new_height, new_width) do
    {_h_in, _w_in, channels} = Nx.shape(image)

    row_grid =
      for r <- row_indices, _c <- col_indices, do: r

    col_grid =
      for _r <- row_indices, c <- col_indices, do: c

    row_idx = Nx.tensor(row_grid, type: :s64)
    col_idx = Nx.tensor(col_grid, type: :s64)

    gathered =
      Nx.take(image, row_idx, axis: 0)
      |> gather_matching_col(col_idx, channels)

    Nx.reshape(gathered, {new_height, new_width, channels})
  end

  # `image[row_idx, col_idx]` fancy-indexing equivalent: after
  # `Nx.take(image, row_idx, axis: 0)` (shape {N, W, C}), select
  # `col_idx[i]` from row i's W axis, matching numpy/mlx advanced
  # indexing (row_idx and col_idx paired elementwise).
  defp gather_matching_col(rows_selected, col_idx, _channels) do
    n = Nx.axis_size(col_idx, 0)

    indices =
      Nx.stack([Nx.iota({n}, type: :s64), col_idx], axis: 1)

    Nx.gather(rows_selected, indices)
  end
end
