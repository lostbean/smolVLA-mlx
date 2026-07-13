defmodule SmolVLA.Dataset do
  @moduledoc """
  Reads real LeRobotDataset v3.0-format episodes directly off disk -- the
  Elixir-native `FineTuneJob`'s own independent episode-loading (component
  01.4's "Interacts with": "the same LeRobotDataset-format episodes as 01.3
  ... real or simulated, same non-distinction"), sharing no code with the
  Python trainer (ADR-0004) or with LeRobot's own `LeRobotDataset` Python
  class.

  **Real on-disk format** (confirmed directly on 2026-07-13 against a real,
  locally-cached `lerobot/svla_so101_pickplace` snapshot -- not guessed):

      <root>/
        meta/info.json                              -- dataset-level metadata
        meta/episodes/chunk-000/file-000.parquet     -- per-episode index
        meta/tasks.parquet                           -- task_index -> text
        data/chunk-000/file-000.parquet              -- per-FRAME action/state/index columns
        videos/<camera_key>/chunk-000/file-000.mp4    -- per-camera video, one file per chunk

  `meta/info.json`'s `features` map declares each camera's `dtype: "video"`
  feature (vs. `action`/`observation.state`'s plain float-list features);
  `meta/episodes/*.parquet` maps each `episode_index` to its data-file
  location (`data/chunk_index`, `data/file_index`), its row range within
  that file (`dataset_from_index`/`dataset_to_index`), each camera's own
  video-file location and `from_timestamp`/`to_timestamp` (episodes share
  video files -- the per-episode segment is a timestamp RANGE, not a
  separate file), and the episode's task string(s).

  **Parquet**: read via `Explorer` (`elixir-nx`'s own DataFrame library, a
  Rust NIF over Polars -- no Python, no code shared with LeRobot's own
  loader) -- a real, general-purpose format reader, the same category of
  dependency as `Safetensors`/`Tokenizers` already used by this repo's
  Elixir-native inference path.

  **Video**: no viable pure-Elixir or NIF-based AV1 decoder was found for
  this real dataset's actual codec (`meta/info.json` declares
  `video.codec: "av1"`) -- `evision` (OpenCV bindings) was spiked directly
  and its precompiled build cannot open this dataset's real AV1 files
  ("Couldn't read movie file"); building OpenCV from source with `dav1d`
  linked in is unreasonable from-scratch build work for this chunk. This
  module instead shells out to the system's real `ffmpeg` binary (already
  present in this repo's own nix dev shell, real AV1 decode support
  confirmed directly via `ffprobe`/`ffmpeg -vf select`) for frame
  extraction ONLY -- a thin, justified exception to "no code crosses the
  boundary" (ADR-0004's own text: "the only artifact that can legitimately
  cross the boundary is the trained weights"): `ffmpeg` is a general media
  tool, not LeRobot or Python code, and this module invokes no
  training/model logic through it -- it decodes one well-known on-disk
  video format into raw pixels, structurally the same kind of "read a
  standard artifact independently on each side" as `Safetensors`/
  `Tokenizers` already are. Flagged explicitly in this chunk's own report
  rather than decided silently, per the work order's own escalation rule.
  """

  alias Explorer.DataFrame

  require Explorer.DataFrame

  defmodule Frame do
    @moduledoc "One real (image, state, action) sample read from a real episode."
    @enforce_keys [:image, :state, :action, :task, :episode_index, :frame_index]
    defstruct [:image, :state, :action, :task, :episode_index, :frame_index]

    @type t :: %__MODULE__{
            image: Nx.Tensor.t(),
            state: [float()],
            action: [float()],
            task: String.t(),
            episode_index: non_neg_integer(),
            frame_index: non_neg_integer()
          }
  end

  defmodule NoFfmpegError do
    @moduledoc """
    Raised when no `ffmpeg` binary is found on `PATH` -- loud and local,
    never a silent skip of video decoding (matches this repo's own
    "loud/local, never a silent fallback" convention, e.g.
    `SmolVLA.load/2`'s missing-checkpoint handling).
    """
    defexception message:
                   "ffmpeg not found on PATH -- required to decode LeRobotDataset video frames"
  end

  @enforce_keys [:root, :info, :episodes, :tasks]
  defstruct [:root, :info, :episodes, :tasks]

  @type t :: %__MODULE__{
          root: Path.t(),
          info: map(),
          episodes: DataFrame.t(),
          tasks: %{non_neg_integer() => String.t()}
        }

  @doc """
  Opens a real LeRobotDataset v3.0 directory at `root` (the directory
  directly containing `meta/`, `data/`, `videos/` -- e.g. an already
  -downloaded Hugging Face Hub snapshot directory), reading its metadata
  eagerly (small: `info.json`, the episode index, the task table) but not
  its per-episode frame/video data (read lazily by `frames/2`).

  Raises loud and local (via `File.read!`/`Explorer.DataFrame.from_parquet!`'s
  own errors) on a missing or malformed dataset directory -- never a
  silent empty-dataset fallback, matching this repo's established
  "Fails" convention.
  """
  @spec open(Path.t()) :: t()
  def open(root) do
    info_path = Path.join([root, "meta", "info.json"])

    unless File.exists?(info_path) do
      raise File.Error,
        reason: :enoent,
        action: "read (LeRobotDataset meta/info.json)",
        path: IO.chardata_to_string(info_path)
    end

    info = info_path |> File.read!() |> Jason.decode!()

    unless info["codebase_version"] == "v3.0" do
      raise ArgumentError,
            "SmolVLA.Dataset.open/1 only reads LeRobotDataset v3.0 -- got " <>
              "codebase_version=#{inspect(info["codebase_version"])} at #{info_path}"
    end

    episodes = read_all_parquet_glob(Path.join([root, "meta", "episodes", "**", "*.parquet"]))
    tasks_df = DataFrame.from_parquet!(Path.join([root, "meta", "tasks.parquet"]))

    tasks =
      tasks_df
      |> DataFrame.to_rows()
      |> Map.new(fn row -> {row["task_index"], row["__index_level_0__"]} end)

    %__MODULE__{root: root, info: info, episodes: episodes, tasks: tasks}
  end

  defp read_all_parquet_glob(glob) do
    glob
    |> Path.wildcard()
    |> Enum.sort()
    |> Enum.map(&DataFrame.from_parquet!/1)
    |> Enum.reduce(fn df, acc -> DataFrame.concat_rows(acc, df) end)
  end

  @doc "Total episode count, per `meta/info.json`'s own `total_episodes`."
  @spec total_episodes(t()) :: non_neg_integer()
  def total_episodes(%__MODULE__{info: info}), do: info["total_episodes"]

  @doc """
  Reads every real frame of `episode_index` -- action, state, task string,
  and every declared camera's real decoded image (via `ffmpeg`, see this
  module's own moduledoc) -- as a list of `Frame.t()` in `frame_index`
  order.

  Raises `NoFfmpegError` if no `ffmpeg` binary is on `PATH`; raises (via
  `Explorer`/`File.read!`) loud and local on a missing/malformed episode.
  """
  @spec frames(t(), non_neg_integer()) :: [Frame.t()]
  def frames(%__MODULE__{} = dataset, episode_index) do
    ffmpeg = System.find_executable("ffmpeg") || raise NoFfmpegError

    episode_row = episode_metadata!(dataset, episode_index)
    data_df = data_frame_for_episode(dataset, episode_row)

    # Single-camera contract match with `SmolVLA.infer_action/4` (one
    # image per call) -- decodes only the FIRST declared camera
    # (deterministic sorted key order); multi-camera fusion is out of this
    # chunk's scope. Deciding this once per episode (not per frame) matters
    # for real wall-clock time: decoding every declared camera on every
    # frame when only one is ever used would double real `ffmpeg`
    # subprocess spawns for no benefit (confirmed directly: this dataset
    # declares 2 cameras).
    camera_key = dataset |> camera_keys() |> hd()

    %{video_path: video_path, from_timestamp: from_ts} =
      open_video_capture(dataset, episode_row, camera_key)

    # Probed ONCE per video file, not once per frame -- an earlier version
    # of this function called `ffprobe` inside the per-frame decode path,
    # which (combined with one `ffmpeg` subprocess spawn per frame) made a
    # real 303-frame episode take ~46 real seconds; probing the file's
    # fixed width/height once up front and reusing it removes half the
    # real subprocess spawns per frame.
    dimensions = probe_dimensions!(ffmpeg, video_path)

    # The episode-metadata parquet's own "tasks" column already carries the
    # real task text directly (confirmed against the real dataset), so
    # `dataset.tasks` (task_index -> text, from meta/tasks.parquet) is only
    # needed as a fallback for a LeRobotDataset export that omits the
    # per-episode "tasks" text column and relies solely on
    # `data/*.parquet`'s own `task_index` column instead.
    task =
      case episode_row["tasks"] do
        [first | _] -> first
        _ -> dataset.tasks[List.first(DataFrame.to_rows(data_df))["task_index"]]
      end

    data_df
    |> DataFrame.to_rows()
    |> Enum.sort_by(& &1["frame_index"])
    |> Enum.map(fn row ->
      image = decode_frame!(ffmpeg, video_path, from_ts + row["timestamp"], dimensions)

      %Frame{
        image: image,
        state: row["observation.state"],
        action: row["action"],
        task: task,
        episode_index: episode_index,
        frame_index: row["frame_index"]
      }
    end)
  end

  defp episode_metadata!(%__MODULE__{episodes: episodes}, episode_index) do
    row =
      episodes
      |> DataFrame.filter(episode_index == ^episode_index)
      |> DataFrame.to_rows()
      |> List.first()

    unless row do
      raise ArgumentError, "no episode #{episode_index} found in this dataset's episode index"
    end

    row
  end

  defp data_frame_for_episode(%__MODULE__{root: root, info: info}, episode_row) do
    chunk_index = episode_row["data/chunk_index"]
    file_index = episode_row["data/file_index"]
    path = data_path(root, info, chunk_index, file_index)

    df = DataFrame.from_parquet!(path)
    episode_index = episode_row["episode_index"]
    DataFrame.filter(df, episode_index == ^episode_index)
  end

  defp data_path(root, info, chunk_index, file_index) do
    template = info["data_path"] || "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    Path.join(root, expand_path_template(template, chunk_index, file_index))
  end

  defp video_path(root, info, video_key, chunk_index, file_index) do
    template =
      info["video_path"] || "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"

    expanded =
      template
      |> String.replace("{video_key}", video_key)
      |> expand_path_template(chunk_index, file_index)

    Path.join(root, expanded)
  end

  # Expands LeRobot's own `{chunk_index:03d}`/`{file_index:03d}`
  # zero-padded-width path template placeholders -- every real
  # `meta/info.json` this format ships with declares a 3-digit width (per
  # `hf_hub_v3`'s own dataset-writer convention, confirmed directly
  # against the real `lerobot/svla_so101_pickplace` snapshot's own
  # `info.json`), so this reads the declared width straight out of the
  # template string itself (`0(\d)d`) rather than hardcoding `3`, without
  # needing a general `printf`-style formatter for a two-placeholder,
  # always-zero-padded-integer template.
  defp expand_path_template(template, chunk_index, file_index) do
    template
    |> replace_indexed_placeholder("chunk_index", chunk_index)
    |> replace_indexed_placeholder("file_index", file_index)
  end

  defp replace_indexed_placeholder(template, name, value) do
    Regex.replace(~r/\{#{name}:0(\d)d\}/, template, fn _whole, width_str ->
      value |> Integer.to_string() |> String.pad_leading(String.to_integer(width_str), "0")
    end)
  end

  defp camera_keys(%__MODULE__{info: info}) do
    info["features"]
    |> Enum.filter(fn {_k, v} -> is_map(v) and v["dtype"] == "video" end)
    |> Enum.map(fn {k, _v} -> k end)
    |> Enum.sort()
  end

  defp open_video_capture(%__MODULE__{root: root, info: info}, episode_row, camera_key) do
    chunk_index = episode_row["videos/#{camera_key}/chunk_index"]
    file_index = episode_row["videos/#{camera_key}/file_index"]
    from_timestamp = episode_row["videos/#{camera_key}/from_timestamp"]

    %{
      video_path: video_path(root, info, camera_key, chunk_index, file_index),
      from_timestamp: from_timestamp
    }
  end

  # Decodes exactly one frame at `timestamp_seconds` (an absolute position
  # within the shared per-chunk video file, already offset by the
  # episode's own `from_timestamp`) to a real `{height, width, 3}` u8 Nx
  # tensor, via a single `ffmpeg` subprocess writing raw RGB24 pixels to
  # stdout -- confirmed directly to produce exactly `height * width * 3`
  # bytes for this real dataset's real 640x480 videos, matching
  # `SmolVLA.infer_action/4`'s own expected `(H, W, 3)` image shape.
  # `dimensions` is probed ONCE per video file by the caller (`frames/2`),
  # not re-probed here per frame -- see `frames/2`'s own comment on why.
  defp decode_frame!(ffmpeg, video_path, timestamp_seconds, {width, height}) do
    unless File.exists?(video_path) do
      raise File.Error,
        reason: :enoent,
        action: "read (LeRobotDataset video)",
        path: IO.chardata_to_string(video_path)
    end

    args = [
      "-y",
      "-v",
      "error",
      "-ss",
      Float.to_string(timestamp_seconds),
      "-i",
      video_path,
      "-vframes",
      "1",
      "-f",
      "rawvideo",
      "-pix_fmt",
      "rgb24",
      "-"
    ]

    case System.cmd(ffmpeg, args, stderr_to_stdout: false) do
      {raw, 0} ->
        expected_bytes = width * height * 3

        if byte_size(raw) != expected_bytes do
          raise "ffmpeg produced #{byte_size(raw)} bytes decoding a frame from #{video_path} " <>
                  "at t=#{timestamp_seconds}s, expected #{expected_bytes} (#{width}x#{height}x3) " <>
                  "-- corrupt or truncated video, refusing to silently reshape"
        end

        raw
        |> Nx.from_binary(:u8)
        |> Nx.reshape({height, width, 3})

      {output, code} ->
        raise "ffmpeg exited #{code} decoding a frame from #{video_path} at t=#{timestamp_seconds}s: #{output}"
    end
  end

  defp probe_dimensions!(ffmpeg, video_path) do
    ffprobe = ffmpeg |> Path.dirname() |> Path.join("ffprobe")

    ffprobe = if File.exists?(ffprobe), do: ffprobe, else: System.find_executable("ffprobe")

    unless ffprobe do
      raise NoFfmpegError, message: "ffprobe not found on PATH (sibling of #{ffmpeg})"
    end

    args = [
      "-v",
      "error",
      "-select_streams",
      "v:0",
      "-show_entries",
      "stream=width,height",
      "-of",
      "csv=s=x:p=0",
      video_path
    ]

    case System.cmd(ffprobe, args) do
      {output, 0} ->
        [w, h] = output |> String.trim() |> String.split("x") |> Enum.map(&String.to_integer/1)
        {w, h}

      {output, code} ->
        raise "ffprobe exited #{code} probing #{video_path}: #{output}"
    end
  end
end
