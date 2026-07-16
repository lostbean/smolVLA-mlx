# Warm-latency + parity benchmark for SmolVLA.infer_action/4.
#
# Loads the real checkpoint once, uses the same e2e fixture inputs the
# parity test uses, warms up, then reports the median of N warm runs plus
# the fixed-noise parity MRE / max-abs-diff.
#
# Run under MIX_ENV=test (that's where deps are built):
#   MIX_ENV=test mix run bench/warm_latency.exs
#
# Env knobs:
#   BENCH_RUNS=N        number of timed runs (default 7)
#   BENCH_FUSE=1        enable Emily.Compiler fuse: true (off by default;
#                       it helped the original large per-step graph but is
#                       neutral-to-slightly-slower once the KV-cache split
#                       and vision compile shrank the per-step work)

runs = String.to_integer(System.get_env("BENCH_RUNS", "7"))
fuse = System.get_env("BENCH_FUSE") == "1"

Nx.global_default_backend({Emily.Backend, device: :gpu})

if fuse do
  Nx.Defn.global_default_options(compiler: Emily.Compiler, fuse: true)
  IO.puts("[bench] Emily.Compiler fuse: true ENABLED")
else
  Nx.Defn.global_default_options(compiler: Emily.Compiler)
end

checkpoint_dir =
  Path.expand(
    "~/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/c83c3163b8ca9b7e67c509fffd9121e66cb96205"
  )

fixtures_dir = Path.join([File.cwd!(), "test", "fixtures"])

model = SmolVLA.load(checkpoint_dir)

image =
  File.read!(Path.join(fixtures_dir, "e2e_probe_image_f32.bin"))
  |> Nx.from_binary(:f32)
  |> Nx.reshape({224, 224, 3})

state =
  File.read!(Path.join(fixtures_dir, "e2e_probe_state_f32.bin"))
  |> Nx.from_binary(:f32)
  |> Nx.to_flat_list()

instruction = File.read!(Path.join(fixtures_dir, "e2e_probe_instruction.txt"))

noise =
  File.read!(Path.join(fixtures_dir, "e2e_probe_noise_f32.bin"))
  |> Nx.from_binary(:f32)
  |> Nx.reshape({1, 50, 32})

expected =
  File.read!(Path.join(fixtures_dir, "e2e_probe_action_chunk_f32.bin"))
  |> Nx.from_binary(:f32)
  |> Nx.reshape({50, 32})

# ---- parity (fixed noise) ----
actual =
  SmolVLA.infer_action(model, image, state, instruction, noise)
  |> Nx.as_type(:f32)
  |> Nx.backend_transfer(Nx.BinaryBackend)

abs_diff = Nx.abs(Nx.subtract(actual, expected))
max_abs_diff = Nx.to_number(Nx.reduce_max(abs_diff))
mre = Nx.to_number(Nx.mean(abs_diff)) / Nx.to_number(Nx.mean(Nx.abs(expected)))

IO.puts("\n[bench] parity: MRE=#{Float.round(mre * 100, 4)}%  max_abs_diff=#{max_abs_diff}")

# ---- warm latency (median of N) ----
# One warm-up (JIT trace) before timing.
_ = SmolVLA.infer_action(model, image, state, instruction)

times_ms =
  for _ <- 1..runs do
    {us, _} = :timer.tc(fn -> SmolVLA.infer_action(model, image, state, instruction) end)
    us / 1000
  end

sorted = Enum.sort(times_ms)
median = Enum.at(sorted, div(length(sorted), 2))
min_ms = Enum.min(sorted)
max_ms = Enum.max(sorted)

IO.puts(
  "[bench] warm infer_action/4 over #{runs} runs: median=#{Float.round(median, 1)}ms " <>
    "min=#{Float.round(min_ms, 1)}ms max=#{Float.round(max_ms, 1)}ms"
)

IO.puts("[bench] all runs (ms): #{Enum.map_join(sorted, ", ", &Float.round(&1, 1))}")
IO.puts("[bench] target: <= 327ms")
