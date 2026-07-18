# Loaded into the inference-node iex session via --dot-iex (see inference-node.sh).
# Runs in the shell's binding context, so anything bound here stays available
# in the REPL. Auto-loads the checkpoint the launcher resolved into
# DEMO_CHECKPOINT.
checkpoint = System.get_env("DEMO_CHECKPOINT")

case checkpoint && Demo.start_inference_node(checkpoint) do
  {:ok, _srv} ->
    IO.puts("\n[inference node ready -- InferenceServer loaded from #{checkpoint}]\n")

  {:error, reason} ->
    IO.puts("\n[inference node FAILED to load: #{inspect(reason)}]\n")

  nil ->
    IO.puts("\n[no DEMO_CHECKPOINT set -- run Demo.start_inference_node(path) manually]\n")
end
