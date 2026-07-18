# Loaded into the sim-node iex session via --dot-iex (see sim-node.sh).
# Runs in the shell's binding context, so `loop` stays bound in the REPL --
# you land ready to call: Demo.SimNode.run_loop(loop, 200, 50)
inference_node = String.to_atom(System.get_env("DEMO_INFERENCE_NODE") || "inference@127.0.0.1")
cookie = String.to_atom(System.get_env("DEMO_COOKIE") || "demo")
sim_address = System.get_env("DEMO_SIM_ADDRESS") || "tcp://127.0.0.1:5556"

{loop, adapter} =
  case Demo.start_sim_node(
         inference_node: inference_node,
         cookie: cookie,
         sim_address: sim_address
       ) do
    {:ok, %{loop: loop, adapter: adapter}} ->
      IO.puts("""

      [sim node ready]
        inference node: #{inspect(inference_node)}
        sim server:     #{sim_address}
        `loop` and `adapter` are bound. Drive the closed loop with:

          Demo.SimNode.run_loop(loop, 200, 50)   # 200 ticks, 50ms apart

      """)

      {loop, adapter}

    {:error, reason} ->
      IO.puts("\n[sim node FAILED to start: #{inspect(reason)}]\n")
      {nil, nil}
  end
