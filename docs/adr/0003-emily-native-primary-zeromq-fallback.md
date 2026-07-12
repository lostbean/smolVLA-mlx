# emily-native is the primary inference adapter; ZeroMQ-Python is a permanent fallback

<a id="adr-0003"></a>

Two adapters can implement the `infer_action` port: a Python process running
the mlx-vlm fork, reached over the network, or an Elixir-native port of
SmolVLA's forward pass running in-process via `emily` (the MLX↔Nx binding)
and `Nx.Defn`. We considered treating the Python adapter as the only
production path (simpler, known-feasible today) with emily-native as a
speculative future optimization, but the owner's explicit preference is the
opposite: emily-native is the intended production path, because it removes a
process/network hop from the hot 5–30Hz loop entirely. The trade-off this
accepts: `emily` has no story for importing an arbitrary MLX model graph — the
forward pass (SmolVLM2 backbone plus the flow-matching action expert) must be
reimplemented by hand as `Nx.Defn` code, a second, independently maintained
port of the architecture sharing only weights (safetensors) with the Python
reference, never code. This bet is not taken blind: it is gated on a
`/prototype` answering whether the forward pass is expressible correctly and
fast enough against `emily`'s backend
([pending build entry](../design/control-loop/design.md)). The ZeroMQ-Python
adapter is designed as a permanent, first-class part of the system regardless
of the prototype's outcome — not a throwaway shim — because the physical
topology (the Mac as a separate networked inference server from the bb bot's
Elixir cluster) makes it a real resilience boundary and a legitimate
same-host-or-networked deployment option on its own merits.
