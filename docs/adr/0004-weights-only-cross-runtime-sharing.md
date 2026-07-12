# Only weights cross the Python/Elixir boundary, never code

<a id="adr-0004"></a>

The Python mlx-vlm fork and the Elixir-native `Nx.Defn` port are two
independent implementations of the same forward pass. We considered
generating one from the other (e.g. tracing the Python model and exporting a
graph the Elixir side could execute), but MLX has no ONNX/TorchScript-style
serialized-graph story — a model is code building a graph via `nn.Module`,
not an exportable artifact a foreign runtime can load and run. Given that
constraint, the only artifact that can legitimately cross the boundary is the
trained weights (safetensors); the forward-pass code itself is authored twice,
once per runtime, and kept behaviorally equivalent by convention (a
conformance check comparing outputs on fixed inputs), not by a shared build
artifact. This is surprising enough to a newcomer expecting one canonical
implementation that it is recorded here rather than left implicit.
