# SmolVLA exposes infer_action(), not generate()

<a id="adr-0001"></a>

mlx-vlm's whole architecture is built around token generation: every model
implements `generate()`, sampling a text token sequence. SmolVLA's action
expert outputs continuous action-chunk tensors via flow matching — there is no
token vocabulary to sample from. We considered encoding actions as a token
string (e.g. quantized floats as special tokens) so SmolVLA would look uniform
with every other mlx-vlm model, but rejected it: quantization loses precision
a continuous control loop needs, and it hides a real architectural difference
(regression head vs. token sampler) behind a fake interface, which the
invariants lens flags as letting illegal calls (e.g. passing sampling
temperature to a regression head) type-check. SmolVLA's `Model` class
implements the standard mlx-vlm `Config`/weight-loading contract for
registration and reuse of vision/language encoding, but exposes its own
`infer_action(image, state, instruction) -> ActionChunk` method; `generate()`
is simply unimplemented for this `model_type`.
