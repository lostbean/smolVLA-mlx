# SmolVLA's config/weight loading adapts LeRobot's checkpoint shape, never patches mlx-vlm's shared dispatch

<a id="adr-0006"></a>

The design assumed a real SmolVLA checkpoint's `config.json` would carry
mlx-vlm's expected `model_type` key and its nested `vision_config`/
`text_config` shape, so `get_model_and_args()`'s generic dynamic-import
dispatch would reach `SmolVLAModel` unmodified. The real published checkpoint
(`lerobot/smolvla_base`) does not: it is LeRobot's own flat policy-config
format — a `"type"` key instead of `model_type`, no nested vision/text
sub-configs, and a single flat `model.safetensors` with LeRobot-native
tensor-name prefixes (`vlm_with_expert.*`) rather than mlx-vlm's usual
HF-`transformers`-shaped layout. We considered extending mlx-vlm's shared
`get_model_and_args()`/`load_model()` in `utils.py` to also accept
`config.get("type")` as a fallback discriminator, making SmolVLA reachable
through the exact same generic path as every other model. We rejected this:
it touches plumbing every other model's dispatch relies on, for a shape
(LeRobot's own policy-config convention) that is specific to this one model
family, not a second real mlx-vlm convention. Instead, `SmolVLAConfig`'s
loading path and `SmolVLAModel.from_pretrained()` read the checkpoint's real
LeRobot-native `config.json` and `model.safetensors` directly, translating
into the shapes `SmolVLAConfig`/`SmolVLAModel` need internally, without
routing through `get_model_and_args()`'s generic dispatch or requiring any
change to shared mlx-vlm code. The `smolvla/` directory name still matches
`model_type` in spirit — a checkpoint's real `type` field names the same
model — but the automatic directory-name dispatch is not literally exercised
against an unmodified real checkpoint; `from_pretrained()` performs its own
type-keyed loading instead. This keeps every other model's dispatch path
untouched and keeps SmolVLA's own loading code the only place that
understands LeRobot's checkpoint convention, preserving the
upstream-mergeable shape mlx-vlm's own contributors would recognize.
