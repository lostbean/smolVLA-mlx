# CONTEXT-MAP

The entry point to the design layer. This system is two
[bounded contexts](design/model-runtime/CONTEXT.md#term-infer-action-port)
joined by one seam. This map links each context's design document and
glossary, describes what each owns, and declares the relationship between
them in the DDD vocabulary. Start at the [root design
document](design/design.md) for the foundation and the whole-system view;
start at the [coverage map](COVERAGE.md) for the breadth answer.

## The contexts

| Context           | Owns                                                                                                                                       | Design                                     | Glossary                                     |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- | --------------------------------------------- |
| **model-runtime**  | SmolVLA's forward pass, its weights, fine-tuning, and the two adapters (Python/mlx-vlm, Elixir-native/emily) that expose `infer_action`.       | [design](design/model-runtime/design.md)   | [terms](design/model-runtime/CONTEXT.md)     |
| **control-loop**   | The bb bot's tick loop: the `ControlLoop` GenServer, the `ActionQueue` it owns, and the ZeroMQ client to the Python fallback adapter.          | [design](design/control-loop/design.md)    | [terms](design/control-loop/CONTEXT.md)      |

There is no root glossary: every term lives in its owning context.

## The relationship

| Relationship        | From → To                     | Kind                    | Direction                                                                                                                                          |
| -------------------- | ------------------------------ | ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| calls infer_action   | control-loop → model-runtime   | **customer / supplier** | control-loop is the sole customer of the `infer_action` port; model-runtime is the sole supplier, through either of its two adapters                  |

:::info The seam
The `infer_action` port is the only thing that crosses between the two
contexts at runtime. `control-loop` never depends on which adapter is active;
`model-runtime` never depends on anything about the bb bot or the tick rate
beyond answering one call at a time. See the [root's system at a
glance](design/design.md).
:::
