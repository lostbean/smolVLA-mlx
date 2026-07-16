# CONTEXT-MAP

The entry point to the design layer. This system is three
[bounded contexts](design/model-runtime/CONTEXT.md#term-infer-action-port) —
two production, one demo scaffold — joined by one runtime seam. This map links
each context's design document and glossary, describes what each owns, and
declares the relationships between them in the DDD vocabulary. Start at the
[root design document](design/design.md) for the foundation and the
whole-system view; start at the [coverage map](COVERAGE.md) for the breadth
answer.

## The contexts

| Context           | Owns                                                                                                                                       | Design                                     | Glossary                                     |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- | --------------------------------------------- |
| **model-runtime**  | SmolVLA's forward pass, its weights, fine-tuning, the two adapters (Python/mlx-vlm, Elixir-native/emily) that expose `infer_action`, and the `InferenceServer` that answers it across a cluster. | [design](design/model-runtime/design.md)   | [terms](design/model-runtime/CONTEXT.md)     |
| **control-loop**   | The bb bot's tick loop: the `ControlLoop` GenServer, the `ActionQueue` it owns, and the ZeroMQ client to the Python fallback adapter.          | [design](design/control-loop/design.md)    | [terms](design/control-loop/CONTEXT.md)      |
| **demo**           | The runnable sim rig: a sim node (a LeRobot/MuJoCo simulated SO-101 arm and the production control loop) clustered to a Mac inference node — a scaffold assembling the two production contexts into a closed loop. | [design](design/demo/design.md)            | [terms](design/demo/CONTEXT.md)              |

There is no root glossary: every term lives in its owning context.

## The relationships

| Relationship        | From → To                     | Kind                    | Direction                                                                                                                                          |
| -------------------- | ------------------------------ | ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| calls infer_action   | control-loop → model-runtime   | **customer / supplier** | control-loop is the sole customer of the `infer_action` port; model-runtime is the sole supplier, through either of its two adapters                  |
| assembles the loop   | demo → control-loop            | **conformist**          | demo hosts and reuses control-loop's `ControlLoop`/`ActionQueue` unchanged, supplying the observation source and actuator sink; control-loop is unaware of demo |
| assembles inference  | demo → model-runtime           | **conformist**          | demo calls model-runtime's `InferenceServer` across the cluster as the `infer_action` port; model-runtime is unaware of demo                          |

:::info The seam
The `infer_action` port is the only thing that crosses between the two
production contexts at runtime. `control-loop` never depends on which adapter
is active; `model-runtime` never depends on anything about the bb bot or the
tick rate beyond answering one call at a time. The `demo` context is a
one-directional customer of both — it assembles them into a runnable rig and
neither production context references it. See the [root's system at a
glance](design/design.md).
:::
