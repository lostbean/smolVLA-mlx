# COVERAGE

Every meaningful part of this system, captured, standard, or out-of-scope.
This is a greenfield design: every `captured` row below is designed but not
yet built (see each context's pending ledger) — coverage is scoped ahead of
construction, not discovered after it.

| Part                                                          | Status       | Why / pointer                                                                |
| -------------------------------------------------------------- | ------------- | ------------------------------------------------------------------------------ |
| SmolVLA config + model class (Python/mlx-vlm fork)              | captured      | [model-runtime](design/model-runtime/design.md), component 01.1               |
| SmolVLA fine-tuning (action expert)                             | captured      | [model-runtime](design/model-runtime/design.md), component 01.3               |
| Elixir-native SmolVLA forward pass (emily/Nx.Defn)              | captured      | [model-runtime](design/model-runtime/design.md), component 01.2, gated on `/prototype` |
| ControlLoop GenServer + ActionQueue                             | captured      | [control-loop](design/control-loop/design.md), components 01.1–01.2           |
| ZeroMQ client + Python-side fallback server                     | captured      | [control-loop](design/control-loop/design.md), component 01.3                 |
| bb bot actuator/kinematics/safety logic                         | out-of-scope  | explicit no-goal — lives in a different system entirely, not designed here    |
| Design-layer check scripts (`scripts/design-render`, `scripts/layer-integrity`, `scripts/gate-stamp-check`) | standard | copy-installed from the framework, unmodified; standard tooling, not bespoke |
| `schema/design-schema.json`                                     | standard      | the framework's shared kernel, copy-installed unmodified                     |
