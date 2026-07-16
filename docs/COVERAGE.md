# COVERAGE

Every meaningful part of this system, captured, standard, or out-of-scope.
The `Status` column is the coverage axis — whether the design accounts for the
part — not its build state. This system is partially built: the model-runtime
and control-loop contexts are built (see their pending ledgers for conformance
records), while the demo context and the `InferenceServer` remain designed but
not yet built. Per-part build state lives in each context's pending ledger, not
here.

| Part                                                          | Status       | Why / pointer                                                                |
| -------------------------------------------------------------- | ------------- | ------------------------------------------------------------------------------ |
| SmolVLA config + model class (Python/mlx-vlm fork)              | captured      | [model-runtime](design/model-runtime/design.md), component 01.1               |
| SmolVLA fine-tuning, Python/LeRobot (action expert)              | captured      | [model-runtime](design/model-runtime/design.md), component 01.3               |
| Elixir-native SmolVLA forward pass (emily/Nx.Defn)              | captured      | [model-runtime](design/model-runtime/design.md), component 01.2, mechanism prototype-verified |
| Elixir-native fine-tuning (Nx/Axon)                             | captured      | [model-runtime](design/model-runtime/design.md), component 01.4, gated on task-performance parity |
| ControlLoop GenServer + ActionQueue                             | captured      | [control-loop](design/control-loop/design.md), components 01.1–01.2           |
| ZeroMQ client + Python-side fallback server                     | captured      | [control-loop](design/control-loop/design.md), component 01.3                 |
| InferenceServer (cluster-addressable emily-native adapter)      | captured      | [model-runtime](design/model-runtime/design.md), component 01.5               |
| Demo sim env adapter (gym env seam: obs out, action in)         | captured      | [demo](design/demo/design.md), component 01.1                                 |
| Demo sim node (two-node cluster wiring)                         | captured      | [demo](design/demo/design.md), component 01.2                                 |
| Demo sim server (Python LeRobot/MuJoCo gym env over ZeroMQ)     | captured      | [demo](design/demo/design.md), component 01.1 (the sim env adapter drives it); [ADR-0012](adr/0012-sim-env-bridged-via-python-sim-server-over-zeromq.md#adr-0012) |
| bb bot actuator/kinematics/safety logic on real hardware        | out-of-scope  | explicit no-goal — the demo simulates arm dynamics in MuJoCo but drives no real actuator; real kinematics/safety live in a different system entirely, not designed here |
| Design-layer check scripts (`scripts/design-render`, `scripts/layer-integrity`, `scripts/gate-stamp-check`) | standard | copy-installed from the framework, unmodified; standard tooling, not bespoke |
| `schema/design-schema.json`                                     | standard      | the framework's shared kernel, copy-installed unmodified                     |
