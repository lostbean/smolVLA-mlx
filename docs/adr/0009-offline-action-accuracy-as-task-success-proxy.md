# The cutover gate's "task success" is an offline action-accuracy proxy plus throughput, not a live rollout

<a id="adr-0009"></a>

ADR-0005's cutover gate names "task success rate on held-out evaluation
episodes" as the parity bar, but this system has no simulator and no
connected physical robot — [control-loop](../design/control-loop/design.md)'s
own foundation excludes actuator/kinematics/safety logic as a standing
no-goal, so no code path in this repo can execute a predicted action and
observe whether a task actually completed. A literal live-rollout success
rate is not achievable with what exists. We considered deferring the gate
entirely until a simulation or real-robot environment exists, since a proxy
metric risks being mistaken for the real thing — but rejected deferral: both
trainers already exist and produce real, independently loadable weights, and
withholding a comparison indefinitely leaves ADR-0005's one-way cutover
permanently undecidable. Instead, the gate uses two offline, fully
computable dimensions: **action-prediction accuracy** — each trainer's
fine-tuned policy's predicted action chunks compared against the held-out
episodes' real recorded (ground-truth) actions, a standard imitation-learning
proxy for policy quality when a live rollout is unavailable — and
**throughput** (actions or images processed per second), a pure compute-
performance axis orthogonal to accuracy. Neither substitutes for a genuine
task-completion measurement; the gate's report states this explicitly rather
than presenting the proxy as equivalent to real task success. Should a
simulation or real-robot environment be added to this system later, revisit
whether the gate should incorporate live rollout results instead of, or
alongside, this proxy.
