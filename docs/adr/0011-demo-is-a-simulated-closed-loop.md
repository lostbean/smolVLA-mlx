# The demo is a simulated closed loop, not an open-loop hardware scaffold

<a id="adr-0011"></a>

The demo context was first designed as an open-loop scaffold: a real camera
fed images to SmolVLA and a virtual bot merely rendered the returned actions,
so nothing the bot "did" changed what the camera next saw — the perception→action
loop never closed, and the demo proved the plumbing ran but never showed the
policy attempting a task. This reverses that: the demo becomes a **closed
perception→action loop** driven by a LeRobot/MuJoCo simulation of the SO-101
arm (the robot the `lerobot/svla_so101_pickplace` checkpoint was trained on),
where `env.step(action)` both applies SmolVLA's action to the simulated arm and
returns the next rendered frame, so the arm actually moves and the loop is
observable end to end. This **supersedes two demo no-goals** —
"Not real robot control" and the virtual bot's rejection of the word
"simulator" — narrowing rather than deleting them: **simulated dynamics**
(MuJoCo physics inside the sim) are now in scope, while **real actuator,
kinematics, and safety logic on physical hardware** stay a hard no-goal, and no
real Raspberry Pi, camera, or NERVES firmware is built (the hardware-integration
slice is dropped). [ADR-0009](0009-offline-action-accuracy-as-task-success-proxy.md#adr-0009)
anticipated exactly this — it deferred a genuine task-success signal "until a
simulation or real-robot environment exists" — so this is that deferred future
arriving, not a new direction.

We considered keeping the open-loop scaffold and adding the simulator as a
later, separate demo, but rejected it: the open-loop demo demonstrates nothing
about whether the policy works, and the two-node BEAM distribution seam
([ADR-0010](0010-beam-distribution-orthogonal-to-infer-action-port.md#adr-0010))
it exists to prove is proven equally well — better — with the sim node in the
non-inference role. The [InferenceServer](../design/model-runtime/design.md)
(model-runtime component 01.5), the production
[ControlLoop](../design/control-loop/design.md) + `ActionQueue`, and ADR-0010's
cross-node `infer_action` topology are all reused unchanged; only the
non-inference node's identity changes from a robot/Pi node to a
[sim node](../design/demo/CONTEXT.md#term-sim-node) hosting the simulation.
