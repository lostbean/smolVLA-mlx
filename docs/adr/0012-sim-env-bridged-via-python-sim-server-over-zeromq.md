# The simulation is bridged to the BEAM sim node via a Python sim server over ZeroMQ

<a id="adr-0012"></a>

The LeRobot/MuJoCo simulation is Python (gymnasium + MuJoCo); the
[sim node](../design/demo/CONTEXT.md#term-sim-node) that hosts it and the
production [ControlLoop](../design/control-loop/design.md) is Elixir/BEAM. The
closed loop needs `env.reset`, `env.step(action)`, and `env.render` called from
Elixir every tick. We drive them by mirroring the pattern the repo already
uses for its Python inference fallback: a **long-lived Python sim server**
wrapping the gym env, driven from the Elixir sim node over a `chumak` ZeroMQ
REQ/REP socket with MessagePack framing — the same transport and wire machinery
as [model_runtime_server](../design/control-loop/design.md)
([ADR-0007](0007-msgpack-wire-format-for-zeromq-fallback.md#adr-0007),
[ADR-0008](0008-chumak-pure-erlang-zeromq-client.md#adr-0008)). The sim server
is the one seam between Elixir and MuJoCo: the
[observation](../design/model-runtime/CONTEXT.md#term-observation) (image +
state) comes back over the wire, the [action chunk](../design/model-runtime/CONTEXT.md#term-action-chunk)'s
actions go out.

We considered an Elixir Port over stdio and an embedded-CPython approach
(Pythonx), but rejected both: the Port introduces a second cross-language
mechanism the repo does not already have, and running MuJoCo + gymnasium inside
the BEAM is heavy, GIL-bound, and unproven here. Reusing the established
ZeroMQ/MessagePack pattern keeps the repo to **one cross-language transport
idiom**. This does **not** contradict
[ADR-0010](0010-beam-distribution-orthogonal-to-infer-action-port.md#adr-0010)'s
"no foreign serialization hop" advantage for the emily-native inference path:
that advantage is about the `infer_action` port, which still crosses the BEAM
cluster natively; the sim bridge is a demo-side seam to a genuinely external
Python physics engine, exactly the kind of true external boundary a foreign
wire format is appropriate for.
