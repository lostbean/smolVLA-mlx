Status: ready-for-agent
Category: enhancement

## Parent

[control-loop design](../../../docs/design/control-loop/design.md), components 01.1 (`ControlLoop`), 01.2 (`ActionQueue`), 01.3 (`ZeroMQ client`). Pending build entry, [ADR-0002](../../../docs/adr/0002-elixir-owns-the-control-loop.md#adr-0002).

## What to build

Build the full Elixir-side control loop: `ControlLoop`, a supervised process
ticking at the target rate, popping the next queued action from
`{#term-action-queue}` and sending it onward each tick; `ActionQueue` itself,
holding the ordered sequence of not-yet-executed actions and merging newly
returned action chunks with whatever is still queued (never replacing); and
the ZeroMQ client half of the fallback adapter, calling the previous slice's
server and reconnecting on a dropped link.

`ControlLoop` owns the `{#term-low-water-threshold}` policy entirely on the
Elixir side — no policy is inherited from Python (the **"Elixir owns the
queue and the timing, not Python"** goal): when the queue's depth drops below
threshold, `ControlLoop` calls `infer_action` through the ZeroMQ client
without blocking the tick loop, and the returned chunk is aggregated into
the queue when it arrives. Model the full state machine named in the
design (`queue_healthy` / `queue_low`, threshold-crossing as the only
transition edge) — not just the two state names.

This is the first slice that demos the **entire cross-language system**
end to end: a real Elixir process, ticking at a target rate, producing real
actions from a real SmolVLA inference call over the network.

## Acceptance criteria

- [ ] `ControlLoop` ticks at a configurable target rate, popping and
      "sending" (to a stub actuator sink — real bb bot actuator wiring is
      out of scope) one action per tick.
- [ ] When `ActionQueue`'s depth drops below the low-water threshold,
      `ControlLoop` calls `infer_action` via the ZeroMQ client without
      blocking subsequent ticks; the returned chunk is merged into the
      queue by aggregation, not replacement.
- [ ] The queue is never read past its safe depth, and no action is ever
      executed twice (both foundation invariants hold under a sustained
      run, not just a single tick).
- [ ] A dropped ZeroMQ connection reconnects rather than crashing
      `ControlLoop`; a timed-out or errored `infer_action` call leaves the
      queue draining on what it already has rather than blocking the tick
      loop.
- [ ] An end-to-end run (real Elixir process, real ZeroMQ call to the real
      Python server from slice 03, real SmolVLA inference) produces a
      sustained stream of real action chunks at the target rate.

## Out of scope

The emily-native adapter (next slice); real bb bot actuator/kinematics/safety
logic (a standing no-goal); fine-tuning.

## Blocked by

03-zeromq-server
