# Elixir owns the action queue and tick timing, not Python

<a id="adr-0002"></a>

SmolVLA's own upstream (LeRobot) ships an async client-server split where the
Python side holds the action queue and decides when to request a new chunk. We
considered mirroring that shape — porting LeRobot's queueing policy into the
Python side of this system — for consistency with upstream, but rejected it:
the queue is exactly the kind of stateful, timed, supervised process the BEAM
is built for, and this system's Elixir side already owns the bb bot's control
tick. Splitting the queue's ownership across languages (Python decides when to
predict, Elixir decides when to act) would braid one concern across a process
boundary for no benefit. `infer_action()` therefore stays a plain synchronous
call — one observation in, one action chunk out, no opinion about timing — and
the calling Elixir `ControlLoop` process owns the queue, the low-water
threshold, and the request timing entirely on its own side.
