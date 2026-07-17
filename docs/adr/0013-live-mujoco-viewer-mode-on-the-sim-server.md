# The live viewer is an optional MuJoCo-window mode on the sim server, not a separate component

<a id="adr-0013"></a>

The closed-loop demo runs headless — the [sim server](../design/demo/CONTEXT.md#term-sim-server)
renders `rgb_array` frames and the loop's progress is legible only through
saved frames, state deltas, and tests. To let someone actually watch the
simulated SO-101 arm move under SmolVLA inference, we add an **optional live
viewer**: a real 3D MuJoCo window (`mujoco.viewer.launch_passive`) attached to
the *same* env instance the loop already steps, so the window shows exactly
what inference drives. It lives **inside the sim server process as a mode**
(off by default), not as a new demo component — the viewer is a rendering facet
of the one unit that owns the MuJoCo model/data; a separate component would have
to duplicate env access or expose the server's internals, braiding two units
over one env's state.

Two structural consequences a reader would otherwise question. First, the live
window must run on the process **main thread** (a MuJoCo/macOS requirement), so
in viewer mode the ZeroMQ REP loop moves to a **background thread** while the
viewer holds the main thread — the server is already structured for this (its
`serve_forever`/`wait_until_ready` split anticipates background serving), so no
wire-contract or request-handling change is needed. Second, the live window on
macOS requires launching under **`mjpython`** (the main-thread launcher) rather
than plain `python`; the design records this honestly — the live viewer is a
**dev-time aid, macOS via mjpython** (a plain window works on Linux), while
**headless `rgb_array` stays the portable default** the loop and every test use.

We considered frame-streaming the existing `rgb_array` output to a browser or
window (cross-platform, no mjpython) but chose the native window for fidelity —
an interactive 3D scene you can orbit, showing the actual arm the loop drives,
not a 2D video feed. The viewer is **presentation only**: it adds no model or
queue logic (the demo's foundation invariant holds), does not touch the
`infer_action` port, the sim wire contract, `SimEnvAdapter`, or `ControlLoop`,
and leaves ADR-0011's real-hardware no-goal intact — a screen is not hardware.
