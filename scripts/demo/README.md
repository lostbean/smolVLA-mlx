# Demo launcher helpers

Thin launchers for the Level-2 closed-loop demo — watch the simulated SO-101
arm move in real time as SmolVLA drives it, over the two-node BEAM cluster.
Three terminals, run from the repo root inside `nix develop`.

## Terminal 1 — sim server + live window

```bash
./scripts/demo/sim-server-viewer.sh
```

Opens the Python sim server with a live 3D MuJoCo window (macOS uses `mjpython`
automatically). Leave it running; the arm moves here once the loop drives it.
Add `--headless` to run without the window. Close the window to shut down.

## Terminal 2 — inference node

```bash
./scripts/demo/inference-node.sh
```

Opens `iex` named `inference@127.0.0.1`, resolves the local `smolvla_base`
checkpoint, and loads it into a named `InferenceServer` on boot. Wait for
`[inference node ready ...]`. Leave it running.

## Terminal 3 — sim node (drives the loop)

```bash
./scripts/demo/sim-node.sh
```

Opens `iex` named `sim@127.0.0.1`, joins the cluster, starts the sim env adapter
+ production `ControlLoop`, and binds `loop` in the REPL. Then drive the closed
loop:

```elixir
Demo.SimNode.run_loop(loop, 200, 50)   # 200 ticks, 50ms apart
```

The arm moves in Terminal 1's window: each action steps the sim, the new frame
drives the next inference — the closed loop.

## Config (env var overrides)

All three read these, with the defaults shown:

| Var | Default | Meaning |
| --- | --- | --- |
| `DEMO_COOKIE` | `demo` | shared BEAM cluster cookie (both nodes must match) |
| `DEMO_INFERENCE_NODE` | `inference@127.0.0.1` | inference node name |
| `DEMO_SIM_NODE` | `sim@127.0.0.1` | sim node name |
| `DEMO_SIM_ADDRESS` | `tcp://127.0.0.1:5556` | sim server ZeroMQ address |

## Notes

- The `--viewer` window needs `mjpython` (present in `.venv/bin/`) on macOS; the
  scripts fall back to plain `python` (headless) if it is absent.
- If a run leaves a sim server holding the port (MuJoCo ignores SIGTERM), clear
  it with `pkill -9 -f sim_server` before the next run.
- The two node scripts load their setup via `--dot-iex` (a sibling `.iex.exs`),
  so bindings like `loop` stay live in the REPL — `--eval` would not keep them.
