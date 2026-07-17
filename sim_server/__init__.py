"""ZeroMQ server wrapping a LeRobot/MuJoCo SO-101 pick-and-place gym env --
the demo's "sim server" (docs/design/demo/design.md component 01.1).

The sibling of model_runtime_server: same ZeroMQ REP transport and MessagePack
wire format, but it answers env ``reset`` / ``step`` / ``render`` instead of
``infer_action``. See docs/adr/0012-sim-env-bridged-via-python-sim-server-over-zeromq.md
for why a foreign wire format is correct at this seam (a true external Python
physics engine).
"""
