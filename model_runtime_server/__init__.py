"""ZeroMQ server wrapping SmolVLAModel.infer_action() for the infer_action
port's Python fallback adapter.

See docs/design/control-loop/design.md component 01.3 ("ZeroMQ client" --
this package is that component's server-side counterpart) and
docs/adr/0007-msgpack-wire-format-for-zeromq-fallback.md for the wire
format this package implements.
"""
