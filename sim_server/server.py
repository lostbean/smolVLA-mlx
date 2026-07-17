"""ZeroMQ REQ/REP server exposing a LeRobot/MuJoCo SO-101 gym env over the
network, per demo component 01.1 (the "sim server", docs/design/demo/design.md)
and docs/adr/0012-sim-env-bridged-via-python-sim-server-over-zeromq.md.

This is the sibling of model_runtime_server.InferActionServer: same transport
(a ZeroMQ REP socket), same wire format (MessagePack, string keys), same
socket-timeout / wait_until_ready machinery, same one-reply-per-request
robustness contract. Where the infer_action server wraps a loaded model, this
one wraps a loaded gym env (dependency-injected so fast tests can substitute a
lightweight fake -- see sim_server/tests/test_server.py) and answers the three
env operations the Elixir sim-env adapter drives every tick:

    {"op": "reset"}                     -> {"ok": {image, image_shape, state}}
    {"op": "step", "action": [f, ...]}  -> {"ok": {image, image_shape, state}}
    {"op": "render"}                    -> {"ok": {image, image_shape}}

Errors use the same shape as the infer_action server:
    {"error": {"message": <str>}}
"""

import logging
import time
from typing import Optional

import msgpack
import zmq

logger = logging.getLogger(__name__)

DEFAULT_ADDRESS = "tcp://*:5556"

# Operation discriminators carried in the request map's "op" field.
OP_RESET = "reset"
OP_STEP = "step"
OP_RENDER = "render"
_VALID_OPS = (OP_RESET, OP_STEP, OP_RENDER)


class SimServer:
    """Binds a ZeroMQ REP socket and serves ``reset`` / ``step`` / ``render``
    requests against an injected sim env, one request at a time.

    The env is dependency-injected rather than constructed internally so fast
    tests can substitute a lightweight fake (see sim_server/tests/test_server.py);
    production use passes a real ``SimEnv()`` wrapping the MuJoCo gym env (see
    ``sim_server.__main__``).
    """

    def __init__(self, env, address: str = DEFAULT_ADDRESS):
        self.env = env
        self._address = address
        self._context = zmq.Context.instance()
        self._socket = None
        self._bound_address: Optional[str] = None
        self._stop_requested = False

    @property
    def bound_address(self) -> str:
        """The address the socket actually bound to. Differs from the
        constructor's ``address`` when a wildcard port (``:0``) was requested
        -- ZeroMQ resolves it to the real ephemeral port, needed so tests (and
        callers generally) can connect to it."""
        if self._bound_address is None:
            raise RuntimeError("server has not bound its socket yet")
        return self._bound_address

    def wait_until_ready(self, timeout: float = 5.0) -> None:
        """Blocks until the server's socket is bound and it is ready to accept
        requests, or raises TimeoutError. Used by tests/callers that start
        ``serve_forever`` on a background thread and need to know when
        ``bound_address`` is safe to read."""
        deadline = time.monotonic() + timeout
        while self._bound_address is None:
            if time.monotonic() > deadline:
                raise TimeoutError("server did not become ready in time")
            time.sleep(0.005)

    def serve_forever(self) -> None:
        """Binds the REP socket and serves requests indefinitely (or until
        ``stop()`` is called). One request is handled at a time, matching
        REQ/REP's inherently synchronous pattern -- the sim advances one step
        per request, so serialization is inherent to the env, not just the
        socket.
        """
        socket = self._context.socket(zmq.REP)
        # A bounded poll timeout lets the loop notice stop() requests promptly
        # instead of blocking forever in recv().
        socket.setsockopt(zmq.RCVTIMEO, 200)
        socket.setsockopt(zmq.SNDTIMEO, 5000)
        socket.setsockopt(zmq.LINGER, 0)
        socket.bind(self._address)
        self._bound_address = socket.getsockopt_string(zmq.LAST_ENDPOINT)
        self._socket = socket
        logger.info("sim server listening on %s", self._bound_address)

        try:
            while not self._stop_requested:
                try:
                    raw_request = socket.recv()
                except zmq.Again:
                    continue  # poll timeout, no request pending -- check stop flag
                response = self._handle_raw_request(raw_request)
                socket.send(response)
        finally:
            socket.close()
            self._socket = None
            self._bound_address = None

    def stop(self) -> None:
        """Requests that ``serve_forever`` return after its current
        recv-timeout tick. Safe to call from another thread."""
        self._stop_requested = True

    # ------------------------------------------------------------------
    # Request handling.
    # ------------------------------------------------------------------
    def _handle_raw_request(self, raw_request: bytes) -> bytes:
        """Decodes, validates, and dispatches one request, always returning
        encoded response bytes -- never raises. REQ/REP requires exactly one
        reply per request or the socket's state machine desyncs, so every
        failure path (undecodable bytes, unknown op, a bad action, an env that
        raised) must still produce an ``error`` response rather than propagate.
        """
        try:
            request = msgpack.unpackb(raw_request, raw=False)
        except Exception as exc:
            logger.warning("failed to decode request as MessagePack: %s", exc)
            return self._encode_error(
                f"malformed request: could not decode MessagePack: {exc}"
            )

        if not isinstance(request, dict):
            return self._encode_error(
                f"malformed request: expected a map, got {type(request).__name__}"
            )

        op = request.get("op")
        if op not in _VALID_OPS:
            return self._encode_error(
                f"malformed request: 'op' must be one of {list(_VALID_OPS)}, got {op!r}"
            )

        try:
            payload = self._dispatch(op, request)
        except ValueError as exc:
            # A rejected request (e.g. a wrong-shape action) -- fail loud, but
            # the env was not advanced into a bad state.
            logger.info("rejected malformed %s request: %s", op, exc)
            return self._encode_error(str(exc))
        except Exception as exc:
            # A live env failure (dead sim, a step that raised). Surface loud
            # over the wire rather than fabricating a frame.
            logger.exception("%s raised", op)
            return self._encode_error(f"{op} failed: {exc}")

        return self._encode_ok(payload)

    def _dispatch(self, op: str, request: dict) -> dict:
        if op == OP_RESET:
            return self.env.reset()
        if op == OP_RENDER:
            return self.env.render()
        # OP_STEP
        if "action" not in request:
            raise ValueError("malformed request: 'step' requires an 'action' field")
        return self.env.step(request["action"])

    def _encode_ok(self, payload: dict) -> bytes:
        return msgpack.packb({"ok": payload}, use_bin_type=True)

    def _encode_error(self, message: str) -> bytes:
        return msgpack.packb({"error": {"message": message}}, use_bin_type=True)


__all__ = ["SimServer", "DEFAULT_ADDRESS", "OP_RESET", "OP_STEP", "OP_RENDER"]
