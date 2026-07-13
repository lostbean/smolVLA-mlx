"""ZeroMQ REQ/REP server exposing SmolVLAModel.infer_action() over the
network, per docs/design/control-loop/design.md component 01.3 and
docs/adr/0007-msgpack-wire-format-for-zeromq-fallback.md.

This is the permanent, standing-service Python fallback adapter for the
infer_action port (ADR-0003): a future Elixir client is the other side of
the same wire contract. No new inference logic lives here -- this module is
purely a request/response wrapper around an already-loaded model's
``infer_action(image, state, instruction)``.
"""

import logging
import time
from typing import Optional

import msgpack
import numpy as np
import zmq

logger = logging.getLogger(__name__)

DEFAULT_ADDRESS = "tcp://*:5555"


class InferActionServer:
    """Binds a ZeroMQ REP socket and serves ``infer_action`` requests
    against an injected model, one request at a time.

    The model is dependency-injected rather than loaded internally so fast
    tests can substitute a lightweight fake (see
    model_runtime_server/tests/test_server.py); production use passes a
    real ``SmolVLAModel.from_pretrained(checkpoint_path)`` instance (see
    ``model_runtime_server.__main__``).
    """

    def __init__(self, model, address: str = DEFAULT_ADDRESS):
        self.model = model
        self._address = address
        self._context = zmq.Context.instance()
        self._socket = None
        self._bound_address: Optional[str] = None
        self._stop_requested = False

    @property
    def bound_address(self) -> str:
        """The address the socket actually bound to. Differs from the
        constructor's ``address`` when a wildcard port (``:0``) was
        requested -- ZeroMQ resolves it to the real ephemeral port, needed
        so tests (and callers generally) can connect to it."""
        if self._bound_address is None:
            raise RuntimeError("server has not bound its socket yet")
        return self._bound_address

    def wait_until_ready(self, timeout: float = 5.0) -> None:
        """Blocks until the server's socket is bound and it is ready to
        accept requests, or raises TimeoutError. Used by tests/callers that
        start ``serve_forever`` on a background thread and need to know
        when ``bound_address`` is safe to read."""
        deadline = time.monotonic() + timeout
        while self._bound_address is None:
            if time.monotonic() > deadline:
                raise TimeoutError("server did not become ready in time")
            time.sleep(0.005)

    def serve_forever(self) -> None:
        """Binds the REP socket and serves requests indefinitely (or until
        ``stop()`` is called). One request is handled at a time, matching
        REQ/REP's inherently synchronous pattern and the infer_action
        port's own synchronous contract (ADR-0002).
        """
        socket = self._context.socket(zmq.REP)
        # A bounded poll timeout lets the loop notice stop() requests
        # promptly instead of blocking forever in recv().
        socket.setsockopt(zmq.RCVTIMEO, 200)
        socket.setsockopt(zmq.SNDTIMEO, 5000)
        socket.setsockopt(zmq.LINGER, 0)
        socket.bind(self._address)
        self._bound_address = socket.getsockopt_string(zmq.LAST_ENDPOINT)
        self._socket = socket
        logger.info("infer_action server listening on %s", self._bound_address)

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
        encoded response bytes -- never raises. REQ/REP requires exactly
        one reply per request or the socket's state machine desyncs, so
        every failure path (undecodable bytes, malformed request, a
        dimensionality mismatch, an inference exception) must still produce
        an ``error`` response rather than propagate.
        """
        try:
            request = msgpack.unpackb(raw_request, raw=False)
        except Exception as exc:
            logger.warning("failed to decode request as MessagePack: %s", exc)
            return self._encode_error(
                f"malformed request: could not decode MessagePack: {exc}"
            )

        try:
            image, state, instruction = self._validate_request(request)
        except ValueError as exc:
            logger.info("rejected malformed request: %s", exc)
            return self._encode_error(str(exc))

        try:
            action_chunk = self.model.infer_action(image, state, instruction)
        except Exception as exc:
            logger.exception("infer_action() raised")
            return self._encode_error(f"inference failed: {exc}")

        return self._encode_ok(action_chunk)

    def _validate_request(self, request):
        """Validates the decoded request against the wire schema and this
        server's own fail-loud-before-forward-pass invariant (state length
        vs. the loaded checkpoint's max_state_dim), raising ValueError with
        a human-readable message on any violation. Never coerces or
        truncates -- a malformed request is rejected outright.
        """
        if not isinstance(request, dict):
            raise ValueError(
                f"malformed request: expected a map, got {type(request).__name__}"
            )

        required_fields = ("image", "image_shape", "state", "instruction")
        missing = [f for f in required_fields if f not in request]
        if missing:
            raise ValueError(f"malformed request: missing field(s) {missing}")

        image_bytes = request["image"]
        image_shape = request["image_shape"]
        state = request["state"]
        instruction = request["instruction"]

        if not isinstance(image_bytes, (bytes, bytearray)):
            raise ValueError("malformed request: 'image' must be binary bytes")
        if (
            not isinstance(image_shape, (list, tuple))
            or len(image_shape) != 3
            or not all(isinstance(d, int) for d in image_shape)
        ):
            raise ValueError(
                "malformed request: 'image_shape' must be [height, width, channels]"
            )
        if not isinstance(state, (list, tuple)) or not all(
            isinstance(x, (int, float)) for x in state
        ):
            raise ValueError("malformed request: 'state' must be an array of numbers")
        if not isinstance(instruction, str):
            raise ValueError("malformed request: 'instruction' must be a string")

        height, width, channels = image_shape
        expected_bytes = height * width * channels
        if len(image_bytes) != expected_bytes:
            raise ValueError(
                f"malformed request: 'image' has {len(image_bytes)} bytes, "
                f"expected {expected_bytes} for image_shape {image_shape}"
            )

        # Mirrors infer_action()'s own validation exactly (see
        # SmolVLAModel.infer_action's shape check in
        # vendor/mlx-vlm/mlx_vlm/models/smolvla/smolvla.py): a state vector
        # SHORTER than max_state_dim is valid -- infer_action() zero-pads it
        # to the checkpoint's full state width -- but one LONGER than
        # max_state_dim is a genuine dimensionality mismatch and must be
        # rejected here, before any inference runs, never silently
        # truncated.
        max_state_dim = self.model.config.max_state_dim
        if len(state) > max_state_dim:
            raise ValueError(
                f"malformed request: 'state' has length {len(state)}, which "
                f"exceeds the loaded checkpoint's declared "
                f"max_state_dim={max_state_dim}"
            )

        image = np.frombuffer(bytes(image_bytes), dtype=np.uint8).reshape(
            height, width, channels
        )
        state_arr = np.array(state, dtype=np.float32)
        return image, state_arr, instruction

    def _encode_ok(self, action_chunk) -> bytes:
        action_list = np.asarray(action_chunk, dtype=np.float32).tolist()
        return msgpack.packb({"ok": {"action_chunk": action_list}}, use_bin_type=True)

    def _encode_error(self, message: str) -> bytes:
        return msgpack.packb({"error": {"message": message}}, use_bin_type=True)


__all__ = ["InferActionServer", "DEFAULT_ADDRESS"]
