"""Wire-protocol tests for the infer_action ZeroMQ server.

Per the work-order's TDD directive, these tests assert behavior through the
actual wire protocol: a real client socket sending real MessagePack-encoded
requests over a real ZeroMQ socket bound to an ephemeral local port, asserting
on real MessagePack-encoded responses -- never by calling the server's
internal Python functions directly.

A real SmolVLAModel (real checkpoint weights, real forward pass) is slow
(~seconds) and is a true external dependency -- out of scope for this fast
suite. These tests inject a lightweight fake model satisfying
`infer_action(image, state, instruction)`'s interface instead (see
``FakeModel`` below), per the "dependency-inject the boundary" rule. The one
test that exercises the real checkpoint end-to-end lives separately, opt-in,
gated by an env var (see model_runtime_server/tests/integration/).
"""

import threading

import msgpack
import numpy as np
import pytest
import zmq

from model_runtime_server.server import InferActionServer

TEST_ADDRESS = "tcp://127.0.0.1:0"  # ephemeral port; server reports the bound one back


class FakeModel:
    """Satisfies SmolVLAModel's infer_action() interface without loading any
    real weights -- fixed-shape, deterministic, fast. Mirrors the real
    model's config surface only as far as the server's validation logic
    needs (``config.max_state_dim``)."""

    class _Config:
        max_state_dim = 6

    def __init__(self, chunk_size=50, action_dim=32):
        self.config = self._Config()
        self._chunk_size = chunk_size
        self._action_dim = action_dim
        self.calls = []

    def infer_action(self, image, state, instruction):
        self.calls.append((image, state, instruction))
        return np.ones((self._chunk_size, self._action_dim), dtype=np.float32)


@pytest.fixture
def server():
    model = FakeModel()
    srv = InferActionServer(model, address=TEST_ADDRESS)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    srv.wait_until_ready(timeout=5.0)
    yield srv
    srv.stop()
    thread.join(timeout=5.0)


def make_client(address):
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, 5000)
    sock.setsockopt(zmq.SNDTIMEO, 5000)
    sock.connect(address)
    return sock


def make_request(image=None, state=None, instruction="pick up the block"):
    if image is None:
        image = np.zeros((4, 4, 3), dtype=np.uint8)
    if state is None:
        state = np.zeros(6, dtype=np.float32)
    return {
        "image": image.tobytes(),
        "image_shape": list(image.shape),
        "state": [float(x) for x in state],
        "instruction": instruction,
    }


class TestRoundTrip:
    """Trivial request/response plumbing: real socket, real msgpack, fake
    model -- proves the wire format before any real inference is wired in.
    """

    def test_valid_request_returns_ok_action_chunk(self, server):
        client = make_client(server.bound_address)
        client.send(msgpack.packb(make_request(), use_bin_type=True))
        raw = client.recv()
        response = msgpack.unpackb(raw, raw=False)

        assert "ok" in response
        assert "error" not in response
        action_chunk = response["ok"]["action_chunk"]
        assert len(action_chunk) == 50
        assert len(action_chunk[0]) == 32

    def test_server_reachable_from_separate_client_process_socket(self, server):
        """Two independent client sockets (simulating separate processes)
        can each round-trip a request against the same running server."""
        client_a = make_client(server.bound_address)
        client_b = make_client(server.bound_address)

        client_a.send(msgpack.packb(make_request(instruction="a"), use_bin_type=True))
        response_a = msgpack.unpackb(client_a.recv(), raw=False)
        assert "ok" in response_a

        client_b.send(msgpack.packb(make_request(instruction="b"), use_bin_type=True))
        response_b = msgpack.unpackb(client_b.recv(), raw=False)
        assert "ok" in response_b

    def test_server_calls_model_infer_action_with_decoded_fields(self, server):
        client = make_client(server.bound_address)
        image = (np.arange(4 * 4 * 3, dtype=np.uint8)).reshape(4, 4, 3)
        state = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
        client.send(
            msgpack.packb(
                make_request(image=image, state=state, instruction="go left"),
                use_bin_type=True,
            )
        )
        msgpack.unpackb(client.recv(), raw=False)

        assert len(server.model.calls) == 1
        called_image, called_state, called_instruction = server.model.calls[0]
        np.testing.assert_array_equal(called_image, image)
        np.testing.assert_allclose(called_state, state)
        assert called_instruction == "go left"


class TestErrorHandling:
    def test_state_length_exceeding_max_state_dim_rejected_before_inference(
        self, server
    ):
        """Fail-loud-before-forward-pass: a state vector LONGER than the
        loaded checkpoint's max_state_dim is rejected with the error shape,
        and infer_action() is never called. Mirrors infer_action()'s own
        validation exactly (see smolvla.py) -- a state SHORTER than
        max_state_dim is valid (infer_action() zero-pads it internally),
        only exceeding it is a genuine dimensionality mismatch."""
        client = make_client(server.bound_address)
        # FakeModel declares max_state_dim=6.
        too_long_state = np.zeros(7, dtype=np.float32)
        client.send(
            msgpack.packb(make_request(state=too_long_state), use_bin_type=True)
        )
        response = msgpack.unpackb(client.recv(), raw=False)

        assert "error" in response
        assert "ok" not in response
        assert isinstance(response["error"]["message"], str)
        assert len(server.model.calls) == 0

    def test_state_shorter_than_max_state_dim_is_accepted(self, server):
        """A state vector shorter than max_state_dim is a valid request --
        infer_action() zero-pads it to the checkpoint's declared state
        width internally (see smolvla.py's own padding path), so this must
        NOT be rejected at the server's validation boundary."""
        client = make_client(server.bound_address)
        short_state = np.array([1.0, 2.0, 3.0], dtype=np.float32)  # < max_state_dim=6
        client.send(msgpack.packb(make_request(state=short_state), use_bin_type=True))
        response = msgpack.unpackb(client.recv(), raw=False)

        assert "ok" in response
        assert len(server.model.calls) == 1

    def test_missing_required_field_rejected(self, server):
        client = make_client(server.bound_address)
        request = make_request()
        del request["instruction"]
        client.send(msgpack.packb(request, use_bin_type=True))
        response = msgpack.unpackb(client.recv(), raw=False)

        assert "error" in response
        assert len(server.model.calls) == 0

    def test_wrong_type_field_rejected(self, server):
        client = make_client(server.bound_address)
        request = make_request()
        request["state"] = "not a list of floats"
        client.send(msgpack.packb(request, use_bin_type=True))
        response = msgpack.unpackb(client.recv(), raw=False)

        assert "error" in response
        assert len(server.model.calls) == 0

    def test_undecodable_bytes_does_not_crash_server_and_still_replies(self, server):
        """REQ/REP requires exactly one reply per request or the socket
        state machine desyncs. Bytes that are not valid MessagePack at all
        must still produce a reply (the error shape), never a dropped
        response and never a crashed server."""
        client = make_client(server.bound_address)
        client.send(b"\xff\xff\xff not valid msgpack \x00\x01")
        response = msgpack.unpackb(client.recv(), raw=False)

        assert "error" in response

        # Server must still be alive and serving afterwards.
        client.send(msgpack.packb(make_request(), use_bin_type=True))
        response2 = msgpack.unpackb(client.recv(), raw=False)
        assert "ok" in response2

    def test_inference_failure_returns_error_not_crash(self, server, monkeypatch):
        def boom(image, state, instruction):
            raise RuntimeError("simulated inference failure")

        server.model.infer_action = boom

        client = make_client(server.bound_address)
        client.send(msgpack.packb(make_request(), use_bin_type=True))
        response = msgpack.unpackb(client.recv(), raw=False)

        assert "error" in response
        assert "simulated inference failure" in response["error"]["message"]


class TestDisconnectSurvival:
    def test_server_survives_client_disconnect_mid_request(self, server):
        """A client that sends a request then vanishes before reading the
        response must not crash or wedge the server for subsequent,
        different clients' requests."""
        vanishing_client = make_client(server.bound_address)
        vanishing_client.send(msgpack.packb(make_request(), use_bin_type=True))
        # Vanish without reading the reply.
        vanishing_client.close()

        # A fresh client must still be served normally afterwards.
        client = make_client(server.bound_address)
        client.send(msgpack.packb(make_request(), use_bin_type=True))
        response = msgpack.unpackb(client.recv(), raw=False)
        assert "ok" in response

    def test_server_survives_client_that_never_sends(self, server):
        """A client that connects but never sends a request at all must not
        wedge the server for other clients."""
        idle_client = make_client(server.bound_address)  # noqa: F841 -- deliberately unused

        client = make_client(server.bound_address)
        client.send(msgpack.packb(make_request(), use_bin_type=True))
        response = msgpack.unpackb(client.recv(), raw=False)
        assert "ok" in response
