"""Wire-protocol tests for the sim ZeroMQ server.

Per the work-order's TDD directive, these tests assert behavior through the
actual wire protocol: a real client socket sending real MessagePack-encoded
requests over a real ZeroMQ socket bound to an ephemeral local port, asserting
on real MessagePack-encoded responses -- never by calling the server's internal
Python functions directly.

A real MuJoCo SO-101 env is a true external dependency (a physics engine, real
wall-clock time) -- out of scope for this fast suite. These tests inject a
lightweight FakeSimEnv satisfying SimEnv's reset/step/render interface instead
(mirroring how model_runtime_server/tests/test_server.py injects a FakeModel),
per the "dependency-inject the boundary" rule. The one test that drives the
real MuJoCo env end-to-end lives separately, opt-in, gated by an env var (see
sim_server/tests/integration/).
"""

import threading

import msgpack
import numpy as np
import pytest
import zmq

from sim_server.server import SimServer

TEST_ADDRESS = "tcp://127.0.0.1:0"  # ephemeral port; server reports the bound one back


class FakeSimEnv:
    """Satisfies SimEnv's reset/step/render interface without loading MuJoCo --
    fixed-shape, deterministic, fast. Emits a distinct frame per step so tests
    can prove the frame advances, and rejects wrong-length actions exactly as
    the real SimEnv does (fail-loud at the env boundary)."""

    DOF = 6

    def __init__(self, h=8, w=8):
        self._h, self._w = h, w
        self._tick = 0
        self.steps = []
        self.reset_count = 0
        self.render_count = 0
        # Match SimEnv's interface: the viewer holds env.data_lock around sync().
        self.data_lock = threading.RLock()

    def _frame_payload(self):
        # A frame whose content depends on the tick count, so consecutive
        # renders/steps differ.
        img = np.full((self._h, self._w, 3), self._tick % 256, dtype=np.uint8)
        return {
            "image": img.tobytes(),
            "image_shape": [self._h, self._w, 3],
        }

    def _state(self):
        return [float(self._tick)] * self.DOF

    def reset(self):
        self.reset_count += 1
        self._tick = 0
        payload = self._frame_payload()
        payload["state"] = self._state()
        return payload

    def step(self, action):
        if not isinstance(action, (list, tuple, np.ndarray)):
            raise ValueError("'action' must be an array of numbers")
        if len(action) != self.DOF:
            raise ValueError(f"'action' has length {len(action)}, expected {self.DOF}")
        self.steps.append(list(action))
        self._tick += 1
        payload = self._frame_payload()
        payload["state"] = self._state()
        return payload

    def render(self):
        self.render_count += 1
        return self._frame_payload()


@pytest.fixture
def server():
    env = FakeSimEnv()
    srv = SimServer(env, address=TEST_ADDRESS)
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


def roundtrip(client, request):
    client.send(msgpack.packb(request, use_bin_type=True))
    return msgpack.unpackb(client.recv(), raw=False)


class TestResetAndRender:
    def test_reset_returns_initial_observation_payload(self, server):
        client = make_client(server.bound_address)
        response = roundtrip(client, {"op": "reset"})

        assert "ok" in response
        assert "error" not in response
        obs = response["ok"]
        assert isinstance(obs["image"], (bytes, bytearray))
        assert obs["image_shape"] == [8, 8, 3]
        assert len(obs["image"]) == 8 * 8 * 3
        assert isinstance(obs["state"], list)
        assert len(obs["state"]) == 6
        assert server.env.reset_count == 1

    def test_render_returns_frame_without_advancing(self, server):
        client = make_client(server.bound_address)
        roundtrip(client, {"op": "reset"})
        before_steps = len(server.env.steps)
        response = roundtrip(client, {"op": "render"})

        assert "ok" in response
        obs = response["ok"]
        assert isinstance(obs["image"], (bytes, bytearray))
        assert obs["image_shape"] == [8, 8, 3]
        # render must not carry state (it's a frame-only op) and must not step.
        assert len(server.env.steps) == before_steps
        assert server.env.render_count == 1


class TestStep:
    def test_step_advances_and_returns_frame_plus_state(self, server):
        client = make_client(server.bound_address)
        roundtrip(client, {"op": "reset"})
        action = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        response = roundtrip(client, {"op": "step", "action": action})

        assert "ok" in response
        obs = response["ok"]
        assert isinstance(obs["image"], (bytes, bytearray))
        assert obs["image_shape"] == [8, 8, 3]
        assert len(obs["state"]) == 6
        assert server.env.steps == [action]

    def test_state_dimensionality_within_max_state_dim(self, server):
        """Acceptance criterion 3: the proprioceptive state stays <= 6."""
        client = make_client(server.bound_address)
        obs = roundtrip(client, {"op": "reset"})["ok"]
        assert len(obs["state"]) <= 6
        obs2 = roundtrip(client, {"op": "step", "action": [0.0] * 6})["ok"]
        assert len(obs2["state"]) <= 6

    def test_consecutive_steps_advance_the_frame(self, server):
        """The frame the sim returns changes as the sim is driven -- the
        server does not return a frozen/fabricated image."""
        client = make_client(server.bound_address)
        roundtrip(client, {"op": "reset"})
        frames = []
        for _ in range(3):
            obs = roundtrip(client, {"op": "step", "action": [0.0] * 6})["ok"]
            frames.append(bytes(obs["image"]))
        assert frames[0] != frames[1]
        assert frames[1] != frames[2]


class TestErrorHandling:
    def test_unknown_op_rejected(self, server):
        client = make_client(server.bound_address)
        response = roundtrip(client, {"op": "teleport"})
        assert "error" in response
        assert "ok" not in response
        assert isinstance(response["error"]["message"], str)
        # No step was taken.
        assert len(server.env.steps) == 0

    def test_missing_op_rejected(self, server):
        client = make_client(server.bound_address)
        response = roundtrip(client, {"action": [0.0] * 6})
        assert "error" in response

    def test_step_without_action_rejected(self, server):
        client = make_client(server.bound_address)
        response = roundtrip(client, {"op": "step"})
        assert "error" in response
        assert len(server.env.steps) == 0

    def test_wrong_length_action_rejected_no_step_taken(self, server):
        """Acceptance criterion 4: a wrong action shape is an explicit error
        over the wire, never a silent no-op or a fabricated frame."""
        client = make_client(server.bound_address)
        roundtrip(client, {"op": "reset"})
        response = roundtrip(client, {"op": "step", "action": [0.0, 0.0, 0.0]})
        assert "error" in response
        assert "ok" not in response
        assert len(server.env.steps) == 0

    def test_non_map_request_rejected(self, server):
        client = make_client(server.bound_address)
        client.send(msgpack.packb([1, 2, 3], use_bin_type=True))
        response = msgpack.unpackb(client.recv(), raw=False)
        assert "error" in response

    def test_undecodable_bytes_does_not_crash_server_and_still_replies(self, server):
        """REQ/REP requires exactly one reply per request or the socket state
        machine desyncs. Bytes that are not valid MessagePack must still
        produce a reply (the error shape), never a crashed server."""
        client = make_client(server.bound_address)
        client.send(b"\xff\xff\xff not valid msgpack \x00\x01")
        response = msgpack.unpackb(client.recv(), raw=False)
        assert "error" in response

        # Server must still be alive afterwards.
        response2 = roundtrip(client, {"op": "reset"})
        assert "ok" in response2

    def test_env_step_failure_returns_error_not_crash(self, server, monkeypatch):
        def boom(action):
            raise RuntimeError("simulated env explosion")

        server.env.step = boom
        client = make_client(server.bound_address)
        response = roundtrip(client, {"op": "step", "action": [0.0] * 6})
        assert "error" in response
        assert "simulated env explosion" in response["error"]["message"]

        # And a subsequent good op still works (server survived).
        assert "ok" in roundtrip(client, {"op": "render"})


class TestDisconnectSurvival:
    def test_server_survives_client_disconnect_mid_request(self, server):
        vanishing_client = make_client(server.bound_address)
        vanishing_client.send(msgpack.packb({"op": "reset"}, use_bin_type=True))
        vanishing_client.close()

        client = make_client(server.bound_address)
        assert "ok" in roundtrip(client, {"op": "reset"})
