"""Wiring tests for the optional live-viewer mode of the sim server.

``mujoco.viewer.launch_passive`` is a TRUE external (a GUI window that cannot
open in a headless test), so it is the ONLY thing mocked here -- everything
else is real: a real ``SimServer`` bound to a real ephemeral ZeroMQ socket, a
real background serve thread, a real REQ client. These tests assert the WIRING
the viewer mode introduces:

  * the ``--viewer`` flag defaults OFF (headless path unchanged);
  * in viewer mode the ZeroMQ serve loop runs on a BACKGROUND thread and still
    answers reset/step/render with byte-identical replies while the (mocked)
    viewer "holds the main thread";
  * closing the window shuts the server down cleanly;
  * ``SimEnv`` exposes the env's real MuJoCo model/data for ``launch_passive``.

The real 3D window itself is NOT asserted here (it cannot open headless) -- see
the module docstring of ``sim_server.viewer`` for the manual ``mjpython`` command
a human runs to actually watch the arm.
"""

import threading

import msgpack

from sim_server.__main__ import _parse_args
from sim_server.server import SimServer
from sim_server.tests.test_server import FakeSimEnv, make_client, roundtrip

TEST_ADDRESS = "tcp://127.0.0.1:0"


def test_viewer_flag_defaults_off():
    """Acceptance criterion 1: --viewer exists and defaults OFF, so plain
    `python -m sim_server` stays headless."""
    args = _parse_args([])
    assert args.viewer is False


def test_viewer_flag_can_be_enabled():
    args = _parse_args(["--viewer"])
    assert args.viewer is True


class FakeViewerHandle:
    """Stands in for the object ``mujoco.viewer.launch_passive`` returns: a
    context manager whose ``is_running()`` reports whether the window is open,
    and ``sync()`` pushes sim state to the window. We flip ``_running`` to
    False from the test to simulate a human closing the window."""

    def __init__(self):
        self._running = True
        self.sync_count = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._running = False
        return False

    def is_running(self):
        return self._running

    def sync(self):
        self.sync_count += 1

    def close(self):
        self._running = False


def test_launch_passive_is_called_with_the_envs_model_and_data(monkeypatch):
    """Structural: viewer mode passes the env's ACTUAL model/data objects
    (from the SimEnv accessor) to launch_passive -- same instance the loop
    steps, not a copy."""
    from sim_server import viewer as viewer_mod

    sentinel_model = object()
    sentinel_data = object()

    class EnvWithModelData(FakeSimEnv):
        def mujoco_model_data(self):
            return sentinel_model, sentinel_data

    captured = {}

    def fake_launch_passive(model, data):
        captured["model"] = model
        captured["data"] = data
        handle = FakeViewerHandle()
        # Close immediately so run_with_viewer returns rather than looping.
        handle._running = False
        return handle

    monkeypatch.setattr(viewer_mod.mujoco.viewer, "launch_passive", fake_launch_passive)

    env = EnvWithModelData()
    server = SimServer(env, address=TEST_ADDRESS)
    viewer_mod.run_with_viewer(server, env)

    assert captured["model"] is sentinel_model
    assert captured["data"] is sentinel_data


def test_background_serve_answers_while_viewer_holds_main_thread(monkeypatch):
    """Acceptance criterion 2: in viewer mode the serve loop runs on a
    background thread and answers reset/step/render correctly (byte-identical
    to headless) while the mocked viewer holds the calling ("main") thread."""
    from sim_server import viewer as viewer_mod

    handle = FakeViewerHandle()

    # The window "stays open" until the client is done; the test closes it.
    def fake_launch_passive(model, data):
        return handle

    monkeypatch.setattr(viewer_mod.mujoco.viewer, "launch_passive", fake_launch_passive)

    class EnvWithModelData(FakeSimEnv):
        def mujoco_model_data(self):
            return object(), object()

    env = EnvWithModelData()
    server = SimServer(env, address=TEST_ADDRESS)

    results = {}

    def drive_client():
        # Wait for the background serve loop to bind, then exercise the wire.
        server.wait_until_ready(timeout=5.0)
        client = make_client(server.bound_address)
        results["reset"] = roundtrip(client, {"op": "reset"})
        action = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        results["step"] = roundtrip(client, {"op": "step", "action": action})
        results["render"] = roundtrip(client, {"op": "render"})
        client.close()
        # Simulate the human closing the window -> run_with_viewer returns.
        handle.close()

    driver = threading.Thread(target=drive_client, daemon=True)
    driver.start()

    # run_with_viewer holds THIS thread (the "main" thread) with the viewer
    # loop until the window closes; the serve loop is on a background thread.
    viewer_mod.run_with_viewer(server, env)
    driver.join(timeout=5.0)

    assert "ok" in results["reset"]
    assert results["reset"]["ok"]["image_shape"] == [8, 8, 3]
    assert "ok" in results["step"]
    assert env.steps == [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]
    assert "ok" in results["render"]
    assert "state" not in results["render"]["ok"]


def test_replies_are_byte_identical_to_headless(monkeypatch):
    """Acceptance criterion 2/4: a client talking to a --viewer server gets
    byte-identical reply bytes to a headless server (the viewer reads, never
    alters replies)."""
    # Headless reference bytes.
    headless_env = FakeSimEnv()
    headless_server = SimServer(headless_env, address=TEST_ADDRESS)
    headless_thread = threading.Thread(
        target=headless_server.serve_forever, daemon=True
    )
    headless_thread.start()
    headless_server.wait_until_ready(timeout=5.0)
    hc = make_client(headless_server.bound_address)
    hc.send(msgpack.packb({"op": "reset"}, use_bin_type=True))
    headless_reset_bytes = hc.recv()
    hc.close()
    headless_server.stop()
    headless_thread.join(timeout=5.0)

    # Viewer-mode bytes for the same request.
    from sim_server import viewer as viewer_mod

    handle = FakeViewerHandle()
    monkeypatch.setattr(viewer_mod.mujoco.viewer, "launch_passive", lambda m, d: handle)

    class EnvWithModelData(FakeSimEnv):
        def mujoco_model_data(self):
            return object(), object()

    env = EnvWithModelData()
    server = SimServer(env, address=TEST_ADDRESS)

    captured = {}

    def drive_client():
        server.wait_until_ready(timeout=5.0)
        client = make_client(server.bound_address)
        client.send(msgpack.packb({"op": "reset"}, use_bin_type=True))
        captured["bytes"] = client.recv()
        client.close()
        handle.close()

    driver = threading.Thread(target=drive_client, daemon=True)
    driver.start()
    viewer_mod.run_with_viewer(server, env)
    driver.join(timeout=5.0)

    assert captured["bytes"] == headless_reset_bytes


def test_window_close_shuts_the_server_down_cleanly(monkeypatch):
    """Acceptance criterion 5: closing the window stops the background serve
    loop and returns -- the server does not keep running."""
    from sim_server import viewer as viewer_mod

    handle = FakeViewerHandle()
    monkeypatch.setattr(viewer_mod.mujoco.viewer, "launch_passive", lambda m, d: handle)

    class EnvWithModelData(FakeSimEnv):
        def mujoco_model_data(self):
            return object(), object()

    env = EnvWithModelData()
    server = SimServer(env, address=TEST_ADDRESS)

    def close_soon():
        server.wait_until_ready(timeout=5.0)
        handle.close()

    threading.Thread(target=close_soon, daemon=True).start()
    viewer_mod.run_with_viewer(server, env)

    # After run_with_viewer returns, the server was asked to stop.
    assert server._stop_requested is True


def test_viewer_syncs_the_window(monkeypatch):
    """The viewer pushes sim state to the window (viewer.sync) while it runs,
    so the arm visibly moves -- at least one sync happens before close."""
    from sim_server import viewer as viewer_mod

    handle = FakeViewerHandle()
    monkeypatch.setattr(viewer_mod.mujoco.viewer, "launch_passive", lambda m, d: handle)

    class EnvWithModelData(FakeSimEnv):
        def mujoco_model_data(self):
            return object(), object()

    env = EnvWithModelData()
    server = SimServer(env, address=TEST_ADDRESS)

    def close_after_a_few_syncs():
        server.wait_until_ready(timeout=5.0)
        while handle.sync_count < 2:
            pass
        handle.close()

    threading.Thread(target=close_after_a_few_syncs, daemon=True).start()
    viewer_mod.run_with_viewer(server, env)

    assert handle.sync_count >= 1
