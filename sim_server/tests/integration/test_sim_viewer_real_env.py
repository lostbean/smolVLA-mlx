"""Real-MuJoCo integration check for the optional live-viewer wiring.

Loads the real so101-nexus SO-101 env in-process and proves the things the
viewer mode adds that need the REAL env (not the FakeSimEnv), WITHOUT opening a
real GUI window (impossible headless -- a live window is a true external):

  * ``SimEnv.mujoco_model_data()`` returns the env's ACTUAL ``MjModel`` /
    ``MjData`` objects -- the exact instances the env steps, of the right
    MuJoCo types, and stable across calls (ADR-0013 criterion 3);
  * with ``mujoco.viewer.launch_passive`` MOCKED at that boundary only,
    ``run_with_viewer`` hands the env's REAL model/data to launch_passive,
    serves on a background thread, and shuts the server down cleanly when the
    (mocked) window closes (criteria 3/5).

Why this test does NOT drive real reset/step/render over the wire here: on
macOS, MuJoCo's ``rgb_array`` offscreen rendering is bound to the thread/GL
context that first touches it and cannot be safely driven from the background
serve thread while the MAIN thread is parked in a *mocked* viewer loop that
never pumps a real GL window. That is precisely the situation the real
``mjpython -m sim_server --viewer`` deployment avoids -- there the live
``launch_passive`` window owns the main-thread GL context. The
backgrounded-serve-answers-the-wire behaviour is proven separately, GL-free,
against ``FakeSimEnv`` in the fast suite (``sim_server/tests/test_viewer.py``);
this test proves only the real-env-specific wiring that the fake cannot.

The actual 3D window is a documented MANUAL check, not automatable here::

    uv run mjpython -m sim_server --viewer

Gated like the sibling real-env test; opt in with
``RUN_SMOLVLA_INTEGRATION_CHECK=1``.
"""

import os
import threading

import mujoco
import pytest

_RUN_FLAG = "RUN_SMOLVLA_INTEGRATION_CHECK"

pytestmark = pytest.mark.skipif(
    os.environ.get(_RUN_FLAG) != "1",
    reason=(
        f"Loads a real MuJoCo SO-101 env; set {_RUN_FLAG}=1 to opt in "
        f"(see module docstring)."
    ),
)

TEST_ADDRESS = "tcp://127.0.0.1:0"


def test_accessor_returns_the_real_envs_model_and_data():
    from sim_server.env import SimEnv

    env = SimEnv(seed=0)
    try:
        env.reset()
        model, data = env.mujoco_model_data()

        # Right MuJoCo types.
        assert isinstance(model, mujoco.MjModel)
        assert isinstance(data, mujoco.MjData)

        # The SAME instances the env holds -- not copies.
        assert model is env._env.unwrapped.model
        assert data is env._env.unwrapped.data

        # Stable across calls (same objects the loop keeps stepping).
        model2, data2 = env.mujoco_model_data()
        assert model2 is model
        assert data2 is data
    finally:
        env.close()


def test_run_with_viewer_wires_real_model_data_and_shuts_down_cleanly(monkeypatch):
    """run_with_viewer hands launch_passive the env's REAL model/data, serves
    on a background thread, and on window close stops the server and returns.
    No real window opens (launch_passive is mocked) and no real render is
    driven from the background thread (see module docstring)."""
    from sim_server import viewer as viewer_mod
    from sim_server.env import SimEnv
    from sim_server.server import SimServer

    class FakeViewerHandle:
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

    handle = FakeViewerHandle()
    captured = {}

    def fake_launch_passive(model, data):
        captured["model"] = model
        captured["data"] = data
        return handle

    monkeypatch.setattr(viewer_mod.mujoco.viewer, "launch_passive", fake_launch_passive)

    env = SimEnv(seed=0)
    real_model, real_data = None, None
    try:
        env.reset()
        real_model, real_data = env.mujoco_model_data()
        server = SimServer(env, address=TEST_ADDRESS)
        captured_bound = {}

        def close_after_serving_and_a_sync():
            # Prove the serve loop actually reached the bound/ready state on the
            # background thread, and the viewer synced at least once, before we
            # simulate the human closing the window. Capture bound_address now,
            # because serve_forever clears it once it stops.
            server.wait_until_ready(timeout=30.0)
            captured_bound["address"] = server.bound_address
            while handle.sync_count < 1:
                pass
            handle.close()

        closer = threading.Thread(target=close_after_serving_and_a_sync, daemon=True)
        closer.start()
        # Holds this (main) thread until the window "closes".
        viewer_mod.run_with_viewer(server, env)
        closer.join(timeout=30.0)
    finally:
        env.close()

    # launch_passive got the env's real objects (criterion 3).
    assert captured["model"] is real_model
    assert captured["data"] is real_data
    assert isinstance(captured["model"], mujoco.MjModel)
    assert isinstance(captured["data"], mujoco.MjData)

    # The background serve loop bound and became ready (criterion 2 wiring).
    assert captured_bound["address"].startswith("tcp://127.0.0.1:")
    # The viewer synced the window while it ran (presentation).
    assert handle.sync_count >= 1
    # Window close -> clean shutdown (criterion 5).
    assert server._stop_requested is True


def test_concurrent_step_and_sync_do_not_race_on_mjdata():
    """Regression for the "mj_copyDataVisual: stack is in use" crash.

    In viewer mode the serve thread mutates MjData (step) while the viewer
    reads it (sync) from another thread. MjData is not thread-safe, so without
    a shared lock a real sync colliding with a real step aborts MuJoCo after a
    tick or two. This drives exactly that collision -- many real steps on one
    thread against many real ``mj_copyData`` reads (what viewer.sync() does
    internally) on another -- both taking ``env.data_lock`` the way the server
    and viewer do, and asserts no MuJoCo abort occurs.
    """
    from sim_server.env import SimEnv

    env = SimEnv(seed=0)
    try:
        env.reset()
        model, data = env.mujoco_model_data()
        lock = env.data_lock

        errors = []
        stop = threading.Event()
        neutral = [0.0] * 6

        def stepper():
            try:
                while not stop.is_set():
                    env.step(neutral)  # takes env.data_lock internally
            except Exception as exc:  # pragma: no cover - the bug path
                errors.append(("step", exc))

        def syncer():
            # A scratch MjData is the copy target; mj_copyData reads the live
            # `data` exactly as the passive viewer's sync does. Under the lock
            # this must never hit "stack is in use".
            scratch = mujoco.MjData(model)
            try:
                for _ in range(400):
                    with lock:
                        mujoco.mj_copyData(scratch, model, data)
            except Exception as exc:  # pragma: no cover - the bug path
                errors.append(("sync", exc))

        t_step = threading.Thread(target=stepper, daemon=True)
        t_sync = threading.Thread(target=syncer, daemon=True)
        t_step.start()
        t_sync.start()
        t_sync.join(timeout=60.0)
        stop.set()
        t_step.join(timeout=10.0)

        assert not errors, f"MjData race under the lock: {errors}"
    finally:
        env.close()
