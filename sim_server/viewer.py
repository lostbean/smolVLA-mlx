"""The optional live-viewer mode of the sim server (ADR-0013, term
``sim-viewer`` in docs/design/demo/CONTEXT.md).

Off by default: plain ``python -m sim_server`` runs headless with the serve
loop on the main thread and never imports this module. When ``--viewer`` is
passed, ``__main__`` calls :func:`run_with_viewer`, which:

  * moves the ZeroMQ REP serve loop to a BACKGROUND thread, and
  * opens ``mujoco.viewer.launch_passive(model, data)`` on the process MAIN
    thread (a MuJoCo/macOS requirement), attached to the SAME ``MjModel`` /
    ``MjData`` the env steps (via ``SimEnv.mujoco_model_data``), so the window
    shows exactly what the loop drives -- not a copy.

The viewer is PRESENTATION ONLY: it reads the simulation and never drives it.
It injects no actions, mutates no env state, and changes no reply the server
sends -- a client talking to a ``--viewer`` server gets byte-identical replies
to a headless one. It only calls ``viewer.sync()`` to push the current sim
state into the window so the arm visibly moves as inference drives it.

Window-close behavior (chosen; ADR-0013 leaves it to the implementation): when
the human CLOSES THE WINDOW, the server SHUTS DOWN CLEANLY -- the background
serve loop is stopped and the process exits. Closing the window is the "I'm
done watching" signal for this dev-time aid; there is no headless-survivor mode.

macOS requires launching under ``mjpython`` (the main-thread launcher) rather
than plain ``python``. To actually SEE the window (this cannot be asserted in a
headless test -- a live window is a true external), a human runs::

    uv run mjpython -m sim_server --viewer

(optionally ``--env-id``/``--address``/``--seed`` as for the headless server).
On Linux a plain ``python -m sim_server --viewer`` window works too; headless
``rgb_array`` stays the portable default the loop and every test use.
"""

import logging
import threading
import time

import mujoco.viewer

logger = logging.getLogger(__name__)

# How often to sync the window with the sim while it is open. The viewer only
# READS the sim, so this cadence is purely presentational (smoothness of the
# live view), never a driver of the loop.
_SYNC_INTERVAL_S = 1.0 / 60.0


def run_with_viewer(server, env) -> None:
    """Serve on a background thread while a passive MuJoCo window holds this
    (main) thread; return when the window is closed, having stopped the server.

    ``server`` is a bound-capable :class:`~sim_server.server.SimServer`; ``env``
    is the injected sim env exposing ``mujoco_model_data()``. This function must
    be called on the process main thread (MuJoCo/macOS requirement).
    """
    model, data = env.mujoco_model_data()
    data_lock = env.data_lock

    serve_thread = threading.Thread(
        target=server.serve_forever, name="sim-serve", daemon=True
    )
    serve_thread.start()
    server.wait_until_ready(timeout=30.0)
    logger.info("sim server serving in background on %s", server.bound_address)

    logger.info("opening live MuJoCo viewer (close the window to shut down)")
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # Presentation loop: reflect the sim as the background loop drives
            # it. sync() pushes the CURRENT model/data into the window; it does
            # not step or mutate anything. MjData is not thread-safe, and the
            # serve thread mutates it in reset/step/render -- so hold the env's
            # data_lock (the SAME lock those methods take) around sync(), or the
            # main-thread read collides with a background step and MuJoCo aborts
            # ("mj_copyDataVisual: stack is in use").
            while viewer.is_running():
                with data_lock:
                    viewer.sync()
                time.sleep(_SYNC_INTERVAL_S)
    finally:
        # Window closed (or the viewer raised) -> clean shutdown: stop the
        # background serve loop and let it drain.
        logger.info("viewer closed, shutting down sim server")
        server.stop()
        serve_thread.join(timeout=5.0)


__all__ = ["run_with_viewer"]
