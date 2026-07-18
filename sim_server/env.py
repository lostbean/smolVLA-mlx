"""The LeRobot/MuJoCo gym-environment wrapper the sim server drives.

This module owns the single seam to the external physics engine: it wraps
ONE Gymnasium environment of the SO-101 arm doing cube pick-and-place -- the
robot family the ``lerobot/svla_so101_pickplace`` checkpoint was trained on --
and exposes exactly the three operations the sim server answers
(``reset`` / ``step`` / ``render``), each returning a plain-Python payload
(image bytes + shape, and the arm's 6-DoF proprioceptive state as a list of
floats) that the ZeroMQ layer msgpack-encodes verbatim.

Env package
-----------
The env is `so101-nexus <https://pypi.org/project/so101-nexus/>`_'s
``MuJoCoPickAndPlace-v1`` -- a Gymnasium-native, pip-installable MuJoCo
simulation of the SO-101 arm (six actuators: shoulder_pan, shoulder_lift,
elbow_flex, wrist_flex, wrist_roll, gripper) picking a cube and placing it at a
target. It is declared as a hard dependency in pyproject.toml so a missing
simulator fails loud at import time (see ``SimEnv.__init__``), never
mid-episode.

Wire shapes (the mapping this wrapper documents)
------------------------------------------------
- **image**: the env's ``render()`` frame -- an ``(H, W, 3)`` uint8 RGB
  array. Returned as raw bytes (C-contiguous) plus its ``[H, W, 3]`` shape.
- **state**: the arm's proprioceptive state -- the SO-101's SIX controlled
  joint positions (the env's own ``_get_current_qpos()``), as a list of six
  floats. This is exactly ``max_state_dim`` for ``lerobot/smolvla_base`` (6),
  so it never exceeds the checkpoint bound the infer_action port enforces.
  Note the env's flat ``observation_space`` (a 30-vector of privileged sim
  state) is deliberately NOT what we return: SmolVLA's observation.state is the
  robot's joint proprioception, so we extract precisely those six joints.
- **action**: the env's ``action_space`` -- a length-six ``Box`` of joint
  targets. ``step`` accepts a list of six floats. The action expert of SmolVLA
  emits a wider (32-dim) action chunk; the demo's sim-env adapter is
  responsible for slicing the leading six actuated dims before calling
  ``step`` -- this server validates and forwards exactly what it is given and
  fails loud on a wrong-length action rather than silently padding/truncating.
"""

import logging
import threading

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_ENV_ID = "MuJoCoPickAndPlace-v1"

# The SO-101 arm has exactly six controlled joints; this is the checkpoint's
# max_state_dim and the length of the env's action space. Held as a named
# constant so the fail-loud action-length check reads intentionally.
SO101_DOF = 6


class SimEnv:
    """Owns one SO-101 pick-and-place MuJoCo gym env and exposes the three
    operations the sim server serves.

    The env is constructed eagerly in ``__init__`` (fail-loud at startup: a
    missing ``so101-nexus``/``mujoco`` install raises here, before the server
    binds its socket, never mid-episode). ``render_mode="rgb_array"`` so
    ``render()`` yields a frame with no display attached -- the server runs
    headless.
    """

    def __init__(self, env_id: str = DEFAULT_ENV_ID, seed: int | None = None):
        try:
            import gymnasium as gym

            # Import registers so101-nexus's MuJoCo env ids (side-effecting).
            import so101_nexus.mujoco  # noqa: F401
        except ImportError as exc:  # pragma: no cover - exercised via startup
            raise RuntimeError(
                "the SO-101 MuJoCo simulator is not installed -- "
                "'so101-nexus' (and its 'mujoco' dependency) must be present. "
                "It is declared in pyproject.toml; run `uv sync`. "
                f"Original import error: {exc}"
            ) from exc

        self._env_id = env_id
        self._seed = seed
        self._env = gym.make(env_id, render_mode="rgb_array")
        self._closed = False
        # MuJoCo's MjData is NOT thread-safe. In viewer mode the serve loop runs
        # on a background thread (reset/step/render mutate MjData) while the live
        # viewer syncs MjData from the main thread -- concurrent access aborts
        # with "mj_copyDataVisual: stack is in use". This lock serializes every
        # MjData access; the viewer acquires the SAME lock around sync() (see
        # data_lock / sim_server.viewer). Headless mode holds an uncontended
        # lock -- negligible cost, single behavior for both paths.
        self._data_lock = threading.RLock()
        logger.info("sim env %s constructed (render_mode=rgb_array)", env_id)

    @property
    def data_lock(self):
        """The lock serializing all MjData access. Held internally by
        reset/step/render; the live viewer must hold it around sync() so the
        main-thread render never reads MjData mid-mutation on the serve thread.
        """
        return self._data_lock

    # ------------------------------------------------------------------
    # The three operations, each returning a plain-Python payload.
    # ------------------------------------------------------------------
    def reset(self) -> dict:
        """Start a fresh episode; return the initial observation payload."""
        # Validate before taking the lock so a bad call never blocks the viewer.
        with self._data_lock:
            self._env.reset(seed=self._seed)
            return self._observation_payload()

    def step(self, action) -> dict:
        """Advance the simulation one step with ``action`` (a length-six
        sequence of joint targets); return the resulting observation payload.

        Fails loud on a wrong-length or non-numeric action -- the caller gets
        an explicit error over the wire, never a silent no-op or a fabricated
        frame.
        """
        act = self._validate_action(action)
        # The lock spans the mutation AND the payload read: step advances
        # MjData, _observation_payload reads it -- the viewer must not sync
        # between the two, or it renders a half-updated MjData.
        with self._data_lock:
            self._env.step(act)
            return self._observation_payload()

    def render(self) -> dict:
        """Return the current rendered frame payload WITHOUT advancing."""
        with self._data_lock:
            image = self._render_frame()
            return self._image_fields(image)

    def mujoco_model_data(self):
        """Return ``(model, data)`` -- the env's actual MuJoCo ``MjModel`` and
        ``MjData`` instances, the exact objects ``reset``/``step`` advance.

        Read-only accessor for the optional live viewer (see
        ``sim_server.viewer``): ``mujoco.viewer.launch_passive(model, data)``
        attaches a passive 3D window to *these* objects, so the window shows
        precisely what the serve loop drives -- not a copy. Nothing here mutates
        the sim; the viewer reads it and never drives it (ADR-0013).
        """
        unwrapped = self._env.unwrapped
        return unwrapped.model, unwrapped.data

    def close(self) -> None:
        if not self._closed:
            self._env.close()
            self._closed = True

    # ------------------------------------------------------------------
    # Internals.
    # ------------------------------------------------------------------
    def _validate_action(self, action) -> np.ndarray:
        if not isinstance(action, (list, tuple, np.ndarray)):
            raise ValueError(
                f"'action' must be an array of numbers, got {type(action).__name__}"
            )
        if not all(isinstance(x, (int, float)) for x in action):
            raise ValueError("'action' must be an array of numbers")
        if len(action) != SO101_DOF:
            raise ValueError(
                f"'action' has length {len(action)}, expected {SO101_DOF} "
                f"(the SO-101 arm's actuated joint count)"
            )
        return np.asarray(action, dtype=np.float32)

    def _proprio_state(self) -> list:
        """The SO-101's six controlled joint positions -- the arm's
        proprioceptive state, exactly max_state_dim for the checkpoint."""
        qpos = self._env.unwrapped._get_current_qpos()
        state = np.asarray(qpos, dtype=np.float32).reshape(-1)
        if state.shape[0] != SO101_DOF:  # pragma: no cover - guards env drift
            raise RuntimeError(
                f"env proprioceptive state has {state.shape[0]} dims, "
                f"expected {SO101_DOF}"
            )
        return [float(x) for x in state]

    def _render_frame(self) -> np.ndarray:
        frame = self._env.render()
        if frame is None:
            raise RuntimeError(
                "env.render() returned None -- the env was not created with "
                "render_mode='rgb_array'"
            )
        return np.ascontiguousarray(np.asarray(frame, dtype=np.uint8))

    def _image_fields(self, image: np.ndarray) -> dict:
        if image.ndim != 3 or image.shape[2] != 3:
            raise RuntimeError(
                f"expected an (H, W, 3) RGB frame, got shape {image.shape}"
            )
        h, w, c = image.shape
        return {"image": image.tobytes(), "image_shape": [int(h), int(w), int(c)]}

    def _observation_payload(self) -> dict:
        image = self._render_frame()
        payload = self._image_fields(image)
        payload["state"] = self._proprio_state()
        return payload


__all__ = ["SimEnv", "DEFAULT_ENV_ID", "SO101_DOF"]
