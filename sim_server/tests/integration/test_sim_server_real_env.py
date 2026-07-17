"""Real-MuJoCo, real-process integration check for the sim ZeroMQ server.

Launches ``python -m sim_server`` as a genuinely separate OS process (the
deployment shape: the Elixir sim node reaching this Mac's standing sim server),
pointed at the real so101-nexus MuJoCo SO-101 pick-and-place env, and drives a
canned sequence of ``step`` requests over a real ZeroMQ socket using real
MessagePack encoding. It proves the acceptance bar:

  * reset / step / render each answer over the wire (criteria 1-2);
  * step returns both the rendered frame and the arm's <=6-dim proprioceptive
    state (criterion 3);
  * the canned action sequence visibly MOVES the arm -- consecutive rendered
    frames differ, and the proprioceptive state changes (criterion 1).

Evidence: the rendered frames are written to
sim_server/tests/integration/_artifacts/ (as raw .npy plus, if imageio is
available, a .mp4 and per-step .png) so the motion is inspectable by eye.

This is deliberately NOT part of the fast gate: it loads MuJoCo and runs a real
physics rollout, which takes real wall-clock time. Skipped by default; opt in
with RUN_SMOLVLA_INTEGRATION_CHECK=1:

    RUN_SMOLVLA_INTEGRATION_CHECK=1 uv run python -m pytest \
        sim_server/tests/integration/test_sim_server_real_env.py -v -s
"""

import os
import socket
import subprocess
import sys
import time

import msgpack
import numpy as np
import pytest
import zmq

_RUN_FLAG = "RUN_SMOLVLA_INTEGRATION_CHECK"

pytestmark = pytest.mark.skipif(
    os.environ.get(_RUN_FLAG) != "1",
    reason=(
        f"Loads a real MuJoCo SO-101 env and runs a real physics rollout in a "
        f"subprocess; set {_RUN_FLAG}=1 to opt in (see module docstring)."
    ),
)

_ARTIFACTS = os.path.join(os.path.dirname(__file__), "_artifacts")


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _decode_frame(obs) -> np.ndarray:
    h, w, c = obs["image_shape"]
    return np.frombuffer(bytes(obs["image"]), dtype=np.uint8).reshape(h, w, c)


def _save_evidence(frames):
    os.makedirs(_ARTIFACTS, exist_ok=True)
    arr = np.stack(frames)
    np.save(os.path.join(_ARTIFACTS, "rollout.npy"), arr)
    try:
        import imageio.v2 as imageio

        imageio.mimsave(os.path.join(_ARTIFACTS, "rollout.mp4"), frames, fps=10)
        for i in (0, len(frames) // 2, len(frames) - 1):
            imageio.imwrite(os.path.join(_ARTIFACTS, f"frame_{i:03d}.png"), frames[i])
        print(f"\nsaved rollout.mp4 + sample frames under {_ARTIFACTS}")
    except Exception as exc:  # pragma: no cover - imageio optional
        print(f"\nsaved rollout.npy under {_ARTIFACTS} (no video: {exc})")


def test_real_sim_server_process_drives_moving_arm_over_real_socket():
    port = _free_tcp_port()
    address = f"tcp://127.0.0.1:{port}"

    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    )
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "sim_server",
            "--address",
            address,
            "--seed",
            "0",
            "--log-level",
            "INFO",
        ],
        cwd=repo_root,
        env=dict(os.environ),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        ctx = zmq.Context.instance()
        client = ctx.socket(zmq.REQ)
        client.setsockopt(zmq.LINGER, 0)
        client.setsockopt(zmq.RCVTIMEO, 60_000)
        client.setsockopt(zmq.SNDTIMEO, 10_000)
        client.connect(address)

        def send(request):
            client.send(msgpack.packb(request, use_bin_type=True))
            return msgpack.unpackb(client.recv(), raw=False)

        # The subprocess needs real time to construct the MuJoCo env before its
        # socket is bound and accepting -- retry reset until it answers.
        deadline = time.monotonic() + 60.0
        reset_obs = None
        last_error = None
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                out = proc.stdout.read() if proc.stdout else ""
                raise AssertionError(
                    f"sim server process exited early (code {proc.returncode}):\n{out}"
                )
            try:
                resp = send({"op": "reset"})
                assert "ok" in resp, f"reset failed: {resp}"
                reset_obs = resp["ok"]
                break
            except zmq.Again as exc:
                last_error = exc
                continue
        else:
            raise AssertionError(f"no response from real sim server: {last_error}")

        # --- criterion 3: reset payload carries frame + <=6-dim state ---
        h, w, c = reset_obs["image_shape"]
        assert c == 3 and h > 0 and w > 0, reset_obs["image_shape"]
        assert len(reset_obs["state"]) <= 6, reset_obs["state"]
        assert len(reset_obs["state"]) == 6  # SO-101 arm has six joints
        print(
            f"\nreset: frame {h}x{w}x{c}, state(6-dim) = "
            f"{[round(x, 4) for x in reset_obs['state']]}"
        )

        frames = [_decode_frame(reset_obs)]
        states = [reset_obs["state"]]

        # --- criterion 1: a canned action sequence MOVES the arm ---
        # Drive the six joints with a smooth sinusoidal sweep so the motion is
        # unmistakable, not a single micro-step.
        n_steps = 40
        for t in range(n_steps):
            phase = 2.0 * np.pi * t / n_steps
            action = [
                0.6 * float(np.sin(phase)),
                0.4 * float(np.sin(phase + 0.5)),
                0.4 * float(np.cos(phase)),
                0.3 * float(np.sin(phase + 1.0)),
                0.3 * float(np.cos(phase + 0.5)),
                0.5 * float(np.sin(phase * 2.0)),
            ]
            resp = send({"op": "step", "action": action})
            assert "ok" in resp, f"step {t} failed: {resp}"
            obs = resp["ok"]
            assert len(obs["state"]) == 6
            frames.append(_decode_frame(obs))
            states.append(obs["state"])

        # --- criterion 2: render answers without advancing ---
        render_resp = send({"op": "render"})
        assert "ok" in render_resp
        assert "image" in render_resp["ok"] and "state" not in render_resp["ok"]

        _save_evidence(frames)

        # The arm actually moved: consecutive frames differ, and the joint
        # state swept a meaningful range (not a frozen sim).
        frame_arr = np.stack(frames).astype(np.int16)
        consecutive_diffs = (
            np.abs(np.diff(frame_arr, axis=0)).reshape(len(frames) - 1, -1).mean(axis=1)
        )
        n_moving = int((consecutive_diffs > 0.05).sum())
        print(
            f"frames that changed vs previous: {n_moving}/{len(frames) - 1} "
            f"(mean per-pixel delta max={consecutive_diffs.max():.3f})"
        )
        assert n_moving >= (len(frames) - 1) // 2, (
            "arm did not visibly move: too few frames changed"
        )

        state_arr = np.array(states, dtype=np.float32)
        state_span = state_arr.max(axis=0) - state_arr.min(axis=0)
        print(f"per-joint state span over rollout: {np.round(state_span, 4)}")
        assert state_span.max() > 0.05, (
            "proprioceptive state never changed -- the arm did not move"
        )

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
