"""Real-checkpoint, real-process integration check for the infer_action
ZeroMQ server.

Launches ``python -m model_runtime_server`` as a genuinely separate OS
process (not just a separate thread in this test process -- the acceptance
bar per the work-order is a separate PROCESS, proving the deployment shape:
an Elixir cluster node reaching this Mac's standing server over the
network), pointed at the real, publicly available `lerobot/smolvla_base`
checkpoint, and drives one real request/response round trip over a real
ZeroMQ socket using real MessagePack encoding.

This is deliberately NOT part of the fast test gate
(`uv run python -m pytest model_runtime_server`), mirroring the pattern
established by mlx_vlm/tests/integration/test_smolvla_real_checkpoint.py:
it downloads/loads a real ~1.1GB checkpoint and runs a real forward pass,
which takes real wall-clock time. Skipped by default; opt in with
RUN_SMOLVLA_INTEGRATION_CHECK=1:

    RUN_SMOLVLA_INTEGRATION_CHECK=1 uv run python -m pytest \
        model_runtime_server/tests/integration/test_server_real_checkpoint.py -v -s

or point it at an already-downloaded local checkpoint directory:

    RUN_SMOLVLA_INTEGRATION_CHECK=1 \
        SMOLVLA_CHECKPOINT=/path/to/local/smolvla_base \
        uv run python -m pytest \
        model_runtime_server/tests/integration/test_server_real_checkpoint.py -v -s
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

DEFAULT_CHECKPOINT = "lerobot/smolvla_base"
_RUN_FLAG = "RUN_SMOLVLA_INTEGRATION_CHECK"

pytestmark = pytest.mark.skipif(
    os.environ.get(_RUN_FLAG) != "1",
    reason=(
        f"Loads a real ~1.1GB checkpoint and runs a real forward pass in a "
        f"subprocess; set {_RUN_FLAG}=1 to opt in (see module docstring)."
    ),
)


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_real_server_process_serves_real_action_chunk_over_real_socket():
    checkpoint = os.environ.get("SMOLVLA_CHECKPOINT", DEFAULT_CHECKPOINT)
    port = _free_tcp_port()
    address = f"tcp://127.0.0.1:{port}"

    env = dict(os.environ)
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "model_runtime_server",
            "--checkpoint",
            checkpoint,
            "--address",
            address,
            "--log-level",
            "INFO",
        ],
        cwd=os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        ),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        ctx = zmq.Context.instance()
        client = ctx.socket(zmq.REQ)
        client.setsockopt(zmq.LINGER, 0)
        # Real checkpoint load (weights + tokenizer) plus a real forward
        # pass can take a while on first run (cold HF cache); generous
        # timeout for a real subprocess doing real work.
        client.setsockopt(zmq.RCVTIMEO, 180_000)
        client.setsockopt(zmq.SNDTIMEO, 10_000)
        client.connect(address)

        rng = np.random.default_rng(0)
        image = (rng.random((256, 256, 3)) * 255).astype(np.uint8)
        # lerobot/smolvla_base's real observation.state feature is [6].
        state = rng.standard_normal(6).astype(np.float32)
        instruction = "pick up the red block and place it in the bin"

        request = {
            "image": image.tobytes(),
            "image_shape": list(image.shape),
            "state": [float(x) for x in state],
            "instruction": instruction,
        }

        # Retry the connect/send briefly -- the subprocess needs real time
        # to load the checkpoint before its socket is bound and accepting.
        deadline = time.monotonic() + 180.0
        response = None
        last_error = None
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                out = proc.stdout.read() if proc.stdout else ""
                raise AssertionError(
                    f"server process exited early (code {proc.returncode}):\n{out}"
                )
            try:
                client.send(msgpack.packb(request, use_bin_type=True))
                response = msgpack.unpackb(client.recv(), raw=False)
                break
            except zmq.Again as exc:
                last_error = exc
                continue
        else:
            raise AssertionError(f"no response from real server process: {last_error}")

        assert "ok" in response, f"expected ok response, got: {response}"
        action_chunk = response["ok"]["action_chunk"]
        print(
            f"\nreal integration run: request image_shape={image.shape}, "
            f"state={state.tolist()}, instruction={instruction!r}"
        )
        print(
            f"real integration run: action_chunk shape = "
            f"({len(action_chunk)}, {len(action_chunk[0])})"
        )

        assert len(action_chunk) == 50
        assert len(action_chunk[0]) == 32

        action_arr = np.array(action_chunk, dtype=np.float32)
        assert np.isfinite(action_arr).all(), "action chunk contains NaN/Inf"
        assert not np.all(action_arr == 0), "action chunk is degenerate all-zero"
        assert action_arr.std() > 1e-4, "action chunk looks collapsed/constant"

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
