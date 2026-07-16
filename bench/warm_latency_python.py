"""Warm-latency benchmark for the Python reference SmolVLAModel.infer_action.

Same methodology as the Elixir bench (bench/warm_latency.exs): loads the
real checkpoint once, uses the same e2e fixture image/state/instruction,
warms up once, then reports the median of N warm runs. `infer_action`
calls `mx.eval` on its result before returning, so end-to-end wall-clock
timing is honest (no lazy MLX graph left unevaluated).

Run from the repo root under the project venv:

    uv run python bench/warm_latency_python.py

Env knobs:
    PYBENCH_RUNS=N   number of timed runs (default 9)
    SMOLVLA_CHECKPOINT=/path   override checkpoint dir
"""

import os
import statistics
import time
from pathlib import Path

import numpy as np

from mlx_vlm.models.smolvla.smolvla import SmolVLAModel

RUNS = int(os.environ.get("PYBENCH_RUNS", "9"))

CHECKPOINT = os.environ.get(
    "SMOLVLA_CHECKPOINT",
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--lerobot--smolvla_base/"
        "snapshots/c83c3163b8ca9b7e67c509fffd9121e66cb96205"
    ),
)

FIXTURES = Path(__file__).resolve().parent.parent / "test" / "fixtures"


def main() -> None:
    model = SmolVLAModel.from_pretrained(CHECKPOINT)

    image = np.fromfile(FIXTURES / "e2e_probe_image_f32.bin", dtype=np.float32).reshape(
        224, 224, 3
    )
    state = np.fromfile(FIXTURES / "e2e_probe_state_f32.bin", dtype=np.float32).tolist()
    instruction = (FIXTURES / "e2e_probe_instruction.txt").read_text()

    # Warm-up (first call includes graph build / compile, not
    # representative of steady-state latency).
    model.infer_action(image, state, instruction)

    times_ms = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        model.infer_action(image, state, instruction)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    times_ms.sort()
    median = statistics.median(times_ms)
    print(
        f"[pybench] warm infer_action over {RUNS} runs: "
        f"median={median:.1f}ms min={min(times_ms):.1f}ms "
        f"max={max(times_ms):.1f}ms"
    )
    print("[pybench] all runs (ms): " + ", ".join(f"{t:.1f}" for t in times_ms))


if __name__ == "__main__":
    main()
