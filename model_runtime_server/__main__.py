"""CLI entry point for the infer_action ZeroMQ server: loads a real SmolVLA
checkpoint once, then serves requests indefinitely.

Usage:

    uv run python -m model_runtime_server --checkpoint lerobot/smolvla_base
    uv run python -m model_runtime_server --checkpoint /path/to/local/checkpoint --address tcp://*:5555

The checkpoint path can also be set via the SMOLVLA_CHECKPOINT environment
variable, and the bind address via SMOLVLA_SERVER_ADDRESS -- CLI flags take
precedence over either. Address defaults to tcp://*:5555 (all interfaces,
reachable over LAN, not just localhost -- this is designed to run as a
standing service reachable by an Elixir cluster node elsewhere on the
network, per ADR-0003).
"""

import argparse
import logging
import os
import sys

from model_runtime_server.server import DEFAULT_ADDRESS, InferActionServer


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="model_runtime_server",
        description="ZeroMQ server wrapping SmolVLAModel.infer_action().",
    )
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("SMOLVLA_CHECKPOINT"),
        help=(
            "Path to a local SmolVLA checkpoint directory, or a Hugging "
            "Face Hub repo id (e.g. lerobot/smolvla_base). Overrides the "
            "SMOLVLA_CHECKPOINT environment variable."
        ),
    )
    parser.add_argument(
        "--address",
        default=os.environ.get("SMOLVLA_SERVER_ADDRESS", DEFAULT_ADDRESS),
        help=(
            f"ZeroMQ bind address for the REP socket (default: "
            f"{DEFAULT_ADDRESS}). Overrides the SMOLVLA_SERVER_ADDRESS "
            "environment variable."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (default: INFO).",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    if not args.checkpoint:
        logger.error(
            "no checkpoint given -- pass --checkpoint or set SMOLVLA_CHECKPOINT"
        )
        return 2

    # Deferred import: loading mlx_vlm.models.smolvla pulls in mlx/transformers,
    # which is unnecessary weight for --help and argument-parsing errors.
    from mlx_vlm.models.smolvla import SmolVLAModel

    logger.info("loading SmolVLA checkpoint from %s ...", args.checkpoint)
    model = SmolVLAModel.from_pretrained(args.checkpoint)
    logger.info("checkpoint loaded, starting server on %s", args.address)

    server = InferActionServer(model, address=args.address)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("interrupted, shutting down")
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
