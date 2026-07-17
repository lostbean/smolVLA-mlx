"""CLI entry point for the sim ZeroMQ server: constructs one SO-101
pick-and-place MuJoCo gym env once, then serves reset/step/render indefinitely.

Usage:

    uv run python -m sim_server
    uv run python -m sim_server --env-id MuJoCoPickAndPlace-v1 --address tcp://*:5556

To WATCH the arm move in a live 3D window, add ``--viewer`` (off by default,
presentation only). On macOS this must launch under ``mjpython`` (the
main-thread launcher MuJoCo requires); on Linux plain ``python`` works::

    uv run mjpython -m sim_server --viewer

In viewer mode the ZeroMQ serve loop runs on a background thread and the window
holds the main thread; closing the window shuts the server down cleanly. The
headless default is unchanged. See ADR-0013 and ``sim_server.viewer``.

The env id can also be set via the SIM_ENV_ID environment variable, and the
bind address via SIM_SERVER_ADDRESS -- CLI flags take precedence over either.
Address defaults to tcp://*:5556 (all interfaces, reachable over LAN, not just
localhost -- this runs as a standing service reachable by the Elixir sim node,
per ADR-0012). Note the default port (5556) differs from the infer_action
server's (5555) so both standing services can run on one machine.

The simulator dependency (so101-nexus + mujoco) is imported when the env is
constructed here, BEFORE the socket binds -- a missing simulator fails loud at
startup, not mid-episode (acceptance criterion 5).
"""

import argparse
import logging
import os
import sys

from sim_server.server import DEFAULT_ADDRESS, SimServer

DEFAULT_ENV_ID = "MuJoCoPickAndPlace-v1"


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="sim_server",
        description="ZeroMQ server wrapping a LeRobot/MuJoCo SO-101 gym env.",
    )
    parser.add_argument(
        "--env-id",
        default=os.environ.get("SIM_ENV_ID", DEFAULT_ENV_ID),
        help=(
            f"Gymnasium env id for the SO-101 pick-and-place task (default: "
            f"{DEFAULT_ENV_ID}, from so101-nexus). Overrides the SIM_ENV_ID "
            "environment variable."
        ),
    )
    parser.add_argument(
        "--address",
        default=os.environ.get("SIM_SERVER_ADDRESS", DEFAULT_ADDRESS),
        help=(
            f"ZeroMQ bind address for the REP socket (default: "
            f"{DEFAULT_ADDRESS}). Overrides the SIM_SERVER_ADDRESS "
            "environment variable."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional reset seed for deterministic episodes.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (default: INFO).",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        default=False,
        help=(
            "Open a live MuJoCo 3D window onto the running sim (off by "
            "default). Presentation only -- serves ZeroMQ on a background "
            "thread and holds the main thread for the window. On macOS launch "
            "under mjpython: `uv run mjpython -m sim_server --viewer`. Closing "
            "the window shuts the server down. See ADR-0013."
        ),
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Deferred import: constructing SimEnv pulls in gymnasium + mujoco +
    # so101-nexus, unnecessary weight for --help and argument-parsing errors.
    from sim_server.env import SimEnv

    logger.info("constructing sim env %s ...", args.env_id)
    env = SimEnv(env_id=args.env_id, seed=args.seed)
    logger.info("sim env ready, starting server on %s", args.address)

    server = SimServer(env, address=args.address)
    try:
        if args.viewer:
            # Viewer mode: serve on a background thread; the live MuJoCo window
            # holds the main thread. Deferred import so the default (headless)
            # path never pulls in mujoco.viewer. See ADR-0013.
            from sim_server.viewer import run_with_viewer

            run_with_viewer(server, env)
        else:
            server.serve_forever()
    except KeyboardInterrupt:
        logger.info("interrupted, shutting down")
        server.stop()
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
