"""Launch local server and client processes for distributed FL smoke runs."""

from __future__ import annotations

import argparse
import importlib.util
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local distributed FedForge stack")
    parser.add_argument("--num-clients", type=int, default=3)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--run-id", default="local-run")
    parser.add_argument("--monitor-url", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/server"))
    parser.add_argument("--model-id", default="facebook/deit-tiny-patch16-224")
    parser.add_argument("--train-examples", type=int, default=16)
    parser.add_argument("--eval-examples", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--train-mode",
        default="head_only",
        choices=["head_only", "unfreeze_last_block"],
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "auto"])
    return parser.parse_args(argv)


def _check_client_module() -> None:
    if importlib.util.find_spec("src.client.app") is None:
        raise RuntimeError(
            "src.client.app is not implemented yet. "
            "Engineer 2 chunk B06 must be complete before distributed launcher can run."
        )


def _start_process(command: list[str], env: dict[str, str]) -> subprocess.Popen:
    return subprocess.Popen(command, env=env)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        _check_client_module()
    except RuntimeError as exc:
        print(f"run-local-distributed: {exc}")
        return 1

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"

    server_cmd = [
        sys.executable,
        "-m",
        "src.server.app",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--rounds",
        str(args.rounds),
        "--run-id",
        args.run_id,
        "--output-dir",
        str(args.output_dir),
    ]
    if args.monitor_url:
        server_cmd.extend(["--monitor-url", args.monitor_url])

    procs: list[subprocess.Popen] = []
    try:
        procs.append(_start_process(server_cmd, env=env))
        time.sleep(1.5)
        for idx in range(args.num_clients):
            client_cmd = [
                sys.executable,
                "-m",
                "src.client.app",
                "start",
                "--server-address",
                f"{args.host}:{args.port}",
                "--client-id",
                str(idx),
                "--model-id",
                args.model_id,
                "--train-examples",
                str(args.train_examples),
                "--eval-examples",
                str(args.eval_examples),
                "--image-size",
                str(args.image_size),
                "--train-mode",
                args.train_mode,
                "--device",
                args.device,
                "--run-id",
                args.run_id,
            ]
            if args.monitor_url:
                client_cmd.extend(["--monitor-url", args.monitor_url])
            procs.append(_start_process(client_cmd, env=env))

        return procs[0].wait()
    except KeyboardInterrupt:
        return 130
    finally:
        for proc in reversed(procs):
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
        for proc in reversed(procs):
            if proc.poll() is None:
                proc.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
