"""Server runtime configuration and CLI parsing."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ServerConfig:
    """Resolved server configuration used by the runtime."""

    host: str = "0.0.0.0"
    port: int = 8080
    rounds: int = 3
    fraction_fit: float = 1.0
    fraction_evaluate: float = 1.0
    min_fit_clients: int = 2
    min_evaluate_clients: int = 2
    min_available_clients: int = 2
    output_dir: Path = Path("artifacts/server")
    monitor_url: str | None = None
    monitor_timeout_s: float = 2.0
    run_id: str = "local-run"

    @property
    def server_address(self) -> str:
        """Return the host:port address used by Flower server."""
        return f"{self.host}:{self.port}"


def parse_args(argv: list[str] | None = None) -> ServerConfig:
    """Parse CLI args into a validated server config."""
    parser = argparse.ArgumentParser(
        prog="fedforge-server",
        description="FedForge Flower server runtime",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--fraction-fit", type=float, default=1.0)
    parser.add_argument("--fraction-evaluate", type=float, default=1.0)
    parser.add_argument("--min-fit-clients", type=int, default=2)
    parser.add_argument("--min-evaluate-clients", type=int, default=2)
    parser.add_argument("--min-available-clients", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/server"))
    parser.add_argument("--monitor-url", default=None)
    parser.add_argument("--monitor-timeout-s", type=float, default=2.0)
    parser.add_argument("--run-id", default="local-run")
    args = parser.parse_args(argv)

    return ServerConfig(
        host=args.host,
        port=args.port,
        rounds=args.rounds,
        fraction_fit=args.fraction_fit,
        fraction_evaluate=args.fraction_evaluate,
        min_fit_clients=args.min_fit_clients,
        min_evaluate_clients=args.min_evaluate_clients,
        min_available_clients=args.min_available_clients,
        output_dir=args.output_dir,
        monitor_url=args.monitor_url,
        monitor_timeout_s=args.monitor_timeout_s,
        run_id=args.run_id,
    )
