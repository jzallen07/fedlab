"""Flower server runtime entrypoint."""

from __future__ import annotations

import logging

from flwr.server import ServerConfig as FlowerServerConfig
from flwr.server import start_server

from src.monitor.emitter import MonitorEmitter
from src.server.artifacts import ArtifactWriter
from src.server.config import parse_args
from src.server.strategy import build_strategy

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Start the Flower server with configured strategy and instrumentation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = parse_args()
    writer = ArtifactWriter(cfg.output_dir)
    writer.write_config(cfg)
    monitor = MonitorEmitter(
        base_url=cfg.monitor_url,
        timeout_s=cfg.monitor_timeout_s,
    )
    monitor.emit_event(
        event_type="node_heartbeat",
        run_id=cfg.run_id,
        node_id="server",
        role="server",
        status="ready",
        details={"server_address": cfg.server_address},
    )
    strategy = build_strategy(config=cfg, artifact_writer=writer, monitor=monitor)
    LOGGER.info("Starting Flower server on %s", cfg.server_address)
    try:
        start_server(
            server_address=cfg.server_address,
            config=FlowerServerConfig(num_rounds=cfg.rounds),
            strategy=strategy,
        )
    except Exception as exc:
        monitor.emit_event(
            event_type="node_error",
            run_id=cfg.run_id,
            node_id="server",
            role="server",
            status="error",
            details={"stage": "server.start", "error": str(exc)},
        )
        raise


if __name__ == "__main__":
    main()
