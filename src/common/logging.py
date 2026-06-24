import logging
import structlog
from pathlib import Path


def setup_logging(level: str = "INFO", log_dir: str = "logs", stage: str = "pipeline") -> structlog.BoundLogger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(Path(log_dir) / f"{stage}.log"),
        ],
        format="%(message)s",
    )

    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
    )

    return structlog.get_logger(stage)
