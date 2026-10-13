"""Logging configuration."""
import logging


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    # Keep third-party libraries quiet unless debugging.
    for noisy in ("httpx", "httpcore", "urllib3", "huggingface_hub", "chromadb"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
