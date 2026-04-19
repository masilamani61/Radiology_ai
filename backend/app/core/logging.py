import logging
import sys
from pathlib import Path

def setup_logging():
    Path("logs").mkdir(exist_ok=True)
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/radiologyai.log", mode="a"),
        ],
    )
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
