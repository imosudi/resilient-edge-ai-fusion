import logging
from pathlib import Path

# Ensure logs directory exists
Path("../logs").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename="../logs/system.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)