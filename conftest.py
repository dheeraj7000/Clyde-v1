"""Root conftest. Ensures the clyde package is importable from tests/."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
