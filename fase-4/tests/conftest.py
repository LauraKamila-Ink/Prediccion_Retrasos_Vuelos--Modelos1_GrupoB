import sys
from pathlib import Path

FASE4_DIR = Path(__file__).resolve().parents[1]
if str(FASE4_DIR) not in sys.path:
    sys.path.insert(0, str(FASE4_DIR))
