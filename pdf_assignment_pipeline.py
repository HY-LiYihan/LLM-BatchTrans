from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
VENDOR_DIR = PROJECT_ROOT / ".vendor"
if VENDOR_DIR.exists():
    sys.path.insert(0, str(VENDOR_DIR))

from llm_batchtrans.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
