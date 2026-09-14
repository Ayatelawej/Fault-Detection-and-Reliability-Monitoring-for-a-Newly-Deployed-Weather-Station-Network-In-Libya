"""Train the final reason heads or apply a frozen release without labels."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model.final_reason_codes import main

if __name__ == "__main__":
    main()
