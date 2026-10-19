"""Create data/sample_session.csv, a week of demo readings for the web app."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from activity_monitor.data import generate_sample_session

if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "sample_session.csv"
    generate_sample_session().to_csv(out, index=False)
    print(f"Wrote {out}")
