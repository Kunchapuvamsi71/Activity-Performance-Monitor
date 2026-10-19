"""Train the activity classifier.

    python train.py                                   # synthetic sample data (default)
    python train.py --data uci --uci-path "UCI HAR Dataset"   # real UCI HAR data
"""
import argparse

from activity_monitor.data import generate_dataset, load_uci_har
from activity_monitor.model import MODEL_DIR, train_and_evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", choices=["sample", "uci"], default="sample")
    p.add_argument("--uci-path", default="UCI HAR Dataset")
    p.add_argument("--out", default=str(MODEL_DIR))
    a = p.parse_args()

    if a.data == "uci":
        df, source = load_uci_har(a.uci_path), "UCI HAR (real data)"
    else:
        df, source = generate_dataset(), "sample (synthetic)"
    m = train_and_evaluate(df, out_dir=a.out, data_source=source)

    print(f"Data source: {source}  |  train rows: {m['n_train']}  test rows: {m['n_test']}")
    print(f"{'Model':26s} {'CV acc':>8s} {'Test acc':>9s} {'Test F1':>8s}")
    for r in m["models"]:
        print(f"{r['name']:26s} {r['cv_mean']:8.3f} {r['test_accuracy']:9.3f} {r['test_f1_weighted']:8.3f}")
    print(f"Best model: {m['best_model']}  (saved to {a.out})")


if __name__ == "__main__":
    main()
