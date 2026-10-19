import tempfile

import pandas as pd

from activity_monitor.data import FEATURES, generate_dataset, generate_sample_session
from activity_monitor.model import load_metrics, load_model, predict_activities, train_and_evaluate


def test_dataset_shape():
    df = generate_dataset(n_per_class=50)
    assert df.shape == (300, len(FEATURES) + 1)
    assert df["activity"].nunique() == 6


def test_train_save_load_predict():
    with tempfile.TemporaryDirectory() as d:
        m = train_and_evaluate(generate_dataset(n_per_class=120), out_dir=d)
        assert len(m["models"]) == 5
        best = next(r for r in m["models"] if r["name"] == m["best_model"])
        assert best["test_accuracy"] > 0.7
        model = load_model(d)
        sample = generate_sample_session(windows_per_day=20)
        pred = predict_activities(model, sample, load_metrics(d)["features"])
        assert len(pred) == len(sample)


def test_missing_columns_give_clear_error():
    with tempfile.TemporaryDirectory() as d:
        train_and_evaluate(generate_dataset(n_per_class=60), out_dir=d)
        model = load_model(d)
        try:
            predict_activities(model, pd.DataFrame({"a": [1]}), FEATURES)
        except ValueError as e:
            assert "missing" in str(e)
        else:
            raise AssertionError("expected ValueError")
