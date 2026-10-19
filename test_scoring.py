import pytest

from activity_monitor.data import generate_sample_session
from activity_monitor.scoring import performance_score, summarize


def test_all_rest_scores_zero():
    assert performance_score({"LAYING": 60}, active_days=0, total_days=7) == 0.0


def test_perfect_score():
    assert performance_score({"WALKING_UPSTAIRS": 60}, goal=60, active_days=7, total_days=7) == 100.0


def test_score_is_bounded():
    s = performance_score({"WALKING": 500, "SITTING": 5}, goal=60, active_days=7, total_days=7)
    assert 0 <= s <= 100


def test_no_minutes_gives_zero():
    assert performance_score({}) == 0.0


def test_unknown_activity_rejected():
    with pytest.raises(ValueError):
        performance_score({"DANCING": 10})


def test_invalid_goal_rejected():
    with pytest.raises(ValueError):
        performance_score({"WALKING": 10}, goal=0)


def test_summarize_sample_session():
    df = generate_sample_session()
    df["activity"] = "WALKING"
    s = summarize(df, window_seconds=2.56, goal=60)
    assert s["days"] == 7
    assert s["windows"] == len(df)
    assert 0 <= s["score"] <= 100
    assert len(s["daily"]) == 7
