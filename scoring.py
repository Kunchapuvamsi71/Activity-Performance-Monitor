"""Performance scoring: turns detected activities into a 0-100 score."""
import pandas as pd

# How demanding each activity is (0 = rest, 1 = hardest).
INTENSITY = {
    "WALKING": 0.6,
    "WALKING_UPSTAIRS": 1.0,
    "WALKING_DOWNSTAIRS": 0.8,
    "SITTING": 0.1,
    "STANDING": 0.2,
    "LAYING": 0.0,
}
ACTIVE_THRESHOLD = 0.6            # activities at or above this intensity count as "active time"
MIN_ACTIVE_MINUTES_PER_DAY = 10   # a day counts as active if it has at least this many active minutes
WEIGHTS = {"duration": 0.4, "intensity": 0.3, "consistency": 0.3}


def active_minutes(minutes):
    """Minutes spent in active (intensity >= 0.6) activities."""
    return sum(m for a, m in minutes.items() if INTENSITY[a] >= ACTIVE_THRESHOLD)


def performance_score(minutes, goal=60, active_days=5, total_days=7):
    """Score from 0 to 100.

    minutes      dict of activity name -> average minutes per day
    goal         daily active-time goal in minutes
    active_days  number of days on which the user was active
    total_days   number of days covered

    score = 100 * (0.4 * duration + 0.3 * intensity + 0.3 * consistency) where
      duration    = min(active minutes / goal, 1)
      intensity   = time-weighted average intensity of all activity
      consistency = active_days / total_days
    """
    unknown = set(minutes) - set(INTENSITY)
    if unknown:
        raise ValueError(f"Unknown activity: {sorted(unknown)}")
    if goal <= 0 or total_days <= 0:
        raise ValueError("goal and total_days must be positive")
    total = sum(minutes.values())
    if total <= 0:
        return 0.0
    duration = min(active_minutes(minutes) / goal, 1.0)
    intensity = sum(INTENSITY[a] * m for a, m in minutes.items()) / total
    consistency = min(active_days / total_days, 1.0)
    score = 100 * (WEIGHTS["duration"] * duration + WEIGHTS["intensity"] * intensity + WEIGHTS["consistency"] * consistency)
    return round(score, 1)


def summarize(df, window_seconds=2.56, goal=60):
    """Summarise detected activities.

    df must have the columns 'date' and 'activity' (one row per sensor window).
    Returns a dict with totals, per-activity minutes, a per-day table and the overall score.
    """
    minutes_per_row = window_seconds / 60.0
    per_day = df.groupby(["date", "activity"]).size().unstack(fill_value=0) * minutes_per_row
    for a in INTENSITY:
        if a not in per_day.columns:
            per_day[a] = 0.0
    per_day = per_day[list(INTENSITY)]
    days = len(per_day)

    daily = []
    for date, row in per_day.iterrows():
        m = row.to_dict()
        act = active_minutes(m)
        is_active = act >= MIN_ACTIVE_MINUTES_PER_DAY
        daily.append({
            "date": date,
            "active_minutes": round(act, 1),
            "score": performance_score(m, goal=goal, active_days=1 if is_active else 0, total_days=1),
        })
    daily = pd.DataFrame(daily)

    totals = per_day.sum()
    avg_daily = (totals / days).to_dict()
    active_days = int((daily["active_minutes"] >= MIN_ACTIVE_MINUTES_PER_DAY).sum())
    return {
        "windows": int(len(df)),
        "days": days,
        "total_minutes": float(totals.sum()),
        "avg_daily_minutes": float(totals.sum() / days),
        "minutes_by_activity": totals.round(1),
        "per_day": per_day.round(1),
        "daily": daily,
        "active_days": active_days,
        "consistency": active_days / days,
        "score": performance_score(avg_daily, goal=goal, active_days=active_days, total_days=days),
    }
