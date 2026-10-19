"""Activity Performance Monitor: Streamlit web app."""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from activity_monitor.model import ensure_model, load_metrics, MODEL_DIR, predict_activities
from activity_monitor.scoring import summarize
from activity_monitor.ui import apply_theme, hero, sidebar_brand

SAMPLE_CSV = Path(__file__).parent / "data" / "sample_session.csv"
PAGES = ["Home", "Upload & Detect", "Dashboard", "Model Performance"]

st.set_page_config(page_title="Activity Performance Monitor", layout="wide")
apply_theme()


@st.cache_resource
def get_model():
    return ensure_model()


def page_home():
    hero("Know how active you really are",
         "Upload movement readings from a phone or smartwatch. The app recognises what you were doing "
         "(walking, climbing stairs, sitting, standing, laying) and gives you a performance score out of 100.")
    st.subheader("How to use it")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**1. Upload**  \nChoose your CSV file, or tick *Use sample data*.")
    c2.markdown("**2. Detect**  \nPress *Detect activities*. The model labels every reading.")
    c3.markdown("**3. Review**  \nOpen *Dashboard* for your score, charts and activity log.")
    st.info("Tip: no data yet? Go to **Upload & Detect** and tick *Use sample data* to try the app straight away.")


def page_upload():
    st.title("Upload & Detect")
    left, right = st.columns(2)
    with left:
        up = st.file_uploader("1. Upload a CSV of sensor readings", type="csv")
        use_sample = st.checkbox("Use sample data instead", value=False)
        goal = st.number_input("Daily active-time goal (minutes)", min_value=5, max_value=300, value=60)
        window = st.number_input("Seconds of movement covered by each row", min_value=0.5, max_value=60.0, value=2.56)
        go = st.button("2. Detect activities")
    df = None
    if up is not None:
        df = pd.read_csv(up)
    elif use_sample:
        if not SAMPLE_CSV.exists():
            from activity_monitor.data import generate_sample_session
            SAMPLE_CSV.parent.mkdir(exist_ok=True)
            generate_sample_session().to_csv(SAMPLE_CSV, index=False)
        df = pd.read_csv(SAMPLE_CSV)
    with right:
        st.markdown("**Data preview**")
        if df is None:
            st.caption("Upload a file or tick *Use sample data* to see a preview.")
        else:
            st.caption(f"{len(df):,} rows, {df.shape[1]} columns")
            st.dataframe(df.head())
    if go:
        if df is None:
            st.warning("Please upload a CSV file or tick *Use sample data* first.")
            return
        model, metrics = get_model(), load_metrics()
        try:
            df = df.copy()
            df["activity"] = predict_activities(model, df, metrics["features"])
        except ValueError as e:
            st.error(str(e))
            return
        if "user" not in df.columns:
            df["user"] = "User 01"
        if "date" not in df.columns:  # no dates: split the rows evenly into 7 days
            df["date"] = [f"Day {i + 1}" for i in np.repeat(np.arange(7), int(np.ceil(len(df) / 7)))[: len(df)]]
        df["date"] = df["date"].astype(str)
        st.session_state["results"] = {"df": df, "goal": goal, "window": window}
        st.success(f"Detected {len(df):,} activity windows. Open the **Dashboard** page to see your results.")


def page_dashboard():
    st.title("Dashboard")
    res = st.session_state.get("results")
    if not res:
        st.info("No results yet. Go to **Upload & Detect**, load data and press *Detect activities*.")
        return
    df = res["df"]
    users = sorted(df["user"].unique())
    user = st.selectbox("User", users) if len(users) > 1 else users[0]
    s = summarize(df[df["user"] == user], window_seconds=res["window"], goal=res["goal"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total activities", f"{s['windows']:,}")
    c2.metric("Avg. duration per day", f"{s['avg_daily_minutes']:.0f} min")
    c3.metric("Performance score", f"{s['score']:.1f} / 100")
    c4.metric("Consistency", f"{s['consistency'] * 100:.0f}%")

    a, b = st.columns(2)
    with a:
        st.subheader("Time per activity (minutes)")
        st.bar_chart(s["minutes_by_activity"])
    with b:
        st.subheader("Daily performance score")
        st.line_chart(s["daily"].set_index("date")["score"])
    st.subheader("Activity log (minutes per day)")
    st.dataframe(s["per_day"])
    st.download_button("Download detected activities (CSV)", df.to_csv(index=False).encode(), "detected_activities.csv", "text/csv")


def page_model():
    st.title("Model Performance")
    get_model()
    m = load_metrics()
    best = next(r for r in m["models"] if r["name"] == m["best_model"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best model", m["best_model"])
    c2.metric("Test accuracy", f"{best['test_accuracy'] * 100:.1f}%")
    c3.metric("Weighted F1", f"{best['test_f1_weighted']:.3f}")
    c4.metric("Cross-validation", f"{best['cv_mean'] * 100:.1f}%")
    st.caption(f"Trained on: {m['data_source']}  |  {m['n_train']:,} training rows, {m['n_test']:,} test rows")
    left, right = st.columns(2)
    with left:
        st.subheader("Model comparison")
        table = pd.DataFrame(m["models"]).rename(columns={
            "name": "Algorithm", "cv_mean": "CV accuracy", "cv_std": "CV std",
            "test_accuracy": "Test accuracy", "test_f1_weighted": "Weighted F1"})
        st.dataframe(table.round(3), hide_index=True)
    with right:
        st.subheader("Confusion matrix")
        img = MODEL_DIR / "confusion_matrix.png"
        if img.exists():
            st.image(str(img))


sidebar_brand("Activity Performance Monitor")
page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")
{"Home": page_home, "Upload & Detect": page_upload, "Dashboard": page_dashboard, "Model Performance": page_model}[page]()
