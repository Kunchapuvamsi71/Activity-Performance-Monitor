# Activity Performance Monitor

A machine learning web app that recognises what a person is doing from movement-sensor readings
(walking, climbing stairs, sitting, standing, laying) and turns the result into a **performance score from 0 to 100**.

Built with Python, scikit-learn and Streamlit.

## What it does

- **Detects activities** from sensor-style data using a classifier chosen from five compared models.
- **Scores performance** from three things: how much of a daily active-time goal was met, how demanding the activity was, and how consistent the person was across days.
- **Shows a dashboard** with key numbers, time per activity, a daily score trend and an activity log, and lets you download the results.
- **Explains the model** with a comparison table and a confusion matrix.

## How to use the website

| Page | What to do |
|---|---|
| **Home** | Read what the app does and the three steps. |
| **Upload & Detect** | Upload a CSV of sensor readings, or tick *Use sample data*. Set your daily active-time goal, then press **Detect activities**. |
| **Dashboard** | See your performance score, consistency, charts and activity log. Download the detected activities as CSV. |
| **Model Performance** | See which model was chosen, its accuracy and F1-score, the comparison of all five models and the confusion matrix. |

**CSV format.** One row per window of movement. It must contain the feature columns listed in `models/metrics.json` under `features`
(for the sample model: acc_mean_x, acc_mean_y, acc_mean_z, acc_std_x, acc_std_y, acc_std_z, acc_mag_mean, acc_mag_std, gyro_std, dominant_freq, energy, jerk_mean). Optional columns: `user` and `date` (`YYYY-MM-DD`). Without `date`, rows are split evenly into 7 days.
See `data/sample_session.csv` for an example.

## The performance score

```
score = 100 x (0.4 x duration + 0.3 x intensity + 0.3 x consistency)

duration    = min(average daily active minutes / goal, 1)
intensity   = time-weighted average intensity of all activity
consistency = active days / days covered   (a day is active with at least 10 active minutes)
```

| Activity | Intensity weight |
|---|---|
| Walking | 0.6 |
| Walking upstairs | 1.0 |
| Walking downstairs | 0.8 |
| Standing | 0.2 |
| Sitting | 0.1 |
| Laying | 0.0 |

Activities with intensity 0.6 or more count as active time. The code is in `activity_monitor/scoring.py`.

## Model results on the included synthetic data

Five models are trained and compared with 5-fold cross-validation; the best one is saved.

| Algorithm | CV accuracy | Test accuracy | Weighted F1 |
|---|---|---|---|
| Logistic Regression | 94.6% | 93.6% | 0.936 |
| k-Nearest Neighbours | 92.5% | 92.2% | 0.922 |
| Decision Tree | 86.8% | 85.0% | 0.851 |
| Support Vector Machine | 94.5% | 93.9% | 0.939 |
| Random Forest | 93.8% | 93.6% | 0.936 |

Best model: **Logistic Regression**. These numbers are from synthetic data (see the note below).

## Project structure

```
app.py                      Streamlit website (4 pages)
train.py                    Train and compare models, save the best
activity_monitor/
  data.py                   Synthetic data generator and UCI HAR loader
  scoring.py                Performance score and summaries
  model.py                  Training, evaluation, saving and loading
  ui.py                     Theme (navy and gold)
data/                       Sample session CSV and the script that makes it
models/                     Saved model, metrics.json, confusion_matrix.png
tests/                      Automated tests
```

## Train again, or use real data

```bash
python train.py                                            # synthetic sample data
python train.py --data uci --uci-path "UCI HAR Dataset"    # real UCI HAR dataset
```

Download the real dataset (Human Activity Recognition Using Smartphones) from https://archive.ics.uci.edu/dataset/240 and unzip it.
It has 561 features, so after training on it your uploaded CSV must contain those 561 columns.

## Quick start

Requires Python 3.9 or newer.

```bash
git clone https://github.com/<your-username>/activity-performance-monitor.git
cd activity-performance-monitor

python -m venv .venv
# Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

The app opens in your browser at http://localhost:8501. A trained model and sample data are already included, so it works straight away.
If the `models/` folder is ever missing, the app trains one automatically on first start.

## Run the tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Deploy it free (Streamlit Community Cloud)

1. Push this repository to GitHub (public).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, choose this repository, branch `main`, and main file `app.py`.
4. Click **Deploy**. You get a public link you can share on your resume.

## Important note about the included data and results

The sample data in this repository is **synthetic** (generated by code) so that the project runs without any download.
The accuracy figures shown in the app and in the table above come from that synthetic data and are for demonstration only.
For real results, train on a real dataset (instructions above) and report those numbers.

## Screenshots

Run the app and save screenshots of each page into `docs/screenshots/`, then link them here.

## Author

Kunchapu Vamsi Krushna, B.Tech Artificial Intelligence and Machine Learning, Dhanalakshmi Srinivasan University, Tiruchirappalli.
Built during an AI/ML training and internship at Averixis Solutions Pvt. Ltd. (April to May 2026).

## License

MIT. See [LICENSE](LICENSE).
