import streamlit as st
import pandas as pd
import os
import warnings

# Make the app use the full width
st.set_page_config(layout="wide")

# ---- Silence pandas warning about date parsing
warnings.filterwarnings("ignore", category=UserWarning, message="Parsing dates in .* format")

PREDICTIONS_PATH = "Data/predictions.csv"
FIXTURES_PATH    = "Data/upcoming_fixtures.csv"

@st.cache_data
def try_read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

# Button logic (to reload predictions or retrain)
def reload_predictions():
    st.session_state["predictions"] = try_read_csv(PREDICTIONS_PATH)

def retrain_model():
    result = os.system("python train_model_enhanced.py")
    if result == 0:
        reload_predictions()
        st.success("Model retrained and predictions reloaded.")
    else:
        st.error("Retraining failed! Check your script output.")

# --- SIDEBAR ---
st.sidebar.title("Navigation")
week_filter = st.sidebar.selectbox("Select Match Week:", options=list(range(1, 40)), index=0)
st.sidebar.button("Reload Predictions", on_click=reload_predictions)
st.sidebar.button("Retrain Model", on_click=retrain_model)

# --- LOAD DATA ---
predictions = st.session_state.get("predictions", try_read_csv(PREDICTIONS_PATH))
fixtures = try_read_csv(FIXTURES_PATH)

# Convert Date columns
if "Date" in predictions:
    predictions["Date"] = pd.to_datetime(predictions["Date"], dayfirst=True, errors="coerce")
if "Date" in fixtures:
    fixtures["Date"] = pd.to_datetime(fixtures["Date"], dayfirst=True, errors="coerce")

# Week filtering (assume one week = 7 days, starts from first fixture)
if not fixtures.empty and "Date" in fixtures:
    start_date = fixtures["Date"].min() + pd.to_timedelta((week_filter - 1) * 7, unit='d')
    end_date   = start_date + pd.Timedelta(days=6)
    fixtures_week = fixtures[(fixtures["Date"] >= start_date) & (fixtures["Date"] <= end_date)]
else:
    fixtures_week = fixtures

# --- TABS (Predictions & Fixtures) ---
tabs = st.tabs(["🗂️ Predictions", "📅 Fixtures"])

with tabs[0]:
    st.header(f"Predictions for Week {week_filter}")
    if not predictions.empty:
        # Filter predictions for selected week
        if "Date" in predictions:
            preds_week = predictions[(predictions["Date"] >= start_date) & (predictions["Date"] <= end_date)]
        else:
            preds_week = predictions

        cols_map = {
            "Date": "Date",
            "HomeTeam_original": "Home Team",
            "AwayTeam_original": "Away Team",
            "Prediction": "Prediction"
        }
        display_cols = [c for c in cols_map if c in preds_week.columns]
        df_disp = preds_week[display_cols].rename(columns=cols_map)
        st.dataframe(df_disp, use_container_width=True)
    else:
        st.warning("No predictions available. Please run your prediction script first.")

with tabs[1]:
    st.header(f"Upcoming Fixtures for Week {week_filter}")
    if not fixtures_week.empty:
        cols_map = {
            "Date": "Date",
            "HomeTeam_original": "Home Team",
            "AwayTeam_original": "Away Team"
        }
        display_cols = [c for c in cols_map if c in fixtures_week.columns]
        df_fix = fixtures_week[display_cols].rename(columns=cols_map)
        st.dataframe(df_fix, use_container_width=True)
    else:
        st.warning("No fixtures available for this week.")





















