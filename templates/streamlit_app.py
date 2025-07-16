import streamlit as st
import pandas as pd

PREDICTIONS_PATH = "Data/predictions.csv"
FIXTURES_PATH    = "Data/upcoming_fixtures.csv"

@st.cache_data
def try_read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

predictions = try_read_csv(PREDICTIONS_PATH)
fixtures = try_read_csv(FIXTURES_PATH)

st.sidebar.title("Navigation")
tabs = st.tabs(["🗂️ Predictions", "📅 Fixtures"])

# --- Tab 1: Predictions ---
with tabs[0]:
    st.header("Predictions (Upcoming Matches)")
    if not predictions.empty:
        cols_map = {
            "Date": "Date",
            "HomeTeam_original": "Home Team",
            "AwayTeam_original": "Away Team",
            "Prediction": "Prediction"
        }
        display_cols = [c for c in cols_map if c in predictions.columns]
        df_disp = predictions[display_cols].rename(columns=cols_map)
        st.dataframe(df_disp, use_container_width=True)
    else:
        st.warning("No predictions available. Please run your prediction script first.")

# --- Tab 2: Fixtures ---
with tabs[1]:
    st.header("Upcoming Fixtures")
    if not fixtures.empty:
        cols_map = {
            "Date": "Date",
            "HomeTeam_original": "Home Team",
            "AwayTeam_original": "Away Team"
        }
        display_cols = [c for c in cols_map if c in fixtures.columns]
        df_fix = fixtures[display_cols].rename(columns=cols_map)
        st.dataframe(df_fix, use_container_width=True)
    else:
        st.warning("No fixtures available.")















