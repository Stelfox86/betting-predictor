import streamlit as st
import pandas as pd
import os

PREDICTIONS_PATH = "Data/predictions.csv"
FIXTURES_PATH = "Data/upcoming_fixtures.csv"
RESULTS_PATH = "Data/bet_results.csv"

@st.cache_data
def load_predictions():
    try:
        return pd.read_csv(PREDICTIONS_PATH)
    except Exception:
        return pd.DataFrame()

@st.cache_data
def load_fixtures():
    try:
        return pd.read_csv(FIXTURES_PATH)
    except Exception:
        return pd.DataFrame()

@st.cache_data
def load_results():
    try:
        return pd.read_csv(RESULTS_PATH)
    except Exception:
        return pd.DataFrame()

def reload_predictions():
    st.session_state["predictions"] = load_predictions()

def retrain_model():
    with st.spinner("Retraining model..."):
        result = os.system("python train_model_enhanced.py")  # Or your own script
        if result == 0:
            st.session_state["predictions"] = load_predictions()
            st.sidebar.success("Model retrained and predictions updated!")
        else:
            st.sidebar.error("Model retraining failed.")

# --- SIDEBAR ---
st.sidebar.title("Navigation")
nav = st.sidebar.radio("Go to:", ["Predictions", "Fixtures", "Stats/Profit"])

# --- ACTION BUTTONS ---
st.sidebar.markdown("---")
st.sidebar.button("🔄 Update Predictions", on_click=reload_predictions)
st.sidebar.button("🧠 Retrain Model", on_click=retrain_model)

# --- LOAD DATA ---
predictions = st.session_state.get("predictions", load_predictions())
fixtures = load_fixtures()
results = load_results()

# --- MAIN PAGE LOGIC ---

if nav == "Predictions":
    st.header("Predictions")

    # Put the table close to the sidebar by adjusting column widths
    col1, col2 = st.columns([1, 2])
    with col1:
        week_filter = st.selectbox("Select Match Week:", options=list(range(1, 40)), index=0)
    with col2:
        # (Optional) Filter by week if you have a week/date logic
        if not predictions.empty and "Date" in predictions.columns:
            predictions["Date"] = pd.to_datetime(predictions["Date"], dayfirst=True, errors="coerce")
            start_date = predictions["Date"].min() + pd.to_timedelta((week_filter - 1) * 7, unit='d')
            end_date = start_date + pd.Timedelta(days=6)
            preds_week = predictions[(predictions["Date"] >= start_date) & (predictions["Date"] <= end_date)]
        else:
            preds_week = predictions

        display_cols = [col for col in ["Date", "HomeTeam_original", "AwayTeam_original", "Prediction"] if col in preds_week.columns]
        rename_cols = {
            "Date": "Date",
            "HomeTeam_original": "Home Team",
            "AwayTeam_original": "Away Team",
            "Prediction": "Prediction"
        }
        if not preds_week.empty:
            st.dataframe(preds_week[display_cols].rename(columns=rename_cols), use_container_width=True)
        else:
            st.info("No predictions available. Run the prediction script and reload.")

elif nav == "Fixtures":
    st.header("Fixtures")

    # Table close to sidebar
    col1, col2 = st.columns([1, 2])
    with col1:
        week_filter = st.selectbox("Select Match Week:", options=list(range(1, 40)), index=0, key="fixtures_week")
    with col2:
        if not fixtures.empty and "Date" in fixtures.columns:
            fixtures["Date"] = pd.to_datetime(fixtures["Date"], dayfirst=True, errors="coerce")
            start_date = fixtures["Date"].min() + pd.to_timedelta((week_filter - 1) * 7, unit='d')
            end_date = start_date + pd.Timedelta(days=6)
            fixtures_week = fixtures[(fixtures["Date"] >= start_date) & (fixtures["Date"] <= end_date)]
        else:
            fixtures_week = fixtures

        display_cols = [col for col in ["Date", "HomeTeam_original", "AwayTeam_original"] if col in fixtures_week.columns]
        rename_cols = {
            "Date": "Date",
            "HomeTeam_original": "Home Team",
            "AwayTeam_original": "Away Team"
        }
        if not fixtures_week.empty:
            st.dataframe(fixtures_week[display_cols].rename(columns=rename_cols), use_container_width=True)
        else:
            st.info("No fixtures for this week.")

elif nav == "Stats/Profit":
    st.header("📈 Stats & Profit Tracker")

    # Basic: Show profit info if results available
    if results.empty or not all(col in results.columns for col in ["Prediction", "Actual", "Bet_Odds", "Correct"]):
        st.warning("Results file missing required columns ('Prediction', 'Actual', 'Bet_Odds', 'Correct').")
        st.info("Tip: Update 'bet_results.csv' after each matchday for live profit tracking.")
    else:
        # Accuracy
        accuracy = (results["Prediction"] == results["Actual"]).mean()
        st.metric("Prediction Accuracy", f"{accuracy:.1%}")
        # Profit calc (simple)
        profit = (results["Correct"] * (results["Bet_Odds"] - 1) - (1 - results["Correct"])).sum()
        st.metric("Total Profit (1 unit bets)", f"{profit:.2f}")

        st.dataframe(results, use_container_width=True)























