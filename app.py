
import streamlit as st
import pandas as pd
import subprocess
import os

st.set_page_config(page_title="Betting Predictor Dashboard", layout="wide")
st.title("⚽ Football Betting Predictor")

# Utility functions
def run_script(script_name):
    result = subprocess.run(["python", script_name], capture_output=True, text=True)
    return result.stdout + "\n" + result.stderr

def load_csv(file_name):
    if os.path.exists(file_name):
        return pd.read_csv(file_name)
    return pd.DataFrame()

# Sidebar
with st.sidebar:
    st.header("🔧 Run Tools")

    if st.button("🧠 Train Model"):
        st.success("Training model...")
        output = run_script("main.py")
        st.text(output)

    if st.button("🎯 Predict Fixtures"):
        st.success("Running predictions...")
        output = run_script("predict_fixtures_updated.py")
        st.text(output)

    if st.button("💸 Find Value Bets"):
        st.success("Calculating value bets...")
        output = run_script("value_bets.py")
        st.text(output)

    if st.button("📊 Track Profit"):
        st.success("Tracking results...")
        output = run_script("track_profit.py")
        st.text(output)


# Tabs
tab1, tab2, tab3 = st.tabs(["📋 Predictions", "💸 Value Bets", "📈 Bet Results"])

with tab1:
    st.subheader("Predictions")
    df = load_csv("predicted_fixtures.csv")
    if not df.empty:
        st.dataframe(df)
    else:
        st.info("No predictions available.")

with tab2:
    st.subheader("Value Bets")
    df = load_csv("value_bets.csv")
    if not df.empty:
        st.dataframe(df[df['ValueBet'] == True])
    else:
        st.info("No value bets found.")

with tab3:
    st.subheader("Bet Results & Profit")
    df = load_csv("bet_results.csv")
    if not df.empty:
        st.dataframe(df)
        profit = df['Profit'].sum()
        total = len(df)
        wins = df['BetWon'].sum()
        st.metric("🏆 Total Bets", total)
        st.metric("✅ Wins", wins)
        st.metric("💰 Total Profit", f"£{profit:.2f}")
    else:
        st.info("No bet results found.")
