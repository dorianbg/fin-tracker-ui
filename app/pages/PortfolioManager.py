import sqlite3
from contextlib import contextmanager

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import config
from utils import correlation_matrix, get_tickers_w_desc

db_name = "portfolio.db"


@contextmanager
def get_db():
    conn = sqlite3.connect(db_name)
    try:
        yield conn
    finally:
        conn.close()


# Initialize SQLite database
def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS holdings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                broker TEXT NOT NULL,
                asset TEXT NOT NULL,
                amount REAL NOT NULL
            )
        """)
        conn.commit()


init_db()


def fetch_holdings():
    with get_db() as conn:
        return conn.execute("SELECT id, broker, asset, amount FROM holdings").fetchall()


def add_holding(broker, asset, amount):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO holdings (broker, asset, amount) VALUES (?, ?, ?)",
            (broker, asset, amount),
        )
        conn.commit()


def update_holding(id, broker, asset, amount):
    with get_db() as conn:
        conn.execute(
            "UPDATE holdings SET broker=?, asset=?, amount=? WHERE id=?",
            (broker, asset, amount, id),
        )
        conn.commit()


def delete_holding(id):
    with get_db() as conn:
        conn.execute("DELETE FROM holdings WHERE id=?", (id,))
        conn.commit()


# Streamlit app
st.title("Portfolio Manager")

# Sidebar for adding/editing holdings
st.sidebar.header("Add/Edit Holdings")
broker = st.sidebar.selectbox("Broker Name", options=config.BROKER_OPTIONS)

asset = st.sidebar.selectbox("Asset", options=get_tickers_w_desc())
amount = st.sidebar.number_input("Investment Amount (£)", min_value=1.0)

if st.sidebar.button("Add Holding"):
    if broker and asset and amount > 0:
        add_holding(broker, asset.upper(), amount)
        st.sidebar.success(f"Added {asset} to {broker}")
    else:
        st.sidebar.error("Please fill all fields.")


# Fetch all holdings
holdings = fetch_holdings()

# Display current holdings in an editable table
st.header("Current Holdings")
if holdings:
    # Convert holdings to a DataFrame
    holdings_df = pd.DataFrame(holdings, columns=["ID", "Broker", "Asset", "Amount"])
    # Display editable table (excluding the ID column for editing)
    edited_df = st.data_editor(
        holdings_df.drop(columns=["ID"]),  # Hide ID from editing
        num_rows="dynamic",
        key="holdings_editor",
    )

    # Track changes in the edited DataFrame
    if "holdings_editor" in st.session_state:
        edited_rows = st.session_state.holdings_editor["edited_rows"]
        added_rows = st.session_state.holdings_editor["added_rows"]
        deleted_rows = st.session_state.holdings_editor["deleted_rows"]

        # Update existing holdings
        for index, changes in edited_rows.items():
            id = holdings_df.iloc[index]["ID"]
            broker = changes.get("Broker", holdings_df.iloc[index]["Broker"])
            asset = changes.get("Asset", holdings_df.iloc[index]["Asset"])
            amount = changes.get("Amount", holdings_df.iloc[index]["Amount"])
            update_holding(id, broker, asset, amount)

        # Add new holdings
        for row in added_rows:
            broker = row.get("Broker", "")
            asset = row.get("Asset", "")
            amount = row.get("Amount", 0.0)
            if broker and asset and amount > 0:
                add_holding(broker, asset, amount)

        # Delete holdings
        for index in deleted_rows:
            id = holdings_df.iloc[index]["ID"]
            delete_holding(id)
            holdings_df = pd.DataFrame(
                fetch_holdings(), columns=["ID", "Broker", "Asset", "Amount"]
            )

        if edited_rows or added_rows or deleted_rows:
            st.success("Holdings updated successfully!")

else:
    st.info("No holdings added yet. Use the sidebar to add assets.")

# Merge holdings from all brokers
if holdings:
    consolidated_portfolio = pd.DataFrame(
        holdings, columns=["ID", "Broker", "Asset", "Amount"]
    )
    consolidated_portfolio = (
        consolidated_portfolio.groupby("Asset").sum(numeric_only=True).reset_index()
    )

    # Display consolidated portfolio
    st.header("Consolidated Portfolio")
    st.dataframe(consolidated_portfolio)

    # Calculate asset allocation
    st.header("Asset Allocation")
    fig, ax = plt.subplots()
    ax.pie(
        consolidated_portfolio["Amount"],
        labels=consolidated_portfolio["Asset"],
        autopct="%1.1f%%",
    )
    st.pyplot(fig)

    assets = [x.split(" - ")[0] for x in list(consolidated_portfolio["Asset"])]
    correlation_matrix(assets=assets)

else:
    st.info("No holdings added yet. Use the sidebar to add assets.")
