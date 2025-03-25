import sqlite3

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import duckdb_importer as di
from data import get_data
from utils import correlation_matrix

db_name = "portfolio.db"


# Initialize SQLite database
def init_db():
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS holdings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            broker TEXT NOT NULL,
            asset TEXT NOT NULL,
            amount REAL NOT NULL
        )
    """)
    conn.commit()
    conn.close()


init_db()


@st.cache_data
def get_tickers_w_desc():
    df: pd.DataFrame = get_data(
        query=f"select distinct ticker || ' - ' || description as asset from {di.perf_tbl}"
    )["asset"].values.tolist()
    return df


# Function to fetch all holdings from the database
def fetch_holdings():
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    c.execute("SELECT id, broker, asset, amount FROM holdings")
    data = c.fetchall()
    conn.close()
    return data


# Function to add a holding
def add_holding(broker, asset, amount):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    c.execute(
        "INSERT INTO holdings (broker, asset, amount) VALUES (?, ?, ?)",
        (broker, asset, amount),
    )
    conn.commit()
    conn.close()


# Function to update a holding
def update_holding(id, broker, asset, amount):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    c.execute(
        "UPDATE holdings SET broker=?, asset=?, amount=? WHERE id=?",
        (broker, asset, amount, id),
    )
    conn.commit()
    conn.close()


# Function to delete a holding
def delete_holding(id):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    c.execute("DELETE FROM holdings WHERE id=?", (id,))
    conn.commit()
    conn.close()


# Streamlit app
st.title("Portfolio Manager")

# Sidebar for adding/editing holdings
st.sidebar.header("Add/Edit Holdings")
broker = st.sidebar.selectbox(
    "Broker Name", options=["Fidelity", "HL", "IBKR", "Other"]
)


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

        if edited_rows or added_rows or deleted_rows:
            st.success("Holdings updated successfully!")

    # Check for changes in the edited DataFrame
    if not edited_df.equals(holdings_df):
        # Update or delete holdings based on changes
        for index, row in edited_df.iterrows():
            if index < len(holdings):
                # Update existing holdings
                update_holding(index + 1, row["Broker"], row["Asset"], row["Amount"])
            else:
                # Add new holdings
                add_holding(row["Broker"], row["Asset"], row["Amount"])
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
