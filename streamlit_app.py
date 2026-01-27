import streamlit as st
import pandas as pd

# --- Page Configuration ---
st.set_page_config(
    page_title="PostgreSQL Connection Test",
    page_icon="🔗",
    layout="centered",
)

st.title("🐘 PostgreSQL Database Connection Test")

st.write(
    "This app attempts to connect to your PostgreSQL database using the secrets "
    "stored in `.streamlit/secrets.toml`."
)

# --- Database Connection ---
try:
    # Establish the connection using st.connection.
    # Streamlit automatically uses the [connections.postgresql] section of your secrets.
    conn = st.connection("postgresql", type="sql")
    st.success("✅ Successfully connected to PostgreSQL!")

    # --- Test Query ---
    st.subheader("Running a Test Query")
    st.code("SELECT version();", language="sql")

    # Perform a query. `conn.query` returns a Pandas DataFrame.
    # TTL=600 caches the result for 10 minutes.
    df = conn.query("SELECT version();", ttl=600)

    # Display the query result.
    st.dataframe(df)

    # --- Accessing other secrets ---
    st.subheader("Accessing other secrets")
    st.write("You can also access other keys from your secrets file:")
    st.write(f"The first 5 characters of your API key are: `{st.secrets['RETURNGO_API_KEY'][:5]}...`")

except Exception as e:
    st.error(f"🔥 Failed to connect or query the database: {e}")