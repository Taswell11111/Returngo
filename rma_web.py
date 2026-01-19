import streamlit as st
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime, timezone
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="RMA Central", layout="wide", page_icon="📦")

# --- CSS FOR STYLING ---
st.markdown("""
    <style>
    div[data-testid="stMetric"] {
        background-color: #1f2937;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #374151;
    }
    label[data-testid="stMetricLabel"] { color: #9ca3af !important; }
    div[data-testid="stMetricValue"] { color: #ffffff !important; font-size: 24px !important; }
    </style>
    """, unsafe_allow_html=True)

# --- CONFIGURATION ---
MY_API_KEY = os.environ.get("ReturnGo_API")

STORES = [
    {"name": "Diesel", "url": "diesel-dev-south-africa.myshopify.com"},
    {"name": "Hurley", "url": "hurley-dev-south-africa.myshopify.com"},
    {"name": "Jeep Apparel", "url": "jeep-apparel-dev-south-africa.myshopify.com"},
    {"name": "Reebok", "url": "reebok-dev-south-africa.myshopify.com"},
    {"name": "Superdry", "url": "superdry-dev-south-africa.myshopify.com"}
]

# --- SESSION STATE (Memory) ---
if 'data_cache' not in st.session_state:
    st.session_state.data_cache = pd.DataFrame()
if 'logs' not in st.session_state:
    st.session_state.logs = []

# --- FUNCTIONS ---
def log(msg):
    timestamp = datetime.now().strftime('%H:%M:%S')
    st.session_state.logs.append(f"[{timestamp}] {msg}")

@st.cache_resource
def get_session():
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session

def fetch_data(store_filter=None, status_filter=None):
    session = get_session()
    all_rows = []
    
    # Decide which stores to loop
    target_stores = [s for s in STORES if s['url'] == store_filter] if store_filter else STORES
    
    status_list = [status_filter] if status_filter else ["Pending", "Approved", "Received"]
    if status_filter == "NoTrack": status_list = ["Approved"]

    log(f"Starting sync for {len(target_stores)} stores...")

    for store in target_stores:
        headers = {"X-API-KEY": MY_API_KEY, "x-shop-name": store["url"]}
        
        for status in status_list:
            try:
                res = session.get(f"https://api.returngo.ai/rmas?status={status}&pagesize=50", headers=headers, timeout=10)
                if res.status_code == 200:
                    rmas = res.json().get("rmas", [])
                    
                    for rma in rmas:
                        # Logic for "NoTrack" filter
                        if status_filter == "NoTrack":
                            # We need to peek at details to confirm no tracking
                            try:
                                det = session.get(f"https://api.returngo.ai/rma/{rma['rmaId']}", headers=headers, timeout=5).json()
                                shipments = det.get('shipments', [])
                                has_tracking = any(s.get('trackingNumber') for s in shipments)
                                if has_tracking: continue # Skip if it has tracking
                            except: continue

                        # Basic Row Data (Fast)
                        row = {
                            "Store": store["name"],
                            "RMA ID": rma.get('rmaId'),
                            "Order": rma.get('order_name'),
                            "Status": rma.get('status'),
                            "Date": rma.get('createdAt')[:10]
                        }
                        all_rows.append(row)
            except Exception as e:
                log(f"Error {store['name']}: {e}")

    log(f"Sync complete. Found {len(all_rows)} records.")
    return pd.DataFrame(all_rows)

# --- LAYOUT ---
st.title("🚀 RMA Central | Web Dashboard")

if not MY_API_KEY:
    st.error("API Key not found! Please set 'ReturnGo_API' environment variable.")
    st.stop()

# Top Bar: Actions
col1, col2 = st.columns([1, 4])
with col1:
    if st.button("🔄 Sync All Data", use_container_width=True):
        with st.spinner("Fetching data from all stores..."):
            st.session_state.data_cache = fetch_data()

# Status Cards (The "Blue Boxes")
st.markdown("### Store Overview")
cols = st.columns(len(STORES))

for i, store in enumerate(STORES):
    with cols[i]:
        st.subheader(store["name"])
        # In a real app, these would recalculate dynamically
        # For this example, buttons trigger filters
        if st.button(f"Pending", key=f"btn_p_{i}"):
            st.session_state.data_cache = fetch_data(store['url'], "Pending")
        if st.button(f"Approved", key=f"btn_a_{i}"):
            st.session_state.data_cache = fetch_data(store['url'], "Approved")
        if st.button(f"No Track", key=f"btn_n_{i}"):
            st.session_state.data_cache = fetch_data(store['url'], "NoTrack")

st.divider()

# Main Table Area
st.subheader("📋 Active Returns")

if not st.session_state.data_cache.empty:
    # Streamlit Data Editor allows built-in searching, sorting, and copying!
    st.data_editor(
        st.session_state.data_cache,
        use_container_width=True,
        hide_index=True,
        column_config={
            "RMA ID": st.column_config.TextColumn("RMA ID", help="Double click to copy"),
        }
    )
    
    # Download Button
    csv = st.session_state.data_cache.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download to Excel", data=csv, file_name="rma_export.csv", mime="text/csv")
else:
    st.info("No data loaded. Click 'Sync All Data' or select a store status above.")

# Logger Expander
with st.expander("Show System Logs"):
    for msg in st.session_state.logs:
        st.text(msg)