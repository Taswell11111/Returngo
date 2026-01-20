import streamlit as st
import sqlite3
import requests
import json
import pandas as pd
import os
import threading
from datetime import datetime, timedelta, timezone
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import concurrent.futures

# ==========================================
# 1. CONFIGURATION
# ==========================================
st.set_page_config(page_title="RMA Central | Web", layout="wide", page_icon="📦")

# ACCESS SECRETS
try:
    MY_API_KEY = st.secrets["RETURNGO_API_KEY"]
except (FileNotFoundError, KeyError):
    MY_API_KEY = os.environ.get("RETURNGO_API_KEY")

if not MY_API_KEY:
    st.error("API Key not found! Please set 'RETURNGO_API_KEY' in secrets or env vars.")
    st.stop()

CACHE_EXPIRY_HOURS = 4
DB_FILE = "rma_cache.db"
# LOCK FOR THREAD-SAFE DB WRITES
DB_LOCK = threading.Lock()

STORES = [
    {"name": "Diesel", "url": "diesel-dev-south-africa.myshopify.com"},
    {"name": "Hurley", "url": "hurley-dev-south-africa.myshopify.com"},
    {"name": "Jeep Apparel", "url": "jeep-apparel-dev-south-africa.myshopify.com"},
    {"name": "Reebok", "url": "reebok-dev-south-africa.myshopify.com"},
    {"name": "Superdry", "url": "superdry-dev-south-africa.myshopify.com"}
]

# --- STYLING ---
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: white; }
    
    div.stButton > button {
        width: 100%;
        border: 1px solid #4b5563;
        background-color: #1f2937;
        color: white;
        border-radius: 8px;
        padding: 10px 0px;
        font-size: 14px;
        font-weight: bold;
    }
    div.stButton > button:hover {
        border-color: #1f538d; 
        color: #1f538d;
    }
    .action-panel {
        border: 2px solid #1f538d;
        padding: 20px;
        border-radius: 10px;
        background-color: #1a1a1a;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Helper for Session
@st.cache_resource
def get_session():
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session

# ==========================================
# 2. DATABASE MANAGER
# ==========================================
def init_db():
    with DB_LOCK:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS rmas
                     (rma_id TEXT PRIMARY KEY, store_url TEXT, status TEXT, 
                      created_at TEXT, json_data TEXT, last_fetched TIMESTAMP)''')
        conn.commit()
        conn.close()

def save_rma_to_db(rma_id, store_url, status, created_at, data):
    # Use Lock to prevent "Database Locked" errors during multi-threading
    with DB_LOCK:
        try:
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            now = datetime.now().isoformat()
            c.execute('''INSERT OR REPLACE INTO rmas (rma_id, store_url, status, created_at, json_data, last_fetched)
                         VALUES (?, ?, ?, ?, ?, ?)''', 
                         (str(rma_id), store_url, status, created_at, json.dumps(data), now))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"DB Error saving {rma_id}: {e}")

def get_rma_from_db(rma_id):
    # Reads are usually safe, but locking ensures we don't read while writing
    with DB_LOCK:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT json_data, last_fetched FROM rmas WHERE rma_id=?", (str(rma_id),))
        row = c.fetchone()
        conn.close()
    if row:
        return json.loads(row[0]), datetime.fromisoformat(row[1])
    return None, None

def get_all_active_from_db():
    with DB_LOCK:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT json_data, store_url FROM rmas WHERE status IN ('Pending', 'Approved', 'Received')")
        rows = c.fetchall()
        conn.close()
    
    results = []
    for r in rows:
        data = json.loads(r[0])
        # Ensure the store_url is attached to the data object for the UI to use
        data['store_url'] = r[1] 
        results.append(data)
    return results

def clear_db():
    with DB_LOCK:
        try:
            if os.path.exists(DB_FILE):
                os.remove(DB_FILE)
            return True
        except: return False

# Initialize DB on load
init_db()

# ==========================================
# 3. BACKEND LOGIC
# ==========================================

def fetch_all_pages(session, headers, status):
    all_rmas = []
    page = 1
    cursor = None
    
    while True:
        try:
            base_url = f"https://api.returngo.ai/rmas?status={status}&pagesize=50"
            url = f"{base_url}&cursor={cursor}" if cursor else base_url
            
            res = session.get(url, headers=headers, timeout=15)
            if res.status_code != 200: break
            
            data = res.json()
            rmas = data.get("rmas", [])
            if not rmas: break
            
            all_rmas.extend(rmas)
            
            cursor = data.get("next_cursor")
            if not cursor: break
            
            page += 1
            if page > 100: break
        except: break
    return all_rmas

def fetch_rma_detail(args):
    # args tuple: (rma_summary, store_url, force_refresh)
    rma_summary, store_url, force_refresh = args
    rma_id = rma_summary.get('rmaId')
    
    # 1. Check Cache (if not forced)
    if not force_refresh:
        cached_data, last_fetched = get_rma_from_db(rma_id)
        if cached_data and last_fetched:
            if (datetime.now() - last_fetched) < timedelta(hours=CACHE_EXPIRY_HOURS):
                return cached_data

    # 2. Fetch from API
    session = get_session()
    headers = {"X-API-KEY": MY_API_KEY, "x-shop-name": store_url}
    
    try:
        res = session.get(f"https://api.returngo.ai/rma/{rma_id}", headers=headers, timeout=15)
        if res.status_code == 200:
            data = res.json()
            # 3. Save to DB (Thread Safe)
            save_rma_to_db(rma_id, store_url, rma_summary.get('status'), rma_summary.get('createdAt'), data)
            return data
        else:
            print(f"Error fetching {rma_id}: {res.status_code}")
    except Exception as e:
        print(f"Exception fetching {rma_id}: {e}")
    return None

def perform_sync(target_store=None, target_status=None):
    session = get_session()
    
    status_msg = st.empty()
    status_msg.info("⏳ Connecting to ReturnGO API...")
    
    tasks = [] 
    
    # Logic to select stores
    if target_store:
        stores_to_sync = [target_store] # target_store is the full dict object passed from handle_filter_click
    else:
        stores_to_sync = STORES
    
    # Logic to select statuses
    if target_status and target_status in ["Pending", "Approved", "Received"]:
        statuses = [target_status]
    elif target_status == "NoTrack":
        statuses = ["Approved"]
    elif target_status == "Flagged":
        statuses = ["Pending", "Approved"]
    else:
        statuses = ["Pending", "Approved", "Received"] # Default / Sync All

    total_found = 0
    list_bar = None
    if not target_store: 
        list_bar = st.progress(0, text="Fetching Lists from ReturnGO...")

    # 1. Gather all RMA Summaries first
    for i, store in enumerate(stores_to_sync):
        headers = {"X-API-KEY": MY_API_KEY, "x-shop-name": store['url']}
        for s in statuses:
            rmas = fetch_all_pages(session, headers, s)
            for r in rmas:
                # Append task: (rma_summary, store_url, force_refresh=True)
                tasks.append((r, store['url'], True))
            total_found += len(rmas)
        
        if list_bar: 
            list_bar.progress((i + 1) / len(stores_to_sync), text=f"Fetched {store['name']}")
    
    if list_bar: list_bar.empty()

    status_msg.info(f"⏳ Found {total_found} records. Downloading details...")
    
    # 2. Download details in parallel
    if total_found > 0:
        bar = st.progress(0, text="Downloading Details...")
        # Use ThreadPool to fetch details faster
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # Submit all tasks
            futures = {executor.submit(fetch_rma_detail, task): task for task in tasks}
            completed = 0
            
            for future in concurrent.futures.as_completed(futures):
                completed += 1
                if completed % 5 == 0: # Update progress every 5 items
                    bar.progress(completed / total_found, text=f"Syncing: {completed}/{total_found}")
        bar.empty()
                
    st.session_state['last_sync'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state['show_toast'] = True
    
    status_msg.success(f"✅ Sync Complete! {total_found} records updated.")
    
    # Important: Rerun to reload data from the now-updated DB
    st.rerun()

def push_tracking_update(rma_id, shipment_id, tracking_number, store_url):
    session = get_session()
    headers = {"X-API-KEY": MY_API_KEY, "x-shop-name": store_url, "Content-Type": "application/json"}
    
    payload = {
        "status": "LabelCreated",
        "carrierName": "CourierGuy",
        "trackingNumber": tracking_number,
        "trackingURL": f"https://optimise.parcelninja.com/shipment/track/{tracking_number}",
        "labelURL": "https://sellerportal.dpworld.com/api/file-download?link=null"
    }
    
    try:
        res = session.put(f"https://api.returngo.ai/shipment/{shipment_id}", headers=headers, json=payload, timeout=10)
        
        if res.status_code == 200:
            # Update local cache immediately
            fresh_res = session.get(f"https://api.returngo.ai/rma/{rma_id}", headers=headers, timeout=10)
            if fresh_res.status_code == 200:
                fresh_data = fresh_res.json()
                summary = fresh_data.get('rmaSummary', {})
                save_rma_to_db(rma_id, store_url, summary.get('status', 'Approved'), summary.get('createdAt'), fresh_data)
            return True, "Success"
        else:
            return False, f"API Error {res.status_code}: {res.text}"
    except Exception as e:
        return False, str(e)

# ==========================================
# 4. FRONTEND UI LOGIC
# ==========================================

# Initialize Filter State
if 'filter_state' not in st.session_state:
    st.session_state.filter_state = {"store": None, "status": "All"}

# Show toast
if st.session_state.get('show_toast'):
    st.toast("✅ Data Refreshed Successfully!", icon="🔄")
    st.session_state['show_toast'] = False

# Helper to handle button clicks
def handle_filter_click(store_url, status):
    st.session_state.filter_state = {"store": store_url, "status": status}
    
    # If clicking a specific status button (e.g., "Pending"), force a sync for that status
    if status in ["Pending", "Approved", "Received"]:
        # Find the full store object
        store_obj = next((s for s in STORES if s['url'] == store_url), None)
        if store_obj:
            perform_sync(store_obj, status)
    else:
        # For derived statuses like "NoTrack" or "Flagged", just rerun UI filters
        st.rerun()

# --- Header ---
col1, col2 = st.columns([3, 1])
with col1:
    st.title("RMA Central (Multi-Store) 📦")
    last_sync = st.session_state.get('last_sync', 'Not run yet')
    st.markdown(f"**Last Sync:** :green[{last_sync}]")
with col2:
    if st.button("🔄 Sync All Data", type="primary", use_container_width=True):
        perform_sync() # No args = Sync All Stores, All Statuses
    if st.button("🗑️ Reset Cache", type="secondary", use_container_width=True):
        if clear_db():
            st.success("Cache cleared!")
            st.rerun()
        else:
            st.error("Could not clear cache. DB might be locked.")

# --- Process Data (Load from DB) ---
# This runs on every script execution to load what's currently in SQLite
raw_data = get_all_active_from_db()

processed_rows = []
# Initialize counts for all stores
store_counts = {s['url']: {"Pending": 0, "Approved": 0, "Received": 0, "NoTrack": 0, "Flagged": 0} for s in STORES}

for rma in raw_data:
    store_url = rma.get('store_url')
    if not store_url: continue 
    
    summary = rma.get('rmaSummary', {})
    shipments = rma.get('shipments', [])
    comments = rma.get('comments', [])
    status = summary.get('status', 'Unknown')
    
    rma_id = summary.get('rmaId', 'N/A')
    order_name = summary.get('order_name', 'N/A')
    
    track_nums = [s.get('trackingNumber') for s in shipments if s.get('trackingNumber')]
    track_str = ", ".join(track_nums) if track_nums else ""
    shipment_id = shipments[0].get('shipmentId') if shipments else None
    
    # Date Handling
    created_at = summary.get('createdAt')
    if not created_at:
        for evt in summary.get('events', []):
            if evt.get('eventName') == 'RMA_CREATED':
                created_at = evt.get('eventDate')
                break
    
    # Calculate Days Since Update
    updated_at = rma.get('lastUpdated')
    days_since = 0
    if updated_at:
        try:
            d = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
            days_since = (datetime.now(timezone.utc).date() - d.date()).days
        except: pass

    # Update Counts
    if store_url in store_counts:
        # Check if the status key exists before incrementing to avoid KeyErrors
        if status in store_counts[store_url]: 
            store_counts[store_url][status] += 1
        
        is_no_track = False
        if status == "Approved":
            if not shipments or not track_str:
                store_counts[store_url]["NoTrack"] += 1
                is_no_track = True
        
        is_flagged = False
        if any("flagged" in c.get('htmlText', '').lower() for c in comments):
            store_counts[store_url]["Flagged"] += 1
            is_flagged = True

        processed_rows.append({
            "RMA ID": rma_id,
            "Order": order_name,
            "Store URL": store_url,
            "Store Name": next((s['name'] for s in STORES if s['url'] == store_url), "Unknown"),
            "Status": status,
            "Tracking": track_str,
            "Created": str(created_at)[:10] if created_at else "N/A",
            "Updated": str(updated_at)[:10] if updated_at else "N/A",
            "Days": days_since,
            "IsNoTrack": is_no_track,
            "IsFlagged": is_flagged,
            "shipment_id": shipment_id,
            "full_data": rma
        })

# --- Draw Metric Boxes ---
cols = st.columns(len(STORES))

for i, store in enumerate(STORES):
    c = store_counts[store['url']]
    with cols[i]:
        st.markdown(f"**{store['name']}**")
        # Buttons invoke the handler function directly
        if st.button(f"Pending\n{c['Pending']}", key=f"p_{i}"): 
             handle_filter_click(store['url'], "Pending")
        if st.button(f"Approved\n{c['Approved']}", key=f"a_{i}"): 
             handle_filter_click(store['url'], "Approved")
        if st.button(f"Received\n{c['Received']}", key=f"r_{i}"): 
             handle_filter_click(store['url'], "Received")
        if st.button(f"No Track\n{c['NoTrack']}", key=f"n_{i}"): 
             handle_filter_click(store['url'], "NoTrack")
        if st.button(f"🚩 Flagged\n{c['Flagged']}", key=f"f_{i}"): 
             handle_filter_click(store['url'], "Flagged")

st.divider()

# --- Filter & Display Table ---
current_filter = st.session_state.filter_state
filter_desc = f"{current_filter['status']} Records"
if current_filter['store']:
    s_name = next((s['name'] for s in STORES if s['url'] == current_filter['store']), "Unknown")
    filter_desc += f" for {s_name}"

st.subheader(f"📋 {filter_desc}")

df = pd.DataFrame(processed_rows)

if not df.empty:
    # 1. Filter by Store
    if current_filter['store']:
        df = df[df['Store URL'] == current_filter['store']]
    
    # 2. Filter by Status
    f_stat = current_filter['status']
    if f_stat == "Pending": df = df[df['Status'] == 'Pending']
    elif f_stat == "Approved": df = df[df['Status'] == 'Approved']
    elif f_stat == "Received": df = df[df['Status'] == 'Received']
    elif f_stat == "NoTrack": df = df[df['IsNoTrack'] == True]
    elif f_stat == "Flagged": df = df[df['IsFlagged'] == True]

    # 3. Sort & Display
    if not df.empty:
        df = df.sort_values(by="Created", ascending=False)
        df = df.reset_index(drop=True)
        df.insert(0, "No", range(1, len(df) + 1))

        event = st.dataframe(
            df[["No", "Store Name", "RMA ID", "Order", "Status", "Tracking", "Created", "Updated", "Days"]],
            use_container_width=True,
            hide_index=True,
            selection_mode="single-row",
            on_select="rerun",
            key="main_table"
        )

        selected = st.session_state.main_table.get("selection", {}).get("rows", [])
        
        if selected:
            idx = selected[0]
            if idx < len(df):
                record = df.iloc[idx]
                st.markdown("<div class='action-panel'>", unsafe_allow_html=True)
                st.markdown(f"### 🛠️ Manage RMA: `{record['RMA ID']}` ({record['Store Name']})")
                t1, t2 = st.tabs(["📝 Timeline", "✏️ Update Tracking"])
                with t1:
                    full = record['full_data']
                    timeline = full.get('comments', [])
                    if not timeline: st.info("No history.")
                    for t in timeline:
                        d_str = t.get('datetime', '')[:16].replace('T', ' ')
                        who = t.get('triggeredBy', 'System')
                        msg = t.get('htmlText', '')
                        st.markdown(f"**{d_str}** | `{who}`\n> {msg}")
                        st.divider()
                with t2:
                    with st.form("track_upd"):
                        new_track = st.text_input("New Tracking", value=record['Tracking'])
                        if st.form_submit_button("Save"):
                            if not record['shipment_id']:
                                st.error("No Shipment ID.")
                            else:
                                ok, msg = push_tracking_update(record['RMA ID'], record['shipment_id'], new_track, record['Store URL'])
                                if ok:
                                    st.success("Updated!")
                                    st.rerun()
                                else: st.error(msg)
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("👆 Click on any row in the table to view Timeline or Update Tracking.")
    else:
        st.info("No records match the current filter.")
else:
    st.info("No records found in database. Click 'Sync All Data' to start.")
