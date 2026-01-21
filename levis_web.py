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
st.set_page_config(page_title="Levi's ReturnGo RMAs", layout="wide", page_icon="📤")

# ACCESS SECRETS
try:
    MY_API_KEY = st.secrets["RETURNGO_API_KEY"]
except FileNotFoundError:
    st.error("Secrets file not found! Please create .streamlit/secrets.toml for local use.")
    st.stop()
except KeyError:
    st.error("RETURNGO_API_KEY not found in secrets!")
    st.stop()

CACHE_EXPIRY_HOURS = 4
STORE_URL = "levis-sa.myshopify.com"
DB_FILE = "levis_cache.db"
# LOCK FOR THREAD-SAFE DB WRITES
DB_LOCK = threading.Lock()

# --- STYLING ---
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: white; }
    
    /* Metrics/Filter Boxes */
    div.stButton > button {
        width='stretch': 100%;
        border: 1px solid #4b5563;
        background-color: #1f2937;
        color: white;
        border-radius: 8px;
        padding: 12px 24px;
        font-size: 16px;
        font-weight: bold;
    }
    div.stButton > button:hover {
        border-color: #c41230; /* Levi's Red */
        color: #c41230;
    }
    div.stButton > button:focus {
        border-color: #c41230;
        color: #c41230;
        background-color: #2d1b1b;
    }

    /* Modal Styling Fixes */
    div[data-testid="stDialog"] {
        background-color: #1a1a1a;
        border: 1px solid #c41230;
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

def save_rma_to_db(rma_id, status, created_at, data):
    with DB_LOCK:
        try:
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            now = datetime.now().isoformat()
            c.execute('''INSERT OR REPLACE INTO rmas (rma_id, store_url, status, created_at, json_data, last_fetched)
                         VALUES (?, ?, ?, ?, ?, ?)''', 
                         (str(rma_id), STORE_URL, status, created_at, json.dumps(data), now))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"DB Error saving {rma_id}: {e}")

def get_rma_from_db(rma_id):
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
        # Fetch Pending, Approved AND Received
        c.execute("SELECT json_data FROM rmas WHERE store_url=? AND status IN ('Pending', 'Approved', 'Received')", (STORE_URL,))
        rows = c.fetchall()
        conn.close()
    return [json.loads(r[0]) for r in rows]

def get_local_ids_for_status(status):
    with DB_LOCK:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT rma_id FROM rmas WHERE store_url=? AND status=?", (STORE_URL, status))
        rows = c.fetchall()
        conn.close()
    return {r[0] for r in rows}

def clear_db():
    with DB_LOCK:
        try:
            if os.path.exists(DB_FILE):
                os.remove(DB_FILE)
            return True
        except: return False

init_db()

# ==========================================
# 3. BACKEND LOGIC
# ==========================================

def fetch_all_pages(session, headers, status):
    """Iterates through API pages using cursor logic to get TRUE total."""
    all_rmas = []
    cursor = None
    page_count = 1
    
    while True:
        try:
            # Build URL with cursor
            base_url = f"https://api.returngo.ai/rmas?status={status}&pagesize=50"
            if cursor:
                url = f"{base_url}&cursor={cursor}"
            else:
                url = base_url
            
            res = session.get(url, headers=headers, timeout=15)
            if res.status_code != 200: break
            
            data = res.json()
            rmas = data.get("rmas", [])
            if not rmas: break
            
            all_rmas.extend(rmas)
            
            # Check for next cursor
            cursor = data.get("next_cursor")
            if not cursor: break
            
            page_count += 1
            if page_count > 100: break # Safety break
        except: break
    return all_rmas

def fetch_rma_detail(rma_summary):
    # Determine if we are being called with a tuple (from parallel executor in sync) or just a dict
    # Although in this single store version we might pass dict, let's standardize.
    # Actually, in perform_sync below I will submit just the rma_summary dict or tuple.
    # To keep it simple and consistent with previous levis logic, let's assume rma_summary is the dict.
    
    rma_id = rma_summary.get('rmaId')
    
    # We are usually forcing refresh in sync, but let's keep cache logic if needed
    # For full sync we want fresh data.
    
    session = get_session()
    headers = {"X-API-KEY": MY_API_KEY, "x-shop-name": STORE_URL}
    try:
        res = session.get(f"https://api.returngo.ai/rma/{rma_id}", headers=headers, timeout=15)
        if res.status_code == 200:
            data = res.json()
            
            # CRITICAL FIX: Extract TRUE status from fresh data
            fresh_summary = data.get('rmaSummary', {})
            true_status = fresh_summary.get('status', 'Unknown')
            true_created = fresh_summary.get('createdAt')
            
            save_rma_to_db(rma_id, true_status, true_created, data)
            return data
    except: pass
    return None

def perform_sync(statuses=None):
    session = get_session()
    headers = {"X-API-KEY": MY_API_KEY, "x-shop-name": STORE_URL}
    
    status_msg = st.empty()
    status_msg.info("⏳ Starting Deep Sync... fetching all pages.")
    
    active_summaries = []
    
    # 1. Fetch ALL pages for relevant statuses
    if statuses is None:
        statuses = ["Pending", "Approved", "Received"]

    for s in statuses:
        # A. Fetch Fresh List from API
        api_rmas = fetch_all_pages(session, headers, s)
        
        # B. Get local IDs for cleanup logic
        local_ids = get_local_ids_for_status(s)
        
        # C. Identify Stale items
        api_ids = {r.get('rmaId') for r in api_rmas}
        stale_ids = local_ids - api_ids
        
        # D. Add API items to list
        active_summaries.extend(api_rmas)
        
        # E. Add Stale items to list (to force update their status)
        for stale_id in stale_ids:
            active_summaries.append({'rmaId': stale_id, 'status': 'CHECK_UPDATE', 'createdAt': None})
    
    total = len(active_summaries)
    status_msg.info(f"⏳ Found {total} records. Downloading details...")
    
    # 2. Parallel Fetch Details
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_rma_detail, rma): rma for rma in active_summaries}
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            completed += 1
            if total and completed % 5 == 0:
                try:
                    status_msg.progress(completed / total, text=f"Syncing: {completed}/{total} RMAs")
                except Exception: pass
                
    st.session_state['last_sync'] = datetime.now().strftime("%Y-%m-%d %H:%M")
    st.session_state['show_toast'] = True
    status_msg.success(f"✅ Sync Complete! Total: {total}")
    st.rerun()

def push_tracking_update(rma_id, shipment_id, tracking_number):
    session = get_session()
    headers = {"X-API-KEY": MY_API_KEY, "x-shop-name": STORE_URL, "Content-Type": "application/json"}
    
    payload = {
        "status": "LabelCreated",
        "carrierName": "CourierGuy",
        "trackingNumber": tracking_number,
        "trackingURL": f"https://optimise.parcelninja.com/shipment/track/{tracking_number}",
        "labelURL": "https://sellerportal.dpworld.com/api/file-download?link=null"
    }
    
    try:
        # Use PUT /shipment/{id}
        res = session.put(f"https://api.returngo.ai/shipment/{shipment_id}", headers=headers, json=payload, timeout=10)
        
        if res.status_code == 200:
            fresh_res = session.get(f"https://api.returngo.ai/rma/{rma_id}", headers=headers, timeout=10)
            if fresh_res.status_code == 200:
                fresh_data = fresh_res.json()
                summary = fresh_data.get('rmaSummary', {})
                save_rma_to_db(rma_id, summary.get('status', 'Approved'), summary.get('createdAt'), fresh_data)
            return True, "Success"
        else:
            return False, f"API Error {res.status_code}: {res.text}"
    except Exception as e:
        return False, str(e)

# ==========================================
# 4. FRONTEND UI
# ==========================================

if 'filter_status' not in st.session_state:
    st.session_state.filter_status = "All"

if st.session_state.get('show_toast'):
    st.toast("✅ Data Refreshed Successfully!", icon="📤")
    st.session_state['show_toast'] = False

# --- Modal Function ---
@st.dialog("RMA Details")
def show_rma_modal(record):
    st.markdown(f"### 🛠️ Manage RMA: `{record['RMA ID']}`")
    
    t1, t2 = st.tabs(["📝 Timeline", "✏️ Update Tracking"])
    
    with t1:
        full = record['full_data']
        timeline = full.get('comments', [])
        if not timeline:
            st.info("No timeline events found.")
        else:
            for t in timeline:
                d_str = t.get('datetime', '')[:16].replace('T', ' ')
                who = t.get('triggeredBy', 'System')
                msg = t.get('htmlText', '')
                st.markdown(f"**{d_str}** | `{who}`")
                st.caption(msg, unsafe_allow_html=True)
                st.divider()

    with t2:
        # Use DisplayTrack (raw text) for the input box value
        raw_track = record['DisplayTrack']
        with st.form("update_track_form"):
            new_track = st.text_input("New Tracking Number", value=raw_track)
            submitted = st.form_submit_button("Save Changes")
            
            if submitted:
                if not record['shipment_id']:
                    st.error("No Shipment ID associated with this RMA.")
                else:
                    ok, msg = push_tracking_update(record['RMA ID'], record['shipment_id'], new_track)
                    if ok:
                        st.success("Tracking Updated! Refreshing...")
                        st.rerun()
                    else:
                        st.error(msg)

# --- Header ---
col1, col2 = st.columns([3, 1])
with col1:
    st.title("Levi's ReturnGO RMA Operations")
    last_sync = st.session_state.get('last_sync', 'Not run yet')
    st.markdown(f"**Connected to:** {STORE_URL} | **Last Sync:** :green[{last_sync}]")
with col2:
    if st.button("🔄 Sync All Data", type="primary", width='stretch'):
        perform_sync()
    if st.button("🗑️ Reset Cache", type="secondary", width='stretch'):
        if clear_db():
            st.success("Cache cleared!")
            st.rerun()
        else:
            st.error("Could not clear cache.")

# --- Data Processing ---
raw_data = get_all_active_from_db()
processed_rows = []
counts = {"Pending": 0, "Approved": 0, "Received": 0, "NoTrack": 0, "Flagged": 0}

for rma in raw_data:
    summary = rma.get('rmaSummary', {})
    shipments = rma.get('shipments', [])
    comments = rma.get('comments', [])
    
    status = summary.get('status', 'Unknown')
    rma_id = summary.get('rmaId', 'N/A')
    order_name = summary.get('order_name', 'N/A')
    
    # Tracking
    track_nums = [s.get('trackingNumber') for s in shipments if s.get('trackingNumber')]
    track_str = ", ".join(track_nums) if track_nums else ""
    shipment_id = shipments[0].get('shipmentId') if shipments else None
    
    # URL Generation
    track_url = None
    if track_str:
        primary_track = track_nums[0] if track_nums else ""
        track_url = f"https://portal.thecourierguy.co.za/track?ref={primary_track}"
    
    # Created Date
    created_at = summary.get('createdAt')
    if not created_at:
        for evt in summary.get('events', []):
            if evt.get('eventName') == 'RMA_CREATED':
                created_at = evt.get('eventDate')
                break
    created_display = str(created_at)[:10] if created_at else "N/A"

    # Days Since
    updated_at = rma.get('lastUpdated')
    days_since = 0
    if updated_at:
        try:
            d = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
            days_since = (datetime.now(timezone.utc).date() - d.date()).days
        except: pass

    # Counting
    if status in counts: counts[status] += 1
    
    is_no_track = False
    if status == "Approved":
        if not shipments or not track_str:
            counts["NoTrack"] += 1
            is_no_track = True
            
    is_flagged = False
    if any("flagged" in c.get('htmlText', '').lower() for c in comments):
        counts["Flagged"] += 1
        is_flagged = True

    processed_rows.append({
        "RMA ID": rma_id,
        "Order": order_name,
        "Status": status,
        "TrackingNumber": track_url if track_url else track_str,
        "DisplayTrack": track_str,
        "Created": created_display,
        "Updated": str(updated_at)[:10] if updated_at else "N/A",
        "Days since updated": days_since,
        "IsNoTrack": is_no_track,
        "IsFlagged": is_flagged,
        "shipment_id": shipment_id,
        "full_data": rma
    })

# --- Interactive Filter Boxes ---
b1, b2, b3, b4, b5 = st.columns(5)

# updated set_filter to optionally perform targeted syncs
def set_filter(f):
    st.session_state.filter_status = f
    # If the user clicked one of the primary statuses, do a targeted fetch for fresh data
    if f in ["Pending", "Approved", "Received"]:
        perform_sync(statuses=[f])
    else:
        # No network action required for derived filters — just rerun to refresh UI
        st.rerun()

with b1:
    if st.button(f"Pending\n{counts['Pending']}"): set_filter("Pending")
with b2:
    if st.button(f"Approved\n{counts['Approved']}"): set_filter("Approved")
with b3:
    if st.button(f"Received\n{counts['Received']}"): set_filter("Received")
with b4:
    if st.button(f"No Tracking\n{counts['NoTrack']} "): set_filter("NoTrack")
with b5:
    if st.button(f"🚩 Flagged\n{counts['Flagged']} "): set_filter("Flagged")

# --- Table Display ---
st.divider()
st.subheader(f"📋 {st.session_state.filter_status} Records")

df = pd.DataFrame(processed_rows)

if not df.empty:
    f_stat = st.session_state.filter_status
    if f_stat == "Pending": display_df = df[df['Status'] == 'Pending']
    elif f_stat == "Approved": display_df = df[df['Status'] == 'Approved']
    elif f_stat == "Received": display_df = df[df['Status'] == 'Received']
    elif f_stat == "NoTrack": display_df = df[df['IsNoTrack'] == True]
    elif f_stat == "Flagged": display_df = df[df['IsFlagged'] == True]
    else: display_df = df

    if not display_df.empty:
        # sort and add row numbers
        display_df = display_df.sort_values(by="Created", ascending=False).reset_index(drop=True)
        display_df.insert(0, "No", range(1, len(display_df) + 1))
        
        # Format columns for Left Indent
        display_df['No'] = display_df['No'].astype(str)
        display_df['Days since updated'] = display_df['Days since updated'].astype(str)

        # --- MAIN TABLE ---
        event = st.dataframe(
            display_df[["No", "RMA ID", "Order", "Status", "TrackingNumber", "Created", "Updated", "Days since updated"]],
            use_container_width=True,
            height=700,
            hide_index=True,
            selection_mode="single-row",
            on_select="rerun",
            key="main_table",
            column_config={
                "TrackingNumber": st.column_config.LinkColumn(
                    "TrackingNumber",
                    display_text=r"ref=(.*)"
                ),
                "No": st.column_config.TextColumn("No", width="small"),
                "Days since updated": st.column_config.TextColumn("Days since updated", width="small")
            }
        )
        
        # --- Trigger Modal on Selection ---
        selected = st.session_state.main_table.get("selection", {}).get("rows", [])
        if selected:
            idx = selected[0]
            if idx < len(display_df):
                record = display_df.iloc[idx]
                show_rma_modal(record)

    else:
        st.info("No records match the current filter.")
else:
    st.warning("No records found in database. Please click 'Sync All Data' to fetch initial data.")
