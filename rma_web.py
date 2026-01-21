import streamlit as st
import sqlite3
import requests
import json
import pandas as pd
import os
import threading
import time
import re  # Built-in Regex module
from datetime import datetime, timedelta, timezone
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import concurrent.futures

# ==========================================
# 1. CONFIGURATION
# ==========================================
st.set_page_config(page_title="Bounty Apparel ReturnGo RMAs", layout="wide", page_icon="🔄️")

# ACCESS SECRETS
try:
    MY_API_KEY = st.secrets["RETURNGO_API_KEY"]
except (FileNotFoundError, KeyError):
    MY_API_KEY = os.environ.get("RETURNGO_API_KEY")

if not MY_API_KEY:
    st.error("API Key not found! Please set 'RETURNGO_API_KEY' in secrets or env vars.")
    st.stop()

# PARCEL NINJA TOKEN (TASWELL)
PARCEL_NINJA_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbiI6IjJhMDI4MWNhNTBmNDEwOTRiZTkyNzdhNTQ0MDZhZGRkODMyOGExODhhYmNiZGViMiIsIm5iZiI6MTc2ODk1ODUwMSwiZXhwIjoxODYzNjUyOTAxLCJpYXQiOjE3Njg5NTg1MDEsImlzcyI6Imh0dHBzOi8vb3B0aW1pc2UucGFyY2VsbmluamEuY29tIiwiYXVkIjoiaHR0cHM6Ly9vcHRpbWlzZS5wYXJjZWxuaW5qYS5jb20ifQ.lgAi9s2INGKrzGYb3Qn_PY6N1ekh3fSBP7JBgMhX0Pk"

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
        padding: 12px 24px;
        font-size: 14px;
        font-weight: bold;
    }
    div.stButton > button:hover {
        border-color: #1f538d; 
        color: #1f538d;
    }
    div[data-testid="stDialog"] {
        background-color: #1a1a1a;
        border: 1px solid #4b5563;
    }
    .sync-time {
        font-size: 0.8em;
        color: #888;
        text-align: center;
        margin-top: -10px;
        margin-bottom: 10px;
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
        c.execute('''CREATE TABLE IF NOT EXISTS sync_logs
                     (store_url TEXT, status TEXT, last_sync TIMESTAMP, PRIMARY KEY (store_url, status))''')
        conn.commit()
        conn.close()

def save_rma_to_db(rma_id, store_url, status, created_at, data):
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
        data['store_url'] = r[1] 
        results.append(data)
    return results

def get_local_ids_for_status(store_url, status):
    with DB_LOCK:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT rma_id FROM rmas WHERE store_url=? AND status=?", (store_url, status))
        rows = c.fetchall()
        conn.close()
    return {r[0] for r in rows}

def update_sync_log(store_url, status):
    with DB_LOCK:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        c.execute('''INSERT OR REPLACE INTO sync_logs (store_url, status, last_sync)
                     VALUES (?, ?, ?)''', (store_url, status, now))
        conn.commit()
        conn.close()

def get_last_sync(store_url, status):
    with DB_LOCK:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        try:
            c.execute("SELECT last_sync FROM sync_logs WHERE store_url=? AND status=?", (store_url, status))
            row = c.fetchone()
            return row[0] if row else None
        except: return None
        finally: conn.close()

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
    rma_summary, store_url, force_refresh = args
    rma_id = rma_summary.get('rmaId')
    
    if not force_refresh:
        cached_data, last_fetched = get_rma_from_db(rma_id)
        if cached_data and last_fetched:
            if (datetime.now() - last_fetched) < timedelta(hours=CACHE_EXPIRY_HOURS):
                return cached_data

    session = get_session()
    headers = {"X-API-KEY": MY_API_KEY, "x-shop-name": store_url}
    
    try:
        res = session.get(f"https://api.returngo.ai/rma/{rma_id}", headers=headers, timeout=15)
        if res.status_code == 200:
            data = res.json()
            fresh_summary = data.get('rmaSummary', {})
            true_status = fresh_summary.get('status', 'Unknown')
            true_created = fresh_summary.get('createdAt')
            save_rma_to_db(rma_id, store_url, true_status, true_created, data)
            return data
    except: pass
    return None

def perform_sync(target_store=None, target_status=None):
    session = get_session()
    status_msg = st.empty()
    status_msg.info("⏳ Connecting to ReturnGO API...")
    tasks = [] 
    
    if target_store:
        stores_to_sync = [target_store]
    else:
        stores_to_sync = STORES
    
    if target_status and target_status in ["Pending", "Approved", "Received"]:
        statuses = [target_status]
    elif target_status == "NoTrack":
        statuses = ["Approved"]
    elif target_status == "Flagged":
        statuses = ["Pending", "Approved"]
    else:
        statuses = ["Pending", "Approved", "Received"]

    total_found = 0
    list_bar = None
    if not target_store: 
        list_bar = st.progress(0, text="Fetching Lists from ReturnGO...")

    for i, store in enumerate(stores_to_sync):
        headers = {"X-API-KEY": MY_API_KEY, "x-shop-name": store['url']}
        for s in statuses:
            api_rmas = fetch_all_pages(session, headers, s)
            local_ids = get_local_ids_for_status(store['url'], s)
            api_ids = {r.get('rmaId') for r in api_rmas}
            stale_ids = local_ids - api_ids
            
            for r in api_rmas:
                tasks.append((r, store['url'], True))
            for stale_id in stale_ids:
                placeholder = {'rmaId': stale_id, 'status': 'CHECK_UPDATE', 'createdAt': None}
                tasks.append((placeholder, store['url'], True))
                
            total_found += len(api_rmas) + len(stale_ids)
            update_sync_log(store['url'], s)
        
        if list_bar: 
            list_bar.progress((i + 1) / len(stores_to_sync), text=f"Fetched {store['name']}")
    
    if list_bar: list_bar.empty()

    status_msg.info(f"⏳ Syncing {total_found} records...")
    
    if total_found > 0:
        bar = st.progress(0, text="Downloading Details...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(fetch_rma_detail, task): task for task in tasks}
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                completed += 1
                if completed % 5 == 0:
                    try: bar.progress(completed / total_found, text=f"Syncing: {completed}/{total_found}")
                    except: pass
        bar.empty()
                
    st.session_state['last_sync'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state['show_toast'] = True
    status_msg.success(f"✅ Sync Complete!")
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

# Updated Courier Guy Check using Regex (Zero Dependency)
def check_courier_status(tracking_number):
    try:
        url = f"https://optimise.parcelninja.com/shipment/track/{tracking_number}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Authorization': f'Bearer {PARCEL_NINJA_TOKEN}',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
        }
        
        res = requests.get(url, headers=headers, timeout=8)
        
        if res.status_code == 200:
            content = res.text
            
            # 1. Try JSON
            try:
                data = res.json()
                status = data.get('status') or data.get('currentStatus')
                if status: return f"API Status: {status}"
            except: pass 

            # 2. Regex Search for common statuses in HTML text
            # We look for whole words to avoid false positives
            keywords = [
                r"delivered", 
                r"in\s+transit", 
                r"out\s+for\s+delivery", 
                r"collected", 
                r"created",
                r"at\s+delivery\s+depot"
            ]
            
            found_status = []
            for k in keywords:
                if re.search(k, content, re.IGNORECASE):
                    found_status.append(k.replace(r"\s+", " ").title())
            
            if found_status:
                # Return the most relevant status (usually the last one added to list if chronological, but here just first found)
                return f"Found Status: {', '.join(set(found_status))}"

            # Fallback: Look for specific HTML class if we know it (e.g., <div class="status">)
            # This is harder with regex but simple searches work
            
            return "Page Loaded (Status Keyword Not Found)"
            
        elif res.status_code == 401:
            return "Auth Error (401)"
        elif res.status_code == 404:
            return "Tracking ID Not Found"
        else:
            return f"Error {res.status_code}"
    except Exception as e:
        return f"Check Failed: {str(e)[:30]}..."

# ==========================================
# 4. FRONTEND UI LOGIC
# ==========================================

if 'filter_state' not in st.session_state:
    st.session_state.filter_state = {"store": None, "status": "All"}

if st.session_state.get('show_toast'):
    st.toast("✅ Data Refreshed Successfully!", icon="🔄")
    st.session_state['show_toast'] = False

def handle_filter_click(store_url, status):
    st.session_state.filter_state = {"store": store_url, "status": status}
    if status in ["Pending", "Approved", "Received"]:
        store_obj = next((s for s in STORES if s['url'] == store_url), None)
        if store_obj:
            perform_sync(store_obj, status)
    else:
        st.rerun()

# --- Modal Function ---
@st.dialog("RMA Details")
def show_rma_modal(record):
    st.markdown(f"### 🛠️ Manage RMA: `{record['RMA ID']}`")
    st.caption(f"Store: {record['Store Name']}")
    
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
        raw_track = record['DisplayTrack']
        with st.form("track_upd_modal"):
            new_track = st.text_input("New Tracking", value=raw_track)
            
            # Updated Status Check Button
            if raw_track:
                if st.form_submit_button("🔍 Check Courier Status"):
                    with st.spinner("Checking Parcel Ninja..."):
                        status_info = check_courier_status(raw_track)
                    if "API Status" in status_info or "Found Status" in status_info:
                        st.success(status_info)
                    elif "Missing" in status_info:
                        st.error(status_info)
                    else:
                        st.warning(status_info)
            
            if st.form_submit_button("Save"):
                if not record['shipment_id']:
                    st.error("No Shipment ID.")
                else:
                    ok, msg = push_tracking_update(record['RMA ID'], record['shipment_id'], new_track, record['Store URL'])
                    if ok:
                        st.success("Updated!")
                        st.rerun()
                    else: st.error(msg)

# --- Header ---
col1, col2 = st.columns([3, 1])
with col1:
    st.title("Bounty Apparel ReturnGo RMAs 🔄️")
    # SEARCH BAR
    search_query = st.text_input("🔍 Search Order, RMA, or Tracking", placeholder="Type to search...")
    
with col2:
    if st.button("🔄 Sync All Data", type="primary", use_container_width=True):
        perform_sync()
    if st.button("🗑️ Reset Cache", type="secondary", use_container_width=True):
        if clear_db():
            st.success("Cache cleared!")
            st.rerun()
        else:
            st.error("Could not clear cache. DB might be locked.")

# --- Process Data (Load from DB) ---
raw_data = get_all_active_from_db()
processed_rows = []
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
    
    track_url = None
    if track_str:
        primary_track = track_nums[0] if track_nums else ""
        track_url = f"https://portal.thecourierguy.co.za/track?ref={primary_track}"

    created_at = summary.get('createdAt')
    if not created_at:
        for evt in summary.get('events', []):
            if evt.get('eventName') == 'RMA_CREATED':
                created_at = evt.get('eventDate')
                break
    
    updated_at = rma.get('lastUpdated')
    days_since = 0
    if updated_at:
        try:
            d = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
            days_since = (datetime.now(timezone.utc).date() - d.date()).days
        except: pass

    if store_url in store_counts:
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

        # Search Filtering
        add_row = True
        if search_query:
            q = search_query.lower()
            if (q not in str(rma_id).lower() and 
                q not in str(order_name).lower() and 
                q not in str(track_str).lower()):
                add_row = False
        
        if add_row:
            processed_rows.append({
                "RMA ID": rma_id,
                "Order": order_name,
                "Store URL": store_url,
                "Store Name": next((s['name'] for s in STORES if s['url'] == store_url), "Unknown"),
                "Status": status,
                "TrackingNumber": track_url if track_url else track_str,
                "DisplayTrack": track_str,
                "Created": str(created_at)[:10] if created_at else "N/A",
                "Updated": str(updated_at)[:10] if updated_at else "N/A",
                "Days since updated": days_since,
                "IsNoTrack": is_no_track,
                "IsFlagged": is_flagged,
                "shipment_id": shipment_id,
                "full_data": rma
            })

# --- Store Boxes with Timestamps ---
cols = st.columns(len(STORES))

for i, store in enumerate(STORES):
    c = store_counts[store['url']]
    with cols[i]:
        st.markdown(f"**{store['name']}**")
        
        def show_btn_and_time(label, status, key):
            if st.button(f"{label}\n{c[status]}", key=key): 
                handle_filter_click(store['url'], status)
            ts = get_last_sync(store['url'], status)
            if ts:
                st.markdown(f"<div class='sync-time'>Updated: {ts[11:]}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='sync-time'>-</div>", unsafe_allow_html=True)

        show_btn_and_time("Pending", "Pending", f"p_{i}")
        show_btn_and_time("Approved", "Approved", f"a_{i}")
        show_btn_and_time("Received", "Received", f"r_{i}")
        
        if st.button(f"No Track\n{c['NoTrack']}", key=f"n_{i}"): 
             handle_filter_click(store['url'], "NoTrack")
        if st.button(f"🚩 Flagged\n{c['Flagged']}", key=f"f_{i}"): 
             handle_filter_click(store['url'], "Flagged")

st.divider()

current_filter = st.session_state.filter_state
filter_desc = f"{current_filter['status']} Records"
if current_filter['store']:
    s_name = next((s['name'] for s in STORES if s['url'] == current_filter['store']), "Unknown")
    filter_desc += f" for {s_name}"

st.subheader(f"📋 {filter_desc}")

df = pd.DataFrame(processed_rows)

if not df.empty:
    if current_filter['store']:
        df = df[df['Store URL'] == current_filter['store']]
    
    f_stat = current_filter['status']
    if f_stat == "Pending": df = df[df['Status'] == 'Pending']
    elif f_stat == "Approved": df = df[df['Status'] == 'Approved']
    elif f_stat == "Received": df = df[df['Status'] == 'Received']
    elif f_stat == "NoTrack": df = df[df['IsNoTrack'] == True]
    elif f_stat == "Flagged": df = df[df['IsFlagged'] == True]

    if not df.empty:
        df = df.sort_values(by="Created", ascending=False)
        df = df.reset_index(drop=True)
        df.insert(0, "No", range(1, len(df) + 1))
        
        df['No'] = df['No'].astype(str)
        df['Days since updated'] = df['Days since updated'].astype(str)

        event = st.dataframe(
            df[["No", "Store Name", "RMA ID", "Order", "Status", "TrackingNumber", "Created", "Updated", "Days since updated"]],
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

        selected = st.session_state.main_table.get("selection", {}).get("rows", [])
        
        if selected:
            idx = selected[0]
            if idx < len(df):
                record = df.iloc[idx]
                show_rma_modal(record) 
    else:
        st.info("No records match the current filter.")
else:
    st.info("No records found in database. Click 'Sync All Data' to start.")
