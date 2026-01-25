import streamlit as st
import sqlite3
import requests
import json
import pandas as pd
import os
import threading
import time
import re
from datetime import datetime, timedelta, timezone
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import concurrent.futures
from returngo_api import api_url, RMA_COMMENT_PATH

# ==========================================
# 1. CONFIGURATION
# ==========================================
st.set_page_config(page_title="Bounty Apparel ReturnGo RMAs", layout="wide", page_icon="🔄️")

# ACCESS SECRETS
try:
    MY_API_KEY = st.secrets["RETURNGO_API_KEY"]
except (FileNotFoundError, KeyError):
    MY_API_KEY = os.environ.get("RETURNGO_API_KEY")

try:
    PARCEL_NINJA_TOKEN = st.secrets["PARCEL_NINJA_TOKEN"]
except (FileNotFoundError, KeyError):
    PARCEL_NINJA_TOKEN = os.environ.get("PARCEL_NINJA_TOKEN", "")

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
        color: #9ca3af;
        text-align: center;
        margin-top: 0;
        margin-bottom: 6px;
        text-transform: uppercase;
        letter-spacing: 0.03em;
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
        try:
            c.execute("SELECT courier_status FROM rmas LIMIT 1")
        except sqlite3.OperationalError:
            try:
                c.execute("ALTER TABLE rmas ADD COLUMN courier_status TEXT")
            except: pass

        c.execute('''CREATE TABLE IF NOT EXISTS rmas
                     (rma_id TEXT PRIMARY KEY, store_url TEXT, status TEXT, 
                      created_at TEXT, json_data TEXT, last_fetched TIMESTAMP, courier_status TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS sync_logs
                     (store_url TEXT, status TEXT, last_sync TIMESTAMP, PRIMARY KEY (store_url, status))''')
        conn.commit()
        conn.close()

def save_rma_to_db(rma_id, store_url, status, created_at, data, courier_status=None):
    with DB_LOCK:
        try:
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            now = datetime.now().isoformat()
            if courier_status is None:
                c.execute("SELECT courier_status FROM rmas WHERE rma_id=?", (str(rma_id),))
                row = c.fetchone()
                if row: courier_status = row[0]
            
            c.execute('''INSERT OR REPLACE INTO rmas (rma_id, store_url, status, created_at, json_data, last_fetched, courier_status)
                         VALUES (?, ?, ?, ?, ?, ?, ?)''', 
                         (str(rma_id), store_url, status, created_at, json.dumps(data), now, courier_status))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"DB Error saving {rma_id}: {e}")

def update_courier_status_in_db(rma_id, status_text):
    with DB_LOCK:
        try:
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute("UPDATE rmas SET courier_status=? WHERE rma_id=?", (status_text, str(rma_id)))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"DB Error updating status {rma_id}: {e}")

def get_rma_from_db(rma_id):
    with DB_LOCK:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT json_data, last_fetched, courier_status FROM rmas WHERE rma_id=?", (str(rma_id),))
        row = c.fetchone()
        conn.close()
    if row:
        data = json.loads(row[0])
        data['_local_courier_status'] = row[2] 
        return data, datetime.fromisoformat(row[1])
    return None, None

def get_all_active_from_db():
    with DB_LOCK:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT json_data, store_url, courier_status FROM rmas WHERE status IN ('Pending', 'Approved', 'Received')")
        rows = c.fetchall()
        conn.close()
    results = []
    for r in rows:
        data = json.loads(r[0])
        data['store_url'] = r[1]
        data['_local_courier_status'] = r[2]
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

def delete_rmas_from_db(rma_ids):
    if not rma_ids:
        return
    with DB_LOCK:
        with sqlite3.connect(DB_FILE) as conn:
            conn.executemany("DELETE FROM rmas WHERE rma_id=?", [(str(rma_id),) for rma_id in rma_ids])

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

# --- COURIER CHECK (Strict Positional Parsing) ---
def check_courier_status(tracking_number, rma_id=None):
    try:
        url = "https://optimise.parcelninja.com/shipment/track"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        }
        if PARCEL_NINJA_TOKEN:
            headers['Authorization'] = f'Bearer {PARCEL_NINJA_TOKEN}'
        res = requests.get(url, headers=headers, params={"WaybillNo": tracking_number}, timeout=10)
        final_status = "Unknown"
        
        if res.status_code == 200:
            content = res.text
            # 1. Try JSON First
            try:
                data = res.json()
                if 'history' in data and len(data['history']) > 0:
                    final_status = data['history'][0].get('status') or data['history'][0].get('description')
                else:
                    final_status = data.get('status') or data.get('currentStatus')
                if final_status:
                     if rma_id: update_courier_status_in_db(rma_id, final_status)
                     return final_status
            except: pass 

            # 2. Regex First Data Row Parsing
            clean_html = re.sub(r'<(script|style).*?</\1>', '', content, flags=re.DOTALL | re.IGNORECASE)
            history_section = re.search(r'<table[^>]*?tracking-history.*?>(.*?)</table>', clean_html, re.DOTALL | re.IGNORECASE)
            if not history_section:
                history_section = re.search(r'<tbody[^>]*?>(.*?)</tbody>', clean_html, re.DOTALL | re.IGNORECASE)
            
            target_content = history_section.group(1) if history_section else clean_html
            rows = re.findall(r'<tr[^>]*>(.*?)</tr>', target_content, re.DOTALL | re.IGNORECASE)
            
            found_text = None
            for r_html in rows:
                if '<th' in r_html.lower(): continue
                cells = re.findall(r'<td[^>]*>(.*?)</td>', r_html, re.DOTALL | re.IGNORECASE)
                if cells:
                    cleaned_cells = [re.sub(r'<[^>]+>', '', c).strip() for c in cells]
                    cleaned_cells = [c for c in cleaned_cells if c]
                    if cleaned_cells:
                        found_text = " - ".join(cleaned_cells)
                        break

            if found_text:
                final_status = re.sub(r'\s+', ' ', found_text).strip()
            else:
                kw_list = ["Courier Cancelled", "Booked Incorrectly", "Delivered", "Out For Delivery"]
                for kw in kw_list:
                    if re.search(re.escape(kw), clean_html, re.IGNORECASE):
                        final_status = kw
                        break

        elif res.status_code == 404: final_status = "Tracking Not Found"
        else: final_status = f"Error {res.status_code}"
            
        if rma_id: update_courier_status_in_db(rma_id, final_status)
        return final_status
    except Exception as e:
        return f"Check Failed: {str(e)[:20]}"

# ==========================================
# 3. BACKEND LOGIC
# ==========================================

def fetch_all_pages(session, headers, status):
    all_rmas = []
    page = 1
    cursor = None
    while True:
        try:
            base_url = f"{api_url('/rmas')}?status={status}&pagesize=50"
            url = f"{base_url}&cursor={cursor}" if cursor else base_url
            res = session.get(url, headers=headers, timeout=15)
            if res.status_code != 200: break
            data = res.json()
            rmas = data.get("rmas", [])
            if not rmas: break
            all_rmas.extend(rmas)
            cursor = data.get("next_cursor")
            if not cursor or page > 100: break
            page += 1
        except: break
    return all_rmas

def fetch_rma_detail(args):
    rma_summary, store_url, force_refresh = args
    rma_id = rma_summary.get('rmaId')
    local_data, last_f = get_rma_from_db(rma_id)
    if not force_refresh and local_data and last_f:
        if (datetime.now() - last_f) < timedelta(hours=CACHE_EXPIRY_HOURS):
            return local_data
    session = get_session()
    headers = {"X-API-KEY": MY_API_KEY, "x-shop-name": store_url}
    try:
        res = session.get(api_url(f"/rma/{rma_id}"), headers=headers, timeout=15)
        if res.status_code == 200:
            data = res.json()
            fresh_sum = data.get('rmaSummary', {})
            shipments = data.get('shipments', [])
            track_no = None
            for s in shipments:
                if s.get('trackingNumber'): 
                    track_no = s.get('trackingNumber')
                    break
            
            c_status = local_data.get('_local_courier_status') if local_data else None
            if track_no and (not c_status or c_status == "Unknown"):
                try: c_status = check_courier_status(track_no)
                except: pass
            
            save_rma_to_db(rma_id, store_url, fresh_sum.get('status', 'Unknown'), fresh_sum.get('createdAt'), data, c_status)
            data['_local_courier_status'] = c_status
            return data
    except: pass
    return None

def perform_sync(target_store=None, target_status=None):
    session = get_session()
    status_msg = st.empty()
    status_msg.info("⏳ Connecting to API...")
    tasks = [] 
    stores_to_sync = [target_store] if target_store else STORES
    statuses = [target_status] if target_status and target_status in ["Pending", "Approved", "Received"] else ["Pending", "Approved", "Received"]
    if target_status == "NoTrack": statuses = ["Approved"]
    if target_status == "Flagged": statuses = ["Pending", "Approved"]

    total_found = 0
    list_bar = None
    if not target_store: list_bar = st.progress(0, text="Fetching Lists from ReturnGO...")
    for i, store in enumerate(stores_to_sync):
        headers = {"X-API-KEY": MY_API_KEY, "x-shop-name": store['url']}
        for s in statuses:
            api_rmas = fetch_all_pages(session, headers, s)
            local_ids = get_local_ids_for_status(store['url'], s)
            api_ids = {r.get('rmaId') for r in api_rmas}
            stale_ids = local_ids - api_ids
            if stale_ids:
                delete_rmas_from_db(stale_ids)
            for r in api_rmas: tasks.append((r, store['url'], True))
            total_found += len(api_rmas)
            update_sync_log(store['url'], s)
        if list_bar: list_bar.progress((i + 1) / len(stores_to_sync), text=f"Fetched {store['name']}")
    if list_bar: list_bar.empty()

    status_msg.info(f"⏳ Syncing {total_found} records...")
    if total_found > 0:
        bar = st.progress(0, text="Downloading Details...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(fetch_rma_detail, task) for task in tasks]
            
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                # We can do something with the result here if needed, e.g., future.result()
                bar.progress((i + 1) / total_found, text=f"Syncing: {i + 1}/{total_found}")
        bar.empty()
                
    st.session_state['last_sync'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state['show_toast'] = True
    status_msg.success(f"✅ Sync Complete!")
    st.cache_data.clear()
    st.rerun()

def push_tracking_update(rma_id, shipment_id, tracking_number, store_url):
    session = get_session()
    headers = {"X-API-KEY": MY_API_KEY, "x-shop-name": store_url, "Content-Type": "application/json"}
    payload = {
        "status": "LabelCreated", "carrierName": "CourierGuy", "trackingNumber": tracking_number,
        "trackingURL": f"https://optimise.parcelninja.com/shipment/track?WaybillNo={tracking_number}",
        "labelURL": "https://sellerportal.dpworld.com/api/file-download?link=null"
    }
    try:
        res = session.put(api_url(f"/shipment/{shipment_id}"), headers=headers, json=payload, timeout=10)
        if res.status_code == 200:
            fresh_res = session.get(api_url(f"/rma/{rma_id}"), headers=headers, timeout=10)
            if fresh_res.status_code == 200:
                fresh_data = fresh_res.json()
                summary = fresh_data.get('rmaSummary', {})
                save_rma_to_db(rma_id, store_url, summary.get('status', 'Approved'), summary.get('createdAt'), fresh_data)
            return True, "Success"
        return False, f"API Error {res.status_code}: {res.text}"
    except Exception as e: return False, str(e)

def push_comment_update(rma_id, comment_text, store_url):
    session = get_session()
    headers = {"X-API-KEY": MY_API_KEY, "x-shop-name": store_url, "Content-Type": "application/json"}
    payload = { "text": comment_text, "isPublic": False }
    try:
        res = session.post(api_url(RMA_COMMENT_PATH.format(rma_id=rma_id)), headers=headers, json=payload, timeout=10)
        if res.status_code in [200, 201]:
             fresh_res = session.get(api_url(f"/rma/{rma_id}"), headers=headers, timeout=10)
             if fresh_res.status_code == 200:
                 fresh_data = fresh_res.json()
                 summary = fresh_data.get('rmaSummary', {})
                 save_rma_to_db(rma_id, store_url, summary.get('status', 'Approved'), summary.get('createdAt'), fresh_data)
                 return True, "Success"
             return False, f"Detail fetch failed {fresh_res.status_code}: {fresh_res.text}"
        return False, f"API Error {res.status_code}: {res.text}"
    except Exception as e: return False, str(e)

# ==========================================
# 4. FRONTEND UI LOGIC
# ==========================================

if 'filter_state' not in st.session_state:
    st.session_state.filter_state = {"store": None, "status": "All"}
if st.session_state.get('show_toast'):
    st.toast("✅ API Sync Complete!", icon="🔄")
    st.session_state['show_toast'] = False
if st.session_state.get('show_update_toast'):
    st.toast("✅ UI refreshed from database.", icon="🔄")
    st.session_state['show_update_toast'] = False

@st.dialog("Update Tracking")
def show_update_tracking_dialog(record):
    st.markdown(f"### Update Tracking for `{record['RMA ID']}`")
    raw_track = record['DisplayTrack']
    with st.form("update_track_form"):
        new_track = st.text_input("New Tracking Number", value=raw_track)
        if st.form_submit_button("Save Changes"):
            if not record['shipment_id']: st.error("No Shipment ID.")
            else:
                ok, msg = push_tracking_update(record['RMA ID'], record['shipment_id'], new_track, record['Store URL'])
                if ok:
                    st.success("Tracking Updated!")
                    time.sleep(1)
                    st.rerun()
                else: st.error(msg)

@st.dialog("View Timeline")
def show_timeline_dialog(record):
    st.markdown(f"### Timeline for `{record['RMA ID']}`")
    with st.expander("➕ Add Comment", expanded=False):
        with st.form("add_comment_form"):
            comment_text = st.text_area("New Note")
            if st.form_submit_button("Post Comment"):
                ok, msg = push_comment_update(record['RMA ID'], comment_text, record['Store URL'])
                if ok:
                    st.success("Comment Posted!")
                    st.rerun()
                else: st.error(msg)
    full = record['full_data']
    timeline = full.get('comments', [])
    if not timeline: st.info("No timeline events.")
    else:
        for t in timeline:
            d_str = t.get('datetime', '')[:16].replace('T', ' ')
            st.markdown(f"**{d_str}** | `{t.get('triggeredBy', 'System')}`\n> {t.get('htmlText', '')}")
            st.divider()

def handle_filter_click(store_url, status):
    st.session_state.filter_state = {"store": store_url, "status": status}
    if status in ["Pending", "Approved", "Received"]:
        store_obj = next((s for s in STORES if s['url'] == store_url), None)
        perform_sync(store_obj, status)
    else: st.rerun()

# --- Header ---
col1, col2 = st.columns([3, 1])
with col1:
    st.title("Bounty Apparel ReturnGo RMAs 🔄️")
    search_query = st.text_input("🔍 Search Order, RMA, or Tracking", placeholder="Type to search...")
with col2:
    sync_col, update_col = st.columns(2)
    with sync_col:
        if st.button("🔄 Sync All Data", type="primary"):
            perform_sync()
    with update_col:
        if st.button("🔄 Update All", type="secondary"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.session_state.pop("main_table", None)
            st.session_state['show_update_toast'] = True
            st.rerun()
    if st.button("🗑️ Reset Cache", type="secondary"):
        if clear_db():
            st.cache_data.clear()
            st.cache_resource.clear()
            st.session_state.filter_state = {"store": None, "status": "All"}
            st.session_state.pop("last_sync", None)
            st.session_state.pop("main_table", None)
            st.success("Cache cleared! Data will reload on next sync.")
            st.rerun()
        else: st.error("DB might be locked.")

# --- Process Data ---
raw_data = get_all_active_from_db()
processed_rows = []
store_counts = {s['url']: {"Pending": 0, "Approved": 0, "Received": 0, "NoTrack": 0, "Flagged": 0} for s in STORES}
for rma in raw_data:
    store_url = rma.get('store_url')
    if not store_url: continue 
    summary = rma.get('rmaSummary', {}); shipments = rma.get('shipments', []); comments = rma.get('comments', [])
    status = summary.get('status', 'Unknown'); rma_id = summary.get('rmaId', 'N/A'); order_name = summary.get('order_name', 'N/A')
    track_nums = [s.get('trackingNumber') for s in shipments if s.get('trackingNumber')]
    track_str = ", ".join(track_nums) if track_nums else ""
    shipment_id = shipments[0].get('shipmentId') if shipments else None
    local_status = rma.get('_local_courier_status', '')
    
    track_link_url = f"https://optimise.parcelninja.com/shipment/track?WaybillNo={track_nums[0]}" if track_nums else ""
    
    created_at = summary.get('createdAt')
    if not created_at:
        for evt in summary.get('events', []):
            if evt.get('eventName') == 'RMA_CREATED': created_at = evt.get('eventDate'); break
    u_at = rma.get('lastUpdated'); d_since = 0
    if u_at:
        try: d_since = (datetime.now(timezone.utc).date() - datetime.fromisoformat(u_at.replace('Z', '+00:00')).date()).days
        except: pass
    if store_url in store_counts:
        if status in store_counts[store_url]: store_counts[store_url][status] += 1
        is_nt = False
        if status == "Approved" and not track_str: store_counts[store_url]["NoTrack"] += 1; is_nt = True
        is_fg = False
        if any("flagged" in c.get('htmlText', '').lower() for c in comments): store_counts[store_url]["Flagged"] += 1; is_fg = True
        add_row = True
        if search_query:
            q = search_query.lower()
            if (q not in str(rma_id).lower() and q not in str(order_name).lower() and q not in str(track_str).lower()): add_row = False
        if add_row:
            processed_rows.append({
                "No": "", "Store Name": next((s['name'] for s in STORES if s['url'] == store_url), "Unknown"),
                "RMA ID": rma_id, "Order": order_name, "Status": status, "Store URL": store_url,
                "TrackingNumber": track_link_url, "TrackingStatus": local_status,
                "Created": str(created_at)[:10] if created_at else "N/A",
                "Updated": str(u_at)[:10] if u_at else "N/A", "Days": str(d_since),
                "Update TrackingNumber": False, "View Timeline": False,
                "DisplayTrack": track_str, "shipment_id": shipment_id, "full_data": rma, "is_nt": is_nt, "is_fg": is_fg
            })

# --- Store Boxes (ALL CAPS) ---
cols = st.columns(len(STORES))
for i, store in enumerate(STORES):
    c = store_counts[store['url']]
    with cols[i]:
        st.markdown(f"**{store['name'].upper()}**")
        def show_btn(label, stat, key):
            ts = get_last_sync(store['url'], stat)
            st.markdown(
                f"<div class='sync-time'>Updated: {ts[11:] if ts else '-'}</div>",
                unsafe_allow_html=True,
            )
            if st.button(f"{label}\n{c[stat]}", key=key):
                handle_filter_click(store['url'], stat)
        show_btn("Pending", "Pending", f"p_{i}"); show_btn("Approved", "Approved", f"a_{i}"); show_btn("Received", "Received", f"r_{i}")
        if st.button(f"No Track\n{c['NoTrack']}", key=f"n_{i}"): handle_filter_click(store['url'], "NoTrack")
        if st.button(f"🚩 Flagged\n{c['Flagged']}", key=f"f_{i}"): handle_filter_click(store['url'], "Flagged")

st.divider()
cur = st.session_state.filter_state
df_view = pd.DataFrame(processed_rows)
if not df_view.empty:
    if cur['store']: df_view = df_view[df_view['Store URL'] == cur['store']]
    f_stat = cur['status']
    if f_stat == "Pending": df_view = df_view[df_view['Status'] == 'Pending']
    elif f_stat == "Approved": df_view = df_view[df_view['Status'] == 'Approved']
    elif f_stat == "Received": df_view = df_view[df_view['Status'] == 'Received']
    elif f_stat == "NoTrack": df_view = df_view[df_view['is_nt'] == True]
    elif f_stat == "Flagged": df_view = df_view[df_view['is_fg'] == True]

    if not df_view.empty:
        df_view = df_view.sort_values(by="Created", ascending=False).reset_index(drop=True)
        df_view['No'] = range(1, len(df_view) + 1); df_view['No'] = df_view['No'].astype(str)
        
        # Display Table (Updated to fit container and remove Fetch Status)
        edited = st.data_editor(
            df_view[["No", "Store Name", "RMA ID", "Order", "Status", "TrackingNumber", "TrackingStatus", "Created", "Updated", "Days", "Update TrackingNumber", "View Timeline"]],
            use_container_width=True, height=700, hide_index=True, key="main_table",
            column_config={
                "No": st.column_config.TextColumn("No", width="small"),
                "Store Name": st.column_config.TextColumn("Store Name", width="small"),
                "RMA ID": st.column_config.TextColumn("RMA ID", width="small"),
                "Order": st.column_config.TextColumn("Order", width="medium"),
                "TrackingNumber": st.column_config.LinkColumn("Tracking Number", display_text=r"ref=(.*)", width="medium"),
                "TrackingStatus": st.column_config.TextColumn("Tracking Status", width="medium"),
                "Update TrackingNumber": st.column_config.CheckboxColumn("Edit Tracking Number", width="small"),
                "View Timeline": st.column_config.CheckboxColumn("View Timeline", width="small"),
                "Days": st.column_config.TextColumn("Days", width="small")
            },
            disabled=["No", "Store Name", "RMA ID", "Order", "Status", "TrackingNumber", "TrackingStatus", "Created", "Updated", "Days"]
        )
        
        # Handle Action Logic
        # Checking for changes in the data_editor state
        for idx, row in edited.iterrows():
            # If "Edit Tracking Number" checkbox is ticked
            if row["Update TrackingNumber"]:
                show_update_tracking_dialog(df_view.iloc[idx])
                # We do not rerun immediately to allow the dialog to stay open, 
                # but the form inside will trigger a rerun.
                break
            
            # If "View Timeline" checkbox is ticked
            if row["View Timeline"]:
                show_timeline_dialog(df_view.iloc[idx])
                break
                
    else: st.info("No matching records.")
else: st.info("Database empty. Click 'Sync All Data'.")
