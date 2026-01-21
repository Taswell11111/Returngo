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

# ==========================================
# 1. CONFIGURATION
# ==========================================
st.set_page_config(page_title="Levi's RMA Ops", layout="wide", page_icon="📤")

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
STORE_URL = "levis-sa.myshopify.com"
DB_FILE = "levis_cache.db"
DB_LOCK = threading.Lock()

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
        font-size: 16px;
        font-weight: bold;
    }
    div.stButton > button:hover {
        border-color: #c41230; 
        color: #c41230;
    }
    div[data-testid="stDialog"] {
        background-color: #1a1a1a;
        border: 1px solid #c41230;
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
                     (status TEXT, last_sync TIMESTAMP, PRIMARY KEY (status))''')
        conn.commit()
        conn.close()

def save_rma_to_db(rma_id, status, created_at, data, courier_status=None):
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
                         (str(rma_id), STORE_URL, status, created_at, json.dumps(data), now, courier_status))
            conn.commit()
            conn.close()
        except: pass

def update_courier_status_in_db(rma_id, status_text):
    with DB_LOCK:
        try:
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute("UPDATE rmas SET courier_status=? WHERE rma_id=?", (status_text, str(rma_id)))
            conn.commit()
            conn.close()
        except: pass

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
        c.execute("SELECT json_data, store_url, courier_status FROM rmas WHERE store_url=? AND status IN ('Pending', 'Approved', 'Received')", (STORE_URL,))
        rows = c.fetchall()
        conn.close()
    results = []
    for r in rows:
        data = json.loads(r[0])
        data['store_url'] = r[1]
        data['_local_courier_status'] = r[2]
        results.append(data)
    return results

def get_local_ids_for_status(status):
    with DB_LOCK:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT rma_id FROM rmas WHERE store_url=? AND status=?", (STORE_URL, status))
        rows = c.fetchall()
        conn.close()
    return {r[0] for r in rows}

def update_sync_log(status):
    with DB_LOCK:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        c.execute('''INSERT OR REPLACE INTO sync_logs (status, last_sync) VALUES (?, ?)''', (status, now))
        conn.commit()
        conn.close()

def get_last_sync(status):
    with DB_LOCK:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        try:
            c.execute("SELECT last_sync FROM sync_logs WHERE status=?", (status,))
            row = c.fetchone()
            return row[0] if row else None
        except: return None
        finally: conn.close()

def clear_db():
    with DB_LOCK:
        try:
            if os.path.exists(DB_FILE): os.remove(DB_FILE)
            init_db()
            return True
        except: return False

init_db()

# --- COURIER CHECK (Strict Top-Row Extraction) ---
def check_courier_status(tracking_number, rma_id=None):
    try:
        url = f"https://optimise.parcelninja.com/shipment/track/{tracking_number}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Authorization': f'Bearer {PARCEL_NINJA_TOKEN}'
        }
        res = requests.get(url, headers=headers, timeout=10)
        final_status = "Unknown"
        if res.status_code == 200:
            content = res.text
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

            clean_html = re.sub(r'<(script|style).*?</\1>', '', content, flags=re.DOTALL | re.IGNORECASE)
            rows = re.findall(r'<tr[^>]*>(.*?)</tr>', clean_html, re.DOTALL | re.IGNORECASE)
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
            if found_text: final_status = re.sub(r'\s+', ' ', found_text).strip()
            else:
                for kw in ["Courier Cancelled", "Booked Incorrectly", "Delivered", "Out For Delivery"]:
                    if re.search(re.escape(kw), clean_html, re.IGNORECASE):
                        final_status = kw; break
        elif res.status_code == 404: final_status = "Tracking Not Found"
        else: final_status = f"Error {res.status_code}"
        if rma_id: update_courier_status_in_db(rma_id, final_status)
        return final_status
    except Exception as e: return f"Check Failed: {str(e)[:20]}"

# ==========================================
# 3. BACKEND LOGIC
# ==========================================

def fetch_all_pages(session, headers, status):
    all_rmas = []; cursor = None
    while True:
        try:
            base_url = f"https://api.returngo.ai/rmas?status={status}&pagesize=50"
            url = f"{base_url}&cursor={cursor}" if cursor else base_url
            res = session.get(url, headers=headers, timeout=15)
            if res.status_code != 200: break
            data = res.json(); rmas = data.get("rmas", [])
            if not rmas: break
            all_rmas.extend(rmas)
            cursor = data.get("next_cursor")
            if not cursor: break
        except: break
    return all_rmas

def fetch_rma_detail(args):
    rma_summary = args; rma_id = rma_summary.get('rmaId')
    session = get_session(); headers = {"X-API-KEY": MY_API_KEY, "x-shop-name": STORE_URL}
    try:
        res = session.get(f"https://api.returngo.ai/rma/{rma_id}", headers=headers, timeout=15)
        if res.status_code == 200:
            data = res.json(); sum_data = data.get('rmaSummary', {})
            shipments = data.get('shipments', []); track_no = None
            for s in shipments:
                if s.get('trackingNumber'): track_no = s.get('trackingNumber'); break
            c_status = None
            if track_no:
                try: c_status = check_courier_status(track_no)
                except: pass
            save_rma_to_db(rma_id, sum_data.get('status', 'Unknown'), sum_data.get('createdAt'), data, c_status)
            data['_local_courier_status'] = c_status
            return data
    except: pass
    return None

def perform_sync(statuses=None):
    session = get_session(); headers = {"X-API-KEY": MY_API_KEY, "x-shop-name": STORE_URL}
    status_msg = st.empty(); status_msg.info("⏳ Starting Deep Sync...")
    active_summaries = []
    if statuses is None: statuses = ["Pending", "Approved", "Received"]
    for s in statuses:
        api_rmas = fetch_all_pages(session, headers, s)
        local_ids = get_local_ids_for_status(s)
        api_ids = {r.get('rmaId') for r in api_rmas}
        stale_ids = local_ids - api_ids
        active_summaries.extend(api_rmas)
        for stale_id in stale_ids: active_summaries.append({'rmaId': stale_id})
        update_sync_log(s)
    
    total = len(active_summaries)
    if total > 0:
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            list(executor.map(fetch_rma_detail, active_summaries))
    
    st.session_state['last_sync'] = datetime.now().strftime("%Y-%m-%d %H:%M")
    st.session_state['show_toast'] = True
    status_msg.success(f"✅ Sync Complete!")
    st.rerun()

def push_tracking_update(rma_id, shipment_id, tracking_number):
    session = get_session(); headers = {"X-API-KEY": MY_API_KEY, "x-shop-name": STORE_URL, "Content-Type": "application/json"}
    payload = {
        "status": "LabelCreated", "carrierName": "CourierGuy", "trackingNumber": tracking_number,
        "trackingURL": f"https://optimise.parcelninja.com/shipment/track/{tracking_number}",
        "labelURL": "https://sellerportal.dpworld.com/api/file-download?link=null"
    }
    try:
        res = session.put(f"https://api.returngo.ai/shipment/{shipment_id}", headers=headers, json=payload, timeout=10)
        if res.status_code == 200:
            fresh_res = session.get(f"https://api.returngo.ai/rma/{rma_id}", headers=headers, timeout=10)
            if fresh_res.status_code == 200:
                fresh_data = fresh_res.json(); summary = fresh_data.get('rmaSummary', {})
                save_rma_to_db(rma_id, summary.get('status', 'Approved'), summary.get('createdAt'), fresh_data)
            return True, "Success"
        return False, f"API Error {res.status_code}"
    except Exception as e: return False, str(e)

def push_comment_update(rma_id, comment_text):
    session = get_session(); headers = {"X-API-KEY": MY_API_KEY, "x-shop-name": STORE_URL, "Content-Type": "application/json"}
    payload = {"text": comment_text, "isPublic": False}
    try:
        res = session.post(f"https://api.returngo.ai/rma/{rma_id}/note", headers=headers, json=payload, timeout=10)
        if res.status_code in [200, 201]:
             fresh_res = session.get(f"https://api.returngo.ai/rma/{rma_id}", headers=headers, timeout=10)
             if fresh_res.status_code == 200:
                 fresh_data = fresh_res.json(); summary = fresh_data.get('rmaSummary', {})
                 save_rma_to_db(rma_id, summary.get('status', 'Approved'), summary.get('createdAt'), fresh_data)
             return True, "Success"
        return False, "API Error"
    except Exception as e: return False, str(e)

# ==========================================
# 4. FRONTEND UI
# ==========================================

# PERSISTENT TRIGGER FIX:
if 'modal_rma' not in st.session_state: st.session_state.modal_rma = None
if 'modal_action' not in st.session_state: st.session_state.modal_action = None

@st.dialog("Update Tracking")
def show_update_tracking_dialog(record):
    st.markdown(f"### Update Tracking for `{record['RMA ID']}`")
    with st.form("upd_track"):
        new_track = st.text_input("New Tracking Number", value=record['DisplayTrack'])
        if st.form_submit_button("Save Changes"):
            if not record['shipment_id']: st.error("No Shipment ID.")
            else:
                ok, msg = push_tracking_update(record['RMA ID'], record['shipment_id'], new_track)
                if ok: st.success("Updated!"); time.sleep(1); st.rerun()
                else: st.error(msg)

@st.dialog("View Timeline")
def show_timeline_dialog(record):
    st.markdown(f"### Timeline for `{record['RMA ID']}`")
    with st.expander("➕ Add Comment", expanded=False):
        with st.form("add_comm"):
            comment_text = st.text_area("New Note")
            if st.form_submit_button("Post Comment"):
                ok, msg = push_comment_update(record['RMA ID'], comment_text)
                if ok: st.success("Posted!"); st.rerun()
                else: st.error(msg)
    full = record['full_data']; timeline = full.get('comments', [])
    if not timeline: st.info("No timeline events found.")
    else:
        for t in timeline:
            d_str = t.get('datetime', '')[:16].replace('T', ' ')
            st.markdown(f"**{d_str}** | `{t.get('triggeredBy', 'System')}`\n> {t.get('htmlText', '')}")
            st.divider()

# HANDLE MODAL TRIGGER (AFTER RERUN)
# Explicit check 'is not None' for Pandas Series truth value fix
if st.session_state.modal_rma is not None:
    current_rma = st.session_state.modal_rma
    current_act = st.session_state.modal_action
    
    # CLEAR STATE IMMEDIATELY
    st.session_state.modal_rma = None
    st.session_state.modal_action = None
    
    if current_act == 'edit': show_update_tracking_dialog(current_rma)
    elif current_act == 'view': show_timeline_dialog(current_rma)

if 'filter_status' not in st.session_state: st.session_state.filter_status = "All"
if st.session_state.get('show_toast'): st.toast("✅ API Sync Complete!", icon="🔄"); st.session_state['show_toast'] = False

def set_filter(f):
    st.session_state.filter_status = f
    if f in ["Pending", "Approved", "Received"]: perform_sync(statuses=[f])
    else: st.rerun()

# --- Header ---
col1, col2 = st.columns([3, 1])
with col1:
    st.title("Levi's ReturnGO Ops Dashboard")
    st.markdown(f"**CONNECTED TO:** {STORE_URL.upper()} | **LAST SYNC:** :green[{st.session_state.get('last_sync', 'N/A')}]")
    search_query = st.text_input("🔍 Search Order, RMA, or Tracking", placeholder="Type to search...")
with col2:
    if st.button("🔄 Sync All Data", type="primary"): perform_sync()
    if st.button("🗑️ Reset Cache", type="secondary"):
        if clear_db(): st.success("Cache cleared!"); st.rerun()

# --- Data Processing ---
raw_data = get_all_active_from_db(); processed_rows = []
counts = {"Pending": 0, "Approved": 0, "Received": 0, "NoTrack": 0, "Flagged": 0}
for rma in raw_data:
    summary = rma.get('rmaSummary', {}); shipments = rma.get('shipments', []); comments = rma.get('comments', [])
    status = summary.get('status', 'Unknown'); rma_id = summary.get('rmaId', 'N/A'); order_name = summary.get('order_name', 'N/A')
    track_nums = [s.get('trackingNumber') for s in shipments if s.get('trackingNumber')]
    track_str = ", ".join(track_nums) if track_nums else ""
    shipment_id = shipments[0].get('shipmentId') if shipments else None
    local_status = rma.get('_local_courier_status', '')
    track_link_url = f"https://portal.thecourierguy.co.za/track?ref={track_nums[0]}" if track_nums else ""
    
    created_at = summary.get('createdAt')
    if not created_at:
        for evt in summary.get('events', []):
            if evt.get('eventName') == 'RMA_CREATED': created_at = evt.get('eventDate'); break
    u_at = rma.get('lastUpdated'); d_since = 0
    if u_at:
        try: d_since = (datetime.now(timezone.utc).date() - datetime.fromisoformat(u_at.replace('Z', '+00:00')).date()).days
        except: pass
    if status in counts: counts[status] += 1
    is_nt = (status == "Approved" and not track_str); is_fg = any("flagged" in c.get('htmlText', '').lower() for c in comments)
    if is_nt: counts["NoTrack"] += 1
    if is_fg: counts["Flagged"] += 1
    
    if not search_query or (search_query.lower() in str(rma_id).lower() or search_query.lower() in str(order_name).lower() or search_query.lower() in str(track_str).lower()):
        processed_rows.append({
            "No": "", "RMA ID": rma_id, "Order": order_name, "Status": status,
            "Tracking Number": track_link_url, "Tracking Status": local_status,
            "Created": str(created_at)[:10] if created_at else "N/A",
            "Updated": str(u_at)[:10] if u_at else "N/A", "Days since updated": str(d_since),
            "Update Tracking Number": "", "View Timeline": "", 
            "DisplayTrack": track_str, "shipment_id": shipment_id, "full_data": rma, "is_nt": is_nt, "is_fg": is_fg
        })

# --- Metrics (ALL CAPS) ---
b1, b2, b3, b4, b5 = st.columns(5)
def get_status_time(s):
    ts = get_last_sync(s)
    return f"<div class='sync-time'>UPDATED: {ts[11:] if ts else '-'}</div>"
with b1: 
    if st.button(f"PENDING\n{counts['Pending']}"): set_filter("Pending")
    st.markdown(get_status_time("Pending"), unsafe_allow_html=True)
with b2: 
    if st.button(f"APPROVED\n{counts['Approved']}"): set_filter("Approved")
    st.markdown(get_status_time("Approved"), unsafe_allow_html=True)
with b3: 
    if st.button(f"RECEIVED\n{counts['Received']}"): set_filter("Received")
    st.markdown(get_status_time("Received"), unsafe_allow_html=True)
with b4: 
    if st.button(f"NO TRACKING\n{counts['NoTrack']} "): set_filter("NoTrack")
with b5: 
    if st.button(f"🚩 FLAGGED\n{counts['Flagged']} "): set_filter("Flagged")

st.divider()
df_view = pd.DataFrame(processed_rows)
if not df_view.empty:
    f_stat = st.session_state.filter_status
    if f_stat == "Pending": display_df = df_view[df_view['Status'] == 'Pending']
    elif f_stat == "Approved": display_df = df_view[df_view['Status'] == 'Approved']
    elif f_stat == "Received": display_df = df_view[df_view['Status'] == 'Received']
    elif f_stat == "NoTrack": display_df = df_view[df_view['is_nt'] == True]
    elif f_stat == "Flagged": display_df = df_view[df_view['is_fg'] == True]
    else: display_df = df_view

    if not display_df.empty:
        display_df = display_df.sort_values(by="Created", ascending=False).reset_index(drop=True)
        display_df['No'] = (display_df.index + 1).astype(str)
        display_df['Days since updated'] = display_df['Days since updated'].astype(str)

        # RENDER TABLE WITH TEXT TRIGGERS (SELECTBOX)
        edited = st.data_editor(
            display_df[["No", "RMA ID", "Order", "Status", "Tracking Number", "Tracking Status", "Created", "Updated", "Days since updated", "Update Tracking Number", "View Timeline"]],
            use_container_width=True, height=700, hide_index=True, key="main_table",
            column_config={
                "No": st.column_config.TextColumn("No", width="small"),
                "RMA ID": st.column_config.TextColumn("RMA ID", width="small"),
                "Order": st.column_config.TextColumn("Order", width="small"),
                "Tracking Number": st.column_config.LinkColumn("Tracking Number", display_text=r"ref=(.*)", width="medium"),
                "Tracking Status": st.column_config.TextColumn("Tracking Status", width="medium"),
                "Update Tracking Number": st.column_config.SelectboxColumn("Update Tracking Number", options=["", "Edit"], width="small"),
                "View Timeline": st.column_config.SelectboxColumn("View Timeline", options=["", "View"], width="small"),
                "Days since updated": st.column_config.TextColumn("Days since updated", width="small")
            },
            disabled=["No", "RMA ID", "Order", "Status", "Tracking Number", "Tracking Status", "Created", "Updated", "Days since updated"]
        )
        
        # CAPTURE SELECTION VIA edited_rows (STRICT ROUTING)
        if "main_table" in st.session_state:
            edits = st.session_state.main_table.get("edited_rows", {})
            for row_idx, changes in edits.items():
                idx = int(row_idx)
                # Check EXACTLY which key changed in the specific row
                if "Update Tracking Number" in changes and changes["Update Tracking Number"] == "Edit":
                    st.session_state.modal_rma = display_df.iloc[idx]
                    st.session_state.modal_action = 'edit'
                    st.session_state.main_table["edited_rows"] = {} # Flush state
                    st.rerun()
                elif "View Timeline" in changes and changes["View Timeline"] == "View":
                    st.session_state.modal_rma = display_df.iloc[idx]
                    st.session_state.modal_action = 'view'
                    st.session_state.main_table["edited_rows"] = {} # Flush state
                    st.rerun()
    else: st.info("No matching records found.")
else: st.warning("Database empty. Click Sync All Data to start.")
