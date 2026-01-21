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
    # Fallback for local testing or environment variable
    MY_API_KEY = os.environ.get("RETURNGO_API_KEY")

if not MY_API_KEY:
    st.error("API Key not found! Please set 'RETURNGO_API_KEY' in secrets or env vars.")
    st.stop()

# PARCEL NINJA TOKEN (TASWELL) - Shared
PARCEL_NINJA_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbiI6IjJhMDI4MWNhNTBmNDEwOTRiZTkyNzdhNTQ0MDZhZGRkODMyOGExODhhYmNiZGViMiIsIm5iZiI6MTc2ODk1ODUwMSwiZXhwIjoxODYzNjUyOTAxLCJpYXQiOjE3Njg5NTg1MDEsImlzcyI6Imh0dHBzOi8vb3B0aW1pc2UucGFyY2VsbmluamEuY29tIiwiYXVkIjoiaHR0cHM6Ly9vcHRpbWlzZS5wYXJjZWxuaW5qYS5jb20ifQ.lgAi9s2INGKrzGYb3Qn_PY6N1ekh3fSBP7JBgMhX0Pk"

CACHE_EXPIRY_HOURS = 4
STORE_URL = "levis-sa.myshopify.com"
DB_FILE = "levis_cache.db"
# LOCK FOR THREAD-SAFE DB WRITES
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
        border-color: #c41230; /* Levi's Red */
        color: #c41230;
    }
    div.stButton > button:focus {
        border-color: #c41230;
        color: #c41230;
        background-color: #2d1b1b;
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
        
        # Check if column exists, if not migrate
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
            
            # Preserve existing courier_status if not provided
            if courier_status is None:
                c.execute("SELECT courier_status FROM rmas WHERE rma_id=?", (str(rma_id),))
                row = c.fetchone()
                if row:
                    courier_status = row[0]
            
            c.execute('''INSERT OR REPLACE INTO rmas (rma_id, store_url, status, created_at, json_data, last_fetched, courier_status)
                         VALUES (?, ?, ?, ?, ?, ?, ?)''', 
                         (str(rma_id), STORE_URL, status, created_at, json.dumps(data), now, courier_status))
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
        c.execute('''INSERT OR REPLACE INTO sync_logs (status, last_sync)
                     VALUES (?, ?)''', (status, now))
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
    cursor = None
    page_count = 1
    
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
            
            page_count += 1
            if page_count > 100: break
        except: break
    return all_rmas

def fetch_rma_detail(args):
    # In parallel executor, args is just the rma_summary dict usually
    # Or sometimes a tuple if we structured it that way. 
    # For levis_web, previous structure was just passing rma_summary object.
    rma_summary = args
    rma_id = rma_summary.get('rmaId')
    
    # We are usually forcing refresh in sync logic, so we skip cache check for sync details
    
    session = get_session()
    headers = {"X-API-KEY": MY_API_KEY, "x-shop-name": STORE_URL}
    try:
        res = session.get(f"https://api.returngo.ai/rma/{rma_id}", headers=headers, timeout=15)
        if res.status_code == 200:
            data = res.json()
            
            fresh_summary = data.get('rmaSummary', {})
            true_status = fresh_summary.get('status', 'Unknown')
            true_created = fresh_summary.get('createdAt')
            
            # --- AUTO-CHECK COURIER LOGIC (One-Time) ---
            local_data, _ = get_rma_from_db(rma_id)
            current_courier_status = local_data.get('_local_courier_status') if local_data else None
            
            shipments = data.get('shipments', [])
            tracking_number = None
            for s in shipments:
                if s.get('trackingNumber'):
                    tracking_number = s.get('trackingNumber')
                    break
            
            final_courier_status = current_courier_status
            
            # If we have a tracking number AND no status yet, fetch it now.
            if tracking_number and (not current_courier_status or current_courier_status == "Unknown"):
                try:
                    final_courier_status = check_courier_status(tracking_number)
                except: pass
            
            save_rma_to_db(rma_id, true_status, true_created, data, final_courier_status)
            data['_local_courier_status'] = final_courier_status
            return data
    except: pass
    return None

def perform_sync(statuses=None):
    session = get_session()
    headers = {"X-API-KEY": MY_API_KEY, "x-shop-name": STORE_URL}
    
    status_msg = st.empty()
    status_msg.info("⏳ Starting Deep Sync... fetching all pages.")
    
    active_summaries = []
    
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
            
        update_sync_log(s)
    
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

def push_comment_update(rma_id, comment_text):
    session = get_session()
    headers = {"X-API-KEY": MY_API_KEY, "x-shop-name": STORE_URL, "Content-Type": "application/json"}
    
    # ReturnGO typically uses POST /rma/{id}/comment or similar endpoint
    # Adjust payload based on specific API docs if different
    payload = {
        "text": comment_text,
        "isPublic": False # Usually internal notes by default
    }
    
    try:
        # Note: Endpoint might vary. Assuming standard structure based on previous interactions.
        # If this 404s, we need to check docs for exact 'add note' endpoint.
        res = session.post(f"https://api.returngo.ai/rma/{rma_id}/note", headers=headers, json=payload, timeout=10)
        
        if res.status_code in [200, 201]:
             # Refresh data
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

# --- COURIER CHECK (First-Row Priority Regex Logic) ---
def check_courier_status(tracking_number, rma_id=None):
    try:
        url = f"https://optimise.parcelninja.com/shipment/track/{tracking_number}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Authorization': f'Bearer {PARCEL_NINJA_TOKEN}',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
        }
        
        # 8s timeout to prevent hanging the sync
        res = requests.get(url, headers=headers, timeout=8)
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
                     return f"API: {final_status}"
            except: pass 

            # 2. Regex "First Row" Logic
            # Goal: Find the first <tr> that is NOT the header, and grab its text.
            
            # Remove scripts/styles
            clean_html = re.sub(r'<(script|style).*?</\1>', '', content, flags=re.DOTALL)
            
            # Find all <tr> tags
            rows = re.findall(r'<tr[^>]*>(.*?)</tr>', clean_html, re.DOTALL | re.IGNORECASE)
            
            found_text = None
            
            # Usually row 0 is header, row 1 is latest data
            if len(rows) > 1:
                target_row = rows[1] 
                
                # Extract columns <td>...</td>
                cols = re.findall(r'<td[^>]*>(.*?)</td>', target_row, re.DOTALL | re.IGNORECASE)
                
                if cols:
                    # Clean up HTML tags inside cells
                    row_text = [re.sub(r'<[^>]+>', '', c).strip() for c in cols]
                    row_text = [t for t in row_text if t] # Remove empty strings
                    
                    if row_text:
                        # Join them: "2024-01-01 - Delivered"
                        found_text = " - ".join(row_text)
            
            if found_text:
                final_status = re.sub(r'\s+', ' ', found_text).strip()
            else:
                # Fallback to priority keyword search if table parse failed
                prioritized_keywords = [
                    r"courier\s+cancelled", r"booked\s+incorrectly", 
                    r"delivered", r"out\s+for\s+delivery", r"collected", 
                    r"at\s+delivery\s+depot", r"in\s+transit", r"created"
                ]
                for k in prioritized_keywords:
                    if re.search(k, clean_html, re.IGNORECASE):
                        final_status = k.replace(r"\s+", " ").title()
                        break
            
        elif res.status_code == 404:
            final_status = "Tracking Not Found"
        else:
            final_status = f"Error {res.status_code}"
            
        if rma_id:
            update_courier_status_in_db(rma_id, final_status)
            
        return final_status

    except Exception as e:
        return f"Check Failed: {str(e)[:20]}"

# ==========================================
# 4. FRONTEND UI
# ==========================================

if 'filter_status' not in st.session_state:
    st.session_state.filter_status = "All"

if st.session_state.get('show_toast'):
    st.toast("✅ Data Refreshed Successfully!", icon="🔄")
    st.session_state['show_toast'] = False

# --- Dialogs for Actions ---
@st.dialog("Update Tracking")
def show_update_tracking_dialog(record):
    st.markdown(f"### Update Tracking for `{record['RMA ID']}`")
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
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(msg)

@st.dialog("View Timeline")
def show_timeline_dialog(record):
    st.markdown(f"### Timeline for `{record['RMA ID']}`")
    
    # 1. Add Comment Section
    with st.expander("➕ Add Comment", expanded=False):
        with st.form("add_comment_form"):
            comment_text = st.text_area("New Note")
            if st.form_submit_button("Post Comment"):
                ok, msg = push_comment_update(record['RMA ID'], comment_text)
                if ok:
                    st.success("Comment Posted!")
                    st.rerun()
                else:
                    st.error(msg)

    # 2. Timeline Display
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

# --- Header ---
col1, col2 = st.columns([3, 1])
with col1:
    st.title("Levi's ReturnGO RMA Operations")
    last_sync = st.session_state.get('last_sync', 'Not run yet')
    st.markdown(f"**Connected to:** {STORE_URL} | **Last Sync:** :green[{last_sync}]")
    search_query = st.text_input("🔍 Search Order, RMA, or Tracking", placeholder="Type to search...")

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
    
    track_nums = [s.get('trackingNumber') for s in shipments if s.get('trackingNumber')]
    track_str = ", ".join(track_nums) if track_nums else ""
    shipment_id = shipments[0].get('shipmentId') if shipments else None
    
    local_status = rma.get('_local_courier_status', '')
    
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
    created_display = str(created_at)[:10] if created_at else "N/A"
    updated_at = rma.get('lastUpdated')
    days_since = 0
    if updated_at:
        try:
            d = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
            days_since = (datetime.now(timezone.utc).date() - d.date()).days
        except: pass

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

    add_row = True
    if search_query:
        q = search_query.lower()
        if (q not in str(rma_id).lower() and q not in str(order_name).lower() and q not in str(track_str).lower()):
            add_row = False

    if add_row:
        processed_rows.append({
            "RMA ID": rma_id,
            "Order": order_name,
            "Status": status,
            "TrackingNumber": track_url if track_url else track_str,
            "TrackingStatus": local_status,
            "DisplayTrack": track_str,
            "Created": created_display,
            "Updated": str(updated_at)[:10] if updated_at else "N/A",
            "Days since updated": days_since,
            "IsNoTrack": is_no_track,
            "IsFlagged": is_flagged,
            "shipment_id": shipment_id,
            "full_data": rma,
            "Fetch Tracking Status": False,   # Checkbox 1
            "Update TrackingNumber": False,   # Checkbox 2
            "View Timeline": False            # Checkbox 3
        })

# --- Interactive Filter Boxes ---
b1, b2, b3, b4, b5 = st.columns(5)

def set_filter(f):
    st.session_state.filter_status = f
    if f in ["Pending", "Approved", "Received"]:
        perform_sync(statuses=[f])
    else:
        st.rerun()

def get_status_time(s):
    ts = get_last_sync(s)
    return f"<div class='sync-time'>Updated: {ts[11:]}</div>" if ts else "<div class='sync-time'>-</div>"

with b1:
    if st.button(f"Pending\n{counts['Pending']}"): set_filter("Pending")
    st.markdown(get_status_time("Pending"), unsafe_allow_html=True)
with b2:
    if st.button(f"Approved\n{counts['Approved']}"): set_filter("Approved")
    st.markdown(get_status_time("Approved"), unsafe_allow_html=True)
with b3:
    if st.button(f"Received\n{counts['Received']}"): set_filter("Received")
    st.markdown(get_status_time("Received"), unsafe_allow_html=True)
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
        
        display_df['No'] = display_df['No'].astype(str)
        display_df['Days since updated'] = display_df['Days since updated'].astype(str)

        # Use Data Editor for checkbox interactivity
        edited_df = st.data_editor(
            display_df[["No", "RMA ID", "Order", "Status", "TrackingNumber", "TrackingStatus", "Fetch Tracking Status", "Created", "Updated", "Days since updated", "Update TrackingNumber", "View Timeline"]],
            use_container_width=True,
            height=700,
            hide_index=True,
            key="main_table",
            column_config={
                "TrackingNumber": st.column_config.LinkColumn("TrackingNumber", display_text=r"ref=(.*)"),
                "No": st.column_config.TextColumn("No", width="small"),
                "Days since updated": st.column_config.TextColumn("Days since updated", width="small"),
                "TrackingStatus": st.column_config.TextColumn("Tracking Status", width="medium"),
                "Fetch Tracking Status": st.column_config.CheckboxColumn("Fetch Status", help="Check courier status", default=False),
                "Update TrackingNumber": st.column_config.CheckboxColumn("Edit Track", help="Edit Tracking Number", default=False),
                "View Timeline": st.column_config.CheckboxColumn("Timeline", help="View Timeline & Comments", default=False)
            },
            disabled=["No", "RMA ID", "Order", "Status", "TrackingNumber", "TrackingStatus", "Created", "Updated", "Days since updated"]
        )
        
        # --- Handle Checkbox Actions ---
        updates_triggered = False
        row_to_process = None
        action_type = None

        # Check for ticked boxes
        for index, row in edited_df.iterrows():
            if row["Fetch Tracking Status"]:
                row_to_process = display_df.iloc[index]
                action_type = "fetch"
                break # Only process one action at a time to avoid conflicts
            elif row["Update TrackingNumber"]:
                row_to_process = display_df.iloc[index]
                action_type = "update"
                break
            elif row["View Timeline"]:
                row_to_process = display_df.iloc[index]
                action_type = "timeline"
                break
        
        if row_to_process is not None:
            rma_id = row_to_process['RMA ID']
            
            if action_type == "fetch":
                tracking_raw = row_to_process['DisplayTrack']
                if tracking_raw:
                    with st.spinner(f"Checking status for {rma_id}..."):
                        check_courier_status(tracking_raw, rma_id)
                else:
                    st.warning(f"No tracking number for {rma_id}")
                st.rerun()
                
            elif action_type == "update":
                show_update_tracking_dialog(row_to_process)
                # Note: Dialogs in loops/callbacks can be tricky in Streamlit.
                # If the dialog doesn't open, we might need a session_state trigger mechanism.
                # But typically st.dialog works if called during the rerun cycle.
                
            elif action_type == "timeline":
                show_timeline_dialog(row_to_process)

    else:
        st.info("No records match the current filter.")
else:
    st.warning("No records found in database. Please click 'Sync All Data' to fetch initial data.")
