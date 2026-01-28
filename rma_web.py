import streamlit as st
import streamlit.components.v1 as components
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
from typing import Optional, Dict, Any, Set
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

if not MY_API_KEY:
    st.error("API Key not found! Please set 'RETURNGO_API_KEY' in secrets or env vars.")
    st.stop()

CACHE_EXPIRY_HOURS = 4
COURIER_REFRESH_HOURS = 12
DB_FILE = "rma_cache.db"
DB_LOCK = threading.Lock()

STORES = [
    {"name": "Diesel", "url": "diesel-dev-south-africa.myshopify.com"},
    {"name": "Hurley", "url": "hurley-dev-south-africa.myshopify.com"},
    {"name": "Jeep Apparel", "url": "jeep-apparel-dev-south-africa.myshopify.com"},
    {"name": "Reebok", "url": "reebok-dev-south-africa.myshopify.com"},
    {"name": "Superdry", "url": "superdry-dev-south-africa.myshopify.com"}
]

ACTIVE_STATUSES = ["Pending", "Approved", "Received"]

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
    .rows-count {
        font-size: 1em;
        color: #d1d5db;
        text-align: left;
    }
    .rows-count .value {
        font-weight: bold;
        color: #60a5fa;
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
        c.execute('''
            CREATE TABLE IF NOT EXISTS rmas (
                rma_id TEXT PRIMARY KEY,
                store_url TEXT,
                status TEXT,
                created_at TEXT,
                json_data TEXT,
                last_fetched TEXT,
                courier_status TEXT,
                courier_last_checked TEXT
            )
        ''')
        # Check if courier columns exist and add them if not
        try:
            c.execute("SELECT courier_status FROM rmas LIMIT 1")
        except sqlite3.OperationalError:
            try:
                c.execute("ALTER TABLE rmas ADD COLUMN courier_status TEXT")
            except: pass
        try:
            c.execute("SELECT courier_last_checked FROM rmas LIMIT 1")
        except sqlite3.OperationalError:
            try:
                c.execute("ALTER TABLE rmas ADD COLUMN courier_last_checked TEXT")
            except: pass
        
        c.execute('''
            CREATE TABLE IF NOT EXISTS sync_log (
                scope TEXT PRIMARY KEY,
                last_sync TEXT
            )
        ''')
        conn.commit()
        conn.close()

init_db()

def upsert_rma(rma_id, store_url, status, created_at, json_data, courier_status=None, courier_checked=None):
    now = datetime.now(timezone.utc).isoformat()
    with DB_LOCK:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute('''
            INSERT INTO rmas (rma_id, store_url, status, created_at, json_data, last_fetched, courier_status, courier_last_checked)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(rma_id) DO UPDATE SET
                store_url=excluded.store_url,
                status=excluded.status,
                created_at=excluded.created_at,
                json_data=excluded.json_data,
                last_fetched=excluded.last_fetched,
                courier_status=COALESCE(excluded.courier_status, courier_status),
                courier_last_checked=COALESCE(excluded.courier_last_checked, courier_last_checked)
        ''', (rma_id, store_url, status, created_at, json_data, now, courier_status, courier_checked))
        conn.commit()
        conn.close()

def set_last_sync(scope, ts):
    with DB_LOCK:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute('INSERT OR REPLACE INTO sync_log (scope, last_sync) VALUES (?, ?)', (scope, ts))
        conn.commit()
        conn.close()

def get_last_sync(store_url, status):
    scope = f"{store_url}_{status}"
    with DB_LOCK:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute('SELECT last_sync FROM sync_log WHERE scope=?', (scope,))
        r = c.fetchone()
        conn.close()
        return r[0] if r else None

@st.cache_data(ttl=3600)
def get_all_active_from_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('SELECT rma_id, store_url, status, json_data, courier_status, courier_last_checked FROM rmas WHERE status IN (?, ?, ?)', 
              ('Pending', 'Approved', 'Received'))
    rows = c.fetchall()
    conn.close()
    results = []
    for row in rows:
        rma_id, store_url, status, json_str, courier_status, courier_last_checked = row
        try:
            data = json.loads(json_str)
            data['_local_courier_status'] = courier_status or ''
            data['_local_courier_checked'] = courier_last_checked or ''
            results.append(data)
        except: pass
    return results

def should_refresh_courier(rma_data):
    """Check if courier status needs refreshing based on time elapsed"""
    last_checked = rma_data.get('_local_courier_checked')
    if not last_checked:
        return True
    
    try:
        last_checked_dt = datetime.fromisoformat(last_checked.replace('Z', '+00:00'))
        now = datetime.now(timezone.utc)
        hours_elapsed = (now - last_checked_dt).total_seconds() / 3600
        return hours_elapsed >= COURIER_REFRESH_HOURS
    except:
        return True

def clear_db():
    try:
        with DB_LOCK:
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute('DELETE FROM rmas')
            c.execute('DELETE FROM sync_log')
            conn.commit()
            conn.close()
        return True
    except: return False

# ==========================================
# 3. API CALLS
# ==========================================
def fetch_rma_list(store, status):
    url = f"{api_url}/rmas/open?shop={store['url']}&status={status}"
    headers = {"x-api-key": MY_API_KEY}
    try:
        res = get_session().get(url, headers=headers, timeout=20)
        if res.status_code == 200:
            data = res.json()
            return data.get('data', [])
        else:
            st.error(f"Error fetching RMAs: {res.status_code}")
            return []
    except Exception as e:
        st.error(f"Exception: {e}")
        return []

def fetch_rma_detail(rma_id, store_url):
    url = f"{api_url}/rma/{rma_id}"
    headers = {"x-api-key": MY_API_KEY, "x-shop-name": store_url}
    try:
        res = get_session().get(url, headers=headers, timeout=20)
        if res.status_code == 200:
            return res.json()
        return None
    except: return None

def push_tracking_update(rma_id, shipment_id, new_tracking, store_url):
    url = f"{api_url}/shipment/{shipment_id}/tracking"
    headers = {"x-api-key": MY_API_KEY, "x-shop-name": store_url, "Content-Type": "application/json"}
    payload = {"trackingNumber": new_tracking}
    try:
        res = get_session().put(url, headers=headers, json=payload, timeout=15)
        if res.status_code in [200, 201]:
            return True, "Success"
        return False, f"API Error {res.status_code}"
    except Exception as e: return False, str(e)

def push_comment_update(rma_id, comment_text, store_url):
    url = f"{api_url}{RMA_COMMENT_PATH.format(rmaId=rma_id)}"
    headers = {"x-api-key": MY_API_KEY, "X-API-KEY": MY_API_KEY, "x-shop-name": store_url, "Content-Type": "application/json"}
    payload = {"htmlText": comment_text}
    try:
        res = get_session().post(url, headers=headers, json=payload, timeout=15)
        if res.status_code in [200, 201]:
            return True, "Success"
        return False, f"API Error {res.status_code}: {res.text}"
    except Exception as e: return False, str(e)

# ==========================================
# 4. SYNC LOGIC
# ==========================================
def perform_sync(store_obj=None, status=None):
    st.cache_data.clear()
    
    stores_to_sync = [store_obj] if store_obj else STORES
    statuses_to_sync = [status] if status else ACTIVE_STATUSES
    
    progress_placeholder = st.empty()
    status_text = st.empty()
    
    total_tasks = len(stores_to_sync) * len(statuses_to_sync)
    current_task = 0
    
    for store in stores_to_sync:
        for stat in statuses_to_sync:
            current_task += 1
            progress_placeholder.progress(current_task / total_tasks)
            status_text.text(f"Syncing {store['name']} - {stat}...")
            
            rma_list = fetch_rma_list(store, stat)
            
            for rma_summary in rma_list:
                rma_id = rma_summary.get('rmaId')
                if not rma_id:
                    continue
                
                full_data = fetch_rma_detail(rma_id, store['url'])
                if full_data:
                    created_at = rma_summary.get('createdAt', '')
                    upsert_rma(rma_id, store['url'], stat, created_at, json.dumps(full_data))
            
            scope = f"{store['url']}_{stat}"
            set_last_sync(scope, datetime.now(timezone.utc).isoformat())
            time.sleep(0.5)
    
    progress_placeholder.empty()
    status_text.empty()
    st.session_state['show_toast'] = True
    st.rerun()

# ==========================================
# 5. UTILITY FUNCTIONS
# ==========================================
def days_since(date_str, today=None):
    """Calculate days since a given date, returning as integer for proper sorting"""
    if not date_str or date_str == "N/A":
        return 999999  # Return large number for N/A to sort to end
    try:
        if today is None:
            today = datetime.now(timezone.utc).date()
        date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00')).date()
        return (today - date_obj).days
    except:
        return 999999

def get_store_returngo_url(store_url, rma_id):
    """Generate the ReturnGO URL for a specific RMA based on store"""
    return f"https://app.returngo.ai/dashboard/returns?filter_status=open&rmaid={rma_id}"

# ==========================================
# 6. FRONTEND UI LOGIC
# ==========================================

if 'filter_state' not in st.session_state:
    st.session_state.filter_state = {"store": None, "status": "All"}
if st.session_state.get('show_toast'):
    st.toast("✅ API Sync Complete!", icon="🔄")
    st.session_state['show_toast'] = False
if st.session_state.get('show_update_toast'):
    st.toast("✅ UI refreshed from database.", icon="🔄")
    st.session_state['show_update_toast'] = False

@st.dialog("RMA Actions")
def show_rma_actions_dialog(record):
    rma_id = record['_rma_id_text']
    rma_url = record['RMA ID']
    
    tab1, tab2 = st.tabs(["Update Tracking", "View Timeline"])
    
    with tab1:
        st.markdown(f"### RMA `{rma_id}`")
        if rma_url:
            st.markdown(f"[Open in ReturnGO]({rma_url})")
        
        raw_track = record['DisplayTrack']
        with st.form("update_track_form"):
            new_track = st.text_input("Tracking Number", value=raw_track)
            if st.form_submit_button("Save Changes"):
                if not record['shipment_id']:
                    st.error("No Shipment ID.")
                else:
                    ok, msg = push_tracking_update(rma_id, record['shipment_id'], new_track, record['Store URL'])
                    if ok:
                        st.success("Tracking Updated!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(msg)
    
    with tab2:
        st.markdown(f"### Timeline for `{rma_id}`")
        with st.expander("➕ Add Comment", expanded=False):
            with st.form("add_comment_form"):
                comment_text = st.text_area("New Note")
                if st.form_submit_button("Post Comment"):
                    ok, msg = push_comment_update(rma_id, comment_text, record['Store URL'])
                    if ok:
                        st.success("Comment Posted!")
                        st.rerun()
                    else:
                        st.error(msg)
        
        full = record['full_data']
        timeline = full.get('comments', [])
        if not timeline:
            st.info("No timeline events.")
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
    else:
        st.rerun()

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
        else:
            st.error("DB might be locked.")

# --- Process Data ---
raw_data = get_all_active_from_db()
processed_rows = []
store_counts = {s['url']: {"Pending": 0, "Approved": 0, "Received": 0, "NoTrack": 0, "Flagged": 0} for s in STORES}
today = datetime.now(timezone.utc).date()

for rma in raw_data:
    store_url = rma.get('store_url')
    if not store_url:
        continue
    
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
    
    track_link_url = f"https://optimise.parcelninja.com/shipment/track?WaybillNo={track_nums[0]}&ref={track_nums[0]}" if track_nums else ""
    
    created_at = summary.get('createdAt')
    if not created_at:
        for evt in summary.get('events', []):
            if evt.get('eventName') == 'RMA_CREATED':
                created_at = evt.get('eventDate')
                break
    
    u_at = rma.get('lastUpdated')
    d_since = days_since(str(created_at)[:10] if created_at else "N/A", today=today)
    
    if store_url in store_counts:
        if status in store_counts[store_url]:
            store_counts[store_url][status] += 1
        
        is_nt = False
        if status == "Approved" and not track_str:
            store_counts[store_url]["NoTrack"] += 1
            is_nt = True
        
        is_fg = False
        if any("flagged" in c.get('htmlText', '').lower() for c in comments):
            store_counts[store_url]["Flagged"] += 1
            is_fg = True
        
        add_row = True
        if search_query:
            q = search_query.lower()
            if (q not in str(rma_id).lower() and 
                q not in str(order_name).lower() and 
                q not in str(track_str).lower()):
                add_row = False
        
        if add_row:
            processed_rows.append({
                "No": "",
                "Store Name": next((s['name'] for s in STORES if s['url'] == store_url), "Unknown"),
                "RMA ID": get_store_returngo_url(store_url, rma_id),
                "Order": order_name,
                "Status": status,
                "Store URL": store_url,
                "TrackingNumber": track_link_url,
                "TrackingStatus": local_status,
                "Created": str(created_at)[:10] if created_at else "N/A",
                "Updated": str(u_at)[:10] if u_at else "N/A",
                "Days": d_since,  # Now returns integer for proper sorting
                "DisplayTrack": track_str,
                "shipment_id": shipment_id,
                "full_data": rma,
                "_rma_id_text": rma_id,
                "is_nt": is_nt,
                "is_fg": is_fg
            })

# --- Store Boxes ---
cols = st.columns(len(STORES))
for i, store in enumerate(STORES):
    c = store_counts[store['url']]
    with cols[i]:
        st.markdown(f"**{store['name'].upper()}**")
        
        def show_btn(label, stat, key):
            ts = get_last_sync(store['url'], stat)
            st.markdown(
                f"<div class='sync-time'>Updated: {ts[11:19] if ts else '-'}</div>",
                unsafe_allow_html=True,
            )
            if st.button(f"{label}\n{c[stat]}", key=key):
                handle_filter_click(store['url'], stat)
        
        show_btn("Pending", "Pending", f"p_{i}")
        show_btn("Approved", "Approved", f"a_{i}")
        show_btn("Received", "Received", f"r_{i}")
        
        if st.button(f"No Track\n{c['NoTrack']}", key=f"n_{i}"):
            handle_filter_click(store['url'], "NoTrack")
        if st.button(f"🚩 Flagged\n{c['Flagged']}", key=f"f_{i}"):
            handle_filter_click(store['url'], "Flagged")

st.divider()

# --- Filter and Display Data ---
cur = st.session_state.filter_state
df_view = pd.DataFrame(processed_rows)

if not df_view.empty:
    if cur['store']:
        df_view = df_view[df_view['Store URL'] == cur['store']]
    
    f_stat = cur['status']
    if f_stat == "Pending":
        df_view = df_view[df_view['Status'] == 'Pending']
    elif f_stat == "Approved":
        df_view = df_view[df_view['Status'] == 'Approved']
    elif f_stat == "Received":
        df_view = df_view[df_view['Status'] == 'Received']
    elif f_stat == "NoTrack":
        df_view = df_view[df_view['is_nt'] == True]
    elif f_stat == "Flagged":
        df_view = df_view[df_view['is_fg'] == True]
    
    if not df_view.empty:
        # Sort by Days (now an integer) in descending order
        df_view = df_view.sort_values(by="Days", ascending=True).reset_index(drop=True)
        df_view['No'] = range(1, len(df_view) + 1)
        df_view['No'] = df_view['No'].astype(str)
        
        # Show row count
        total_rows = len(df_view)
        count_col, action_col = st.columns([5, 3], vertical_alignment="center")
        with count_col:
            st.markdown(
                f"<div class='rows-count'>Rows in table: <span class='value'>{total_rows}</span></div>",
                unsafe_allow_html=True,
            )
        with action_col:
            # Add copy all button
            display_cols = ["No", "Store Name", "RMA ID", "Order", "Status", "TrackingNumber", 
                          "TrackingStatus", "Created", "Updated", "Days"]
            tsv_rows = [display_cols] + df_view[display_cols].astype(str).values.tolist()
            tsv_text = "\n".join(["\t".join(row) for row in tsv_rows])
            
            if st.button("📋 Copy all", key="copy_all_btn"):
                st.session_state["copy_all_payload"] = tsv_text
        
        # Handle clipboard copy
        if st.session_state.get("copy_all_payload"):
            copy_payload = st.session_state.pop("copy_all_payload")
            json_payload = json.dumps(copy_payload)
            components.html(
                f"""
                <script>
                  (async () => {{
                    try {{
                      const payload = JSON.parse({json_payload});
                      if (navigator.clipboard?.writeText) {{
                        await navigator.clipboard.writeText(payload);
                      }}
                    }} catch (err) {{
                      console.error("Clipboard write failed.", err);
                    }}
                  }})();
                </script>
                """,
                height=0,
            )
            st.toast("Copied table to clipboard.", icon="📋")
        
        # Display Table with row highlighting
        def highlight_missing_tracking(row):
            if row.get("Status") == "Approved" and not row.get("DisplayTrack"):
                return ["background-color: rgba(220, 38, 38, 0.35); color: #fee2e2;"] * 12
            return [""] * 12
        
        display_df = df_view[["No", "Store Name", "RMA ID", "Order", "Status", "TrackingNumber", 
                             "TrackingStatus", "Created", "Updated", "Days", 
                             "Store URL", "DisplayTrack", "shipment_id", "full_data", "_rma_id_text"]].copy()
        
        styled_table = display_df[["No", "Store Name", "RMA ID", "Order", "Status", "TrackingNumber", 
                                   "TrackingStatus", "Created", "Updated", "Days"]].style.apply(
            highlight_missing_tracking, axis=1
        )
        
        sel_event = st.dataframe(
            styled_table,
            use_container_width=True,
            height=700,
            hide_index=True,
            key="main_table",
            on_select="rerun",
            selection_mode="single-row",
            column_config={
                "No": st.column_config.TextColumn("No", width="small"),
                "Store Name": st.column_config.TextColumn("Store Name", width="small"),
                "RMA ID": st.column_config.LinkColumn("RMA ID", display_text=r"rmaid=([^&]+)", width="small"),
                "Order": st.column_config.TextColumn("Order", width="medium"),
                "TrackingNumber": st.column_config.LinkColumn("Tracking Number", display_text=r"ref=(.*)", width="medium"),
                "TrackingStatus": st.column_config.TextColumn("Tracking Status", width="medium"),
                "Days": st.column_config.NumberColumn("Days", width="small", format="%d")
            }
        )
        
        # Handle row selection
        sel_rows = (sel_event.selection.rows if sel_event and sel_event.selection and 
                   hasattr(sel_event.selection, "rows") else []) or []
        if sel_rows:
            idx = int(sel_rows[0])
            show_rma_actions_dialog(display_df.iloc[idx])
    else:
        st.info("No matching records.")
else:
    st.info("No data available. Click 'Sync All Data' to load RMAs.")
