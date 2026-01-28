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
import sys
import logging
from datetime import datetime, timedelta, timezone
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import concurrent.futures
from typing import Optional, Dict, Any, Set
from returngo_api import api_url, RMA_COMMENT_PATH

# ==========================================
# LOGGING SETUP
# ==========================================
# Create logs directory if it doesn't exist
LOG_DIR = r"C:\Users\Taswell\OneDrive\Documents\GitHub\Returngo\Connection"
try:
    os.makedirs(LOG_DIR, exist_ok=True)
    log_dir_created = True
except Exception as e:
    LOG_DIR = "."  # Fallback to current directory
    log_dir_created = False

# Create log filename with timestamp
log_filename = os.path.join(LOG_DIR, f"rma_web_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

# Configure logging to both console and file
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename, mode='w', encoding='utf-8')
    ],
    force=True
)
logger = logging.getLogger(__name__)

# Log startup
logger.info("=" * 100)
logger.info("RMA WEB APPLICATION STARTING")
logger.info(f"Log file: {log_filename}")
logger.info(f"Log directory created: {log_dir_created}")
logger.info(f"Timestamp: {datetime.now().isoformat()}")
logger.info(f"Python version: {sys.version}")
logger.info("=" * 100)

# ==========================================
# 1. CONFIGURATION
# ==========================================
logger.info("Setting up page configuration...")
st.set_page_config(page_title="Bounty Apparel ReturnGo RMAs", layout="wide", page_icon="🔄️")

# ACCESS SECRETS
logger.info("Loading API credentials...")
try:
    MY_API_KEY = st.secrets["RETURNGO_API_KEY"]
    logger.info("✓ API Key loaded from secrets")
except (FileNotFoundError, KeyError):
    MY_API_KEY = os.environ.get("RETURNGO_API_KEY")
    if MY_API_KEY:
        logger.info("✓ API Key loaded from environment")
    else:
        logger.error("✗ API Key not found in secrets or environment")

if not MY_API_KEY:
    logger.critical("FATAL: API Key not found! Application stopped.")
    st.error("API Key not found! Please set 'RETURNGO_API_KEY' in secrets or env vars.")
    st.stop()

logger.info(f"API Key present: {MY_API_KEY[:8]}..." if MY_API_KEY else "API Key: None")

CACHE_EXPIRY_HOURS = 4
COURIER_REFRESH_HOURS = 12

# Database file location - stored in same directory as this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if os.path.abspath(__file__) != "" else os.getcwd()
DB_FILE = os.path.join(SCRIPT_DIR, "rma_cache.db")
DB_LOCK = threading.Lock()

logger.info(f"Database file will be stored at: {DB_FILE}")
logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f"Script directory: {SCRIPT_DIR}")

STORES = [
    {"name": "Diesel", "url": "diesel-dev-south-africa.myshopify.com"},
    {"name": "Hurley", "url": "hurley-dev-south-africa.myshopify.com"},
    {"name": "Jeep Apparel", "url": "jeep-apparel-dev-south-africa.myshopify.com"},
    {"name": "Reebok", "url": "reebok-dev-south-africa.myshopify.com"},
    {"name": "Superdry", "url": "superdry-dev-south-africa.myshopify.com"}
]

logger.info(f"Configured {len(STORES)} stores:")
for store in STORES:
    logger.info(f"  - {store['name']}: {store['url']}")

ACTIVE_STATUSES = ["Pending", "Approved", "Received"]
logger.info(f"Active statuses: {ACTIVE_STATUSES}")

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
    logger.info("Creating HTTP session with retry logic...")
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    logger.info("✓ HTTP session created successfully")
    return session

# ==========================================
# 2. DATABASE MANAGER
# ==========================================
def init_db():
    logger.info("Initializing database...")
    logger.info(f"Database file: {DB_FILE}")
    with DB_LOCK:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        logger.debug("Creating rmas table if not exists...")
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
            logger.debug("✓ courier_status column exists")
        except sqlite3.OperationalError:
            logger.warning("courier_status column missing, adding...")
            try:
                c.execute("ALTER TABLE rmas ADD COLUMN courier_status TEXT")
                logger.info("✓ Added courier_status column")
            except:
                logger.error("✗ Failed to add courier_status column")
        try:
            c.execute("SELECT courier_last_checked FROM rmas LIMIT 1")
            logger.debug("✓ courier_last_checked column exists")
        except sqlite3.OperationalError:
            logger.warning("courier_last_checked column missing, adding...")
            try:
                c.execute("ALTER TABLE rmas ADD COLUMN courier_last_checked TEXT")
                logger.info("✓ Added courier_last_checked column")
            except:
                logger.error("✗ Failed to add courier_last_checked column")
        
        logger.debug("Creating sync_log table if not exists...")
        c.execute('''
            CREATE TABLE IF NOT EXISTS sync_log (
                scope TEXT PRIMARY KEY,
                last_sync TEXT
            )
        ''')
        conn.commit()
        conn.close()
        logger.info("✓ Database initialized successfully")

logger.info("Calling init_db()...")
init_db()

def upsert_rma(rma_id, store_url, status, created_at, json_data, courier_status=None, courier_checked=None):
    logger.debug(f"Upserting RMA: {rma_id} | Store: {store_url} | Status: {status}")
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
    logger.debug(f"✓ RMA {rma_id} upserted successfully")

def set_last_sync(scope, ts):
    logger.debug(f"Setting last sync for scope: {scope} at {ts}")
    with DB_LOCK:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute('INSERT OR REPLACE INTO sync_log (scope, last_sync) VALUES (?, ?)', (scope, ts))
        conn.commit()
        conn.close()
    logger.debug(f"✓ Last sync set for {scope}")

def get_last_sync(store_url, status):
    scope = f"{store_url}_{status}"
    with DB_LOCK:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute('SELECT last_sync FROM sync_log WHERE scope=?', (scope,))
        r = c.fetchone()
        conn.close()
        result = r[0] if r else None
        logger.debug(f"Last sync for {scope}: {result}")
        return result

def get_all_active_from_db():
    """Load all active RMAs from database - NO CACHE so data shows immediately after sync"""
    logger.info("Loading all active RMAs from database...")
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('SELECT rma_id, store_url, status, json_data, courier_status, courier_last_checked FROM rmas WHERE status IN (?, ?, ?)', 
              ('Pending', 'Approved', 'Received'))
    rows = c.fetchall()
    conn.close()
    logger.info(f"✓ Loaded {len(rows)} RMAs from database")
    results = []
    for row in rows:
        rma_id, store_url, status, json_str, courier_status, courier_last_checked = row
        try:
            data = json.loads(json_str)
            data['_local_courier_status'] = courier_status or ''
            data['_local_courier_checked'] = courier_last_checked or ''
            data['store_url'] = store_url  # Make sure store_url is in the data
            results.append(data)
        except Exception as e:
            logger.error(f"Error parsing RMA {rma_id}: {e}")
    logger.info(f"✓ Successfully parsed {len(results)} RMAs")
    return results

def should_refresh_courier(rma_data):
    """Check if courier status needs refreshing based on time elapsed"""
    last_checked = rma_data.get('_local_courier_checked')
    if not last_checked:
        logger.debug("No courier check timestamp, needs refresh")
        return True
    
    try:
        last_checked_dt = datetime.fromisoformat(last_checked.replace('Z', '+00:00'))
        now = datetime.now(timezone.utc)
        hours_elapsed = (now - last_checked_dt).total_seconds() / 3600
        needs_refresh = hours_elapsed >= COURIER_REFRESH_HOURS
        logger.debug(f"Courier last checked {hours_elapsed:.1f}h ago, needs refresh: {needs_refresh}")
        return needs_refresh
    except Exception as e:
        logger.error(f"Error checking courier refresh time: {e}")
        return True

def clear_db():
    logger.info("Clearing database...")
    try:
        with DB_LOCK:
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute('SELECT COUNT(*) FROM rmas')
            rma_count = c.fetchone()[0]
            c.execute('SELECT COUNT(*) FROM sync_log')
            sync_count = c.fetchone()[0]
            logger.info(f"Deleting {rma_count} RMAs and {sync_count} sync logs")
            c.execute('DELETE FROM rmas')
            c.execute('DELETE FROM sync_log')
            conn.commit()
            conn.close()
        logger.info("✓ Database cleared successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Error clearing database: {e}")
        return False

# ==========================================
# 3. API CALLS
# ==========================================
def fetch_rma_list(store, status):
    """Fetch all RMAs for a given store and status with pagination"""
    logger.info(f"→ Fetching RMA list for {store['name']} ({store['url']}) - Status: {status}")
    all_rmas = []
    cursor = None
    page = 1
    max_pages = 100  # Safety limit
    
    # Include both casings of API key header for compatibility
    headers = {
        "x-api-key": MY_API_KEY,
        "X-API-KEY": MY_API_KEY,
        "x-shop-name": store['url']
    }
    logger.debug(f"Headers prepared: x-shop-name={store['url']}, API key present")
    
    while page <= max_pages:
        try:
            # Build URL with cursor pagination
            base_url = f"{api_url('/rmas')}?status={status}&pagesize=500"
            url = f"{base_url}&cursor={cursor}" if cursor else base_url
            
            logger.info(f"  Page {page}: GET {url}")
            
            res = get_session().get(url, headers=headers, timeout=20)
            
            logger.info(f"  Response: Status {res.status_code}, Content-Length: {len(res.content) if res.content else 0}")
            
            if res.status_code != 200:
                error_detail = ""
                try:
                    error_detail = res.text[:500] if res.text else "(empty response)"
                except:
                    pass
                logger.error(f"  ✗ API Error: {res.status_code}")
                logger.error(f"  Response body: {error_detail}")
                
                if page == 1:  # Only show UI error on first page
                    st.error(f"Error fetching RMAs for {store['name']}: {res.status_code}\nURL: {url}\nResponse: {error_detail}")
                break
            
            data = res.json()
            rmas = data.get("rmas", [])
            
            logger.info(f"  ✓ Received {len(rmas)} RMAs on page {page}")
            
            if not rmas:
                logger.info(f"  No more RMAs, stopping pagination")
                break
            
            all_rmas.extend(rmas)
            
            # Check for next page
            cursor = data.get("next_cursor")
            if cursor:
                logger.debug(f"  Next cursor: {cursor[:50]}...")
            else:
                logger.info(f"  No next cursor, pagination complete")
                break
                
            page += 1
            
        except Exception as e:
            logger.error(f"  ✗ Exception on page {page}: {str(e)}", exc_info=True)
            if page == 1:  # Only show UI error on first page
                st.error(f"Exception fetching RMAs for {store['name']}: {str(e)}")
            break
    
    logger.info(f"✓ Fetch complete: {len(all_rmas)} total RMAs for {store['name']} - {status}")
    return all_rmas

def fetch_rma_detail(rma_id, store_url):
    """Fetch detailed RMA data"""
    logger.debug(f"  → Fetching detail for RMA {rma_id}")
    url = api_url(f"/rma/{rma_id}")
    headers = {
        "x-api-key": MY_API_KEY,
        "X-API-KEY": MY_API_KEY,
        "x-shop-name": store_url
    }
    try:
        res = get_session().get(url, headers=headers, timeout=20)
        logger.debug(f"    Response: {res.status_code}")
        if res.status_code == 200:
            logger.debug(f"    ✓ Detail fetched for {rma_id}")
            return res.json()
        logger.warning(f"    ✗ Failed to fetch detail for {rma_id}: {res.status_code}")
        return None
    except Exception as e:
        logger.error(f"    ✗ Exception fetching {rma_id}: {e}")
        return None

def push_tracking_update(rma_id, shipment_id, new_tracking, store_url):
    """Update tracking number for a shipment"""
    url = api_url(f"/shipment/{shipment_id}/tracking")
    headers = {
        "x-api-key": MY_API_KEY,
        "X-API-KEY": MY_API_KEY,
        "x-shop-name": store_url,
        "Content-Type": "application/json"
    }
    payload = {"trackingNumber": new_tracking}
    try:
        res = get_session().put(url, headers=headers, json=payload, timeout=15)
        if res.status_code in [200, 201]:
            return True, "Success"
        return False, f"API Error {res.status_code}"
    except Exception as e:
        return False, str(e)

def push_comment_update(rma_id, comment_text, store_url):
    """Post a comment to an RMA"""
    url = api_url(RMA_COMMENT_PATH.format(rmaId=rma_id))
    headers = {"X-API-KEY": MY_API_KEY, "x-api-key": MY_API_KEY, "x-shop-name": store_url, "Content-Type": "application/json"}
    payload = {"htmlText": comment_text}
    try:
        res = get_session().post(url, headers=headers, json=payload, timeout=15)
        if res.status_code in [200, 201]:
            return True, "Success"
        return False, f"API Error {res.status_code}: {res.text}"
    except Exception as e:
        return False, str(e)

# ==========================================
# 4. SYNC LOGIC
# ==========================================
def perform_sync(store_obj=None, status=None):
    """Sync RMAs from ReturnGO API"""
    logger.info("=" * 100)
    logger.info("SYNC OPERATION STARTED")
    logger.info(f"Store filter: {store_obj['name'] if store_obj else 'ALL STORES'}")
    logger.info(f"Status filter: {status if status else 'ALL STATUSES'}")
    logger.info("=" * 100)
    
    st.cache_data.clear()
    logger.info("Cache cleared")
    
    stores_to_sync = [store_obj] if store_obj else STORES
    statuses_to_sync = [status] if status else ACTIVE_STATUSES
    
    logger.info(f"Will sync {len(stores_to_sync)} stores × {len(statuses_to_sync)} statuses = {len(stores_to_sync) * len(statuses_to_sync)} combinations")
    
    progress_placeholder = st.empty()
    status_text = st.empty()
    
    total_tasks = len(stores_to_sync) * len(statuses_to_sync)
    current_task = 0
    total_rmas_synced = 0
    
    for store in stores_to_sync:
        logger.info(f"Processing store: {store['name']} ({store['url']})")
        for stat in statuses_to_sync:
            current_task += 1
            progress_pct = current_task / total_tasks
            
            logger.info(f"  Task {current_task}/{total_tasks}: Syncing {stat} status")
            status_text.text(f"Syncing {store['name']} - {stat}... ({current_task}/{total_tasks})")
            progress_placeholder.progress(progress_pct)
            
            # Fetch list of RMAs
            rma_list = fetch_rma_list(store, stat)
            logger.info(f"  Retrieved {len(rma_list)} RMAs for {stat}")
            
            # Fetch details for each RMA
            for idx, rma_summary in enumerate(rma_list):
                rma_id = rma_summary.get('rmaId')
                if not rma_id:
                    logger.warning(f"    Skipping RMA with no ID")
                    continue
                
                # Update sub-progress within this store/status combo
                sub_progress_text = f"Syncing {store['name']} - {stat}... ({idx+1}/{len(rma_list)} RMAs)"
                status_text.text(sub_progress_text)
                
                full_data = fetch_rma_detail(rma_id, store['url'])
                if full_data:
                    created_at = rma_summary.get('createdAt', '')
                    upsert_rma(rma_id, store['url'], stat, created_at, json.dumps(full_data))
                    total_rmas_synced += 1
                else:
                    logger.warning(f"    ✗ Failed to get detail for {rma_id}")
                
                # Small delay to avoid rate limiting
                time.sleep(0.1)
            
            # Save sync timestamp
            scope = f"{store['url']}_{stat}"
            sync_time = datetime.now(timezone.utc).isoformat()
            set_last_sync(scope, sync_time)
            logger.info(f"  ✓ Sync timestamp saved for {scope}")
    
    # Clean up progress indicators
    progress_placeholder.empty()
    status_text.empty()
    
    logger.info("=" * 100)
    logger.info(f"SYNC OPERATION COMPLETE: {total_rmas_synced} RMAs synced")
    logger.info("=" * 100)
    
    # Show completion message
    st.toast(f"✅ Sync Complete! {total_rmas_synced} RMAs synced.", icon="🔄")
    st.session_state['show_toast'] = True
    time.sleep(0.5)
    
    logger.info("Triggering rerun...")
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
logger.info("-" * 100)
logger.info("INITIALIZING UI")
logger.info("-" * 100)

if 'filter_state' not in st.session_state:
    st.session_state.filter_state = {"store": None, "status": "All"}
    logger.info("Initialized filter_state in session")
    
if st.session_state.get('show_toast'):
    st.toast("✅ API Sync Complete!", icon="🔄")
    st.session_state['show_toast'] = False
    logger.info("Displayed sync complete toast")
    
if st.session_state.get('show_update_toast'):
    st.toast("✅ UI refreshed from database.", icon="🔄")
    st.session_state['show_update_toast'] = False
    logger.info("Displayed update complete toast")

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
    logger.info(f"Filter clicked: Store={store_url}, Status={status}")
    st.session_state.filter_state = {"store": store_url, "status": status}
    if status in ["Pending", "Approved", "Received"]:
        store_obj = next((s for s in STORES if s['url'] == store_url), None)
        logger.info(f"Triggering sync for {store_obj['name'] if store_obj else 'Unknown'} - {status}")
        perform_sync(store_obj, status)
    else:
        logger.info(f"Filter set, triggering rerun")
        st.rerun()

# --- Header ---
logger.info("Rendering header section...")

# Show database location
with st.expander("ℹ️ Database Info", expanded=False):
    st.code(f"Database Location: {DB_FILE}")
    st.code(f"Script Directory: {SCRIPT_DIR}")
    st.code(f"Working Directory: {os.getcwd()}")
    
    # Check if database exists and show stats
    if os.path.exists(DB_FILE):
        db_size = os.path.getsize(DB_FILE) / 1024  # KB
        db_modified = datetime.fromtimestamp(os.path.getmtime(DB_FILE)).strftime('%Y-%m-%d %H:%M:%S')
        st.success(f"✅ Database exists ({db_size:.1f} KB, last modified: {db_modified})")
        
        # Show row counts
        try:
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM rmas")
            rma_count = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM sync_log")
            sync_count = c.fetchone()[0]
            conn.close()
            st.info(f"📊 Database contains: {rma_count} RMAs, {sync_count} sync logs")
        except Exception as e:
            st.warning(f"Could not read database: {e}")
    else:
        st.warning("⚠️ Database file does not exist yet. It will be created on first sync.")

col1, col2 = st.columns([3, 1])
with col1:
    st.title("Bounty Apparel ReturnGo RMAs 🔄️")
    search_query = st.text_input("🔍 Search Order, RMA, or Tracking", placeholder="Type to search...")
    if search_query:
        logger.info(f"Search query: {search_query}")
with col2:
    sync_col, update_col = st.columns(2)
    with sync_col:
        if st.button("🔄 Sync All Data", type="primary", help="Fetch latest data from ReturnGO API for all stores and statuses. This may take several minutes."):
            logger.info("USER ACTION: Sync All Data button clicked")
            perform_sync()
    with update_col:
        if st.button("🔄 Update All", type="secondary", help="Refresh the display from local database without fetching new data from API. Use when data looks stale."):
            logger.info("USER ACTION: Update All button clicked")
            st.cache_data.clear()
            st.cache_resource.clear()
            st.session_state.pop("main_table", None)
            st.session_state['show_update_toast'] = True
            logger.info("Caches cleared, triggering rerun")
            st.rerun()
    if st.button("🗑️ Reset Cache", type="secondary", help="⚠️ Delete ALL cached data from database. Use only for troubleshooting. Next sync will take longer."):
        logger.info("USER ACTION: Reset Cache button clicked")
        if clear_db():
            st.cache_data.clear()
            st.cache_resource.clear()
            st.session_state.filter_state = {"store": None, "status": "All"}
            st.session_state.pop("last_sync", None)
            st.session_state.pop("main_table", None)
            st.success("Cache cleared! Data will reload on next sync.")
            logger.info("✓ Cache reset successful")
            st.rerun()
        else:
            st.error("DB might be locked.")
            logger.error("✗ Cache reset failed - DB locked")

# --- Process Data ---
logger.info("Loading and processing RMA data...")
raw_data = get_all_active_from_db()
processed_rows = []
store_counts = {s['url']: {"Pending": 0, "Approved": 0, "Received": 0, "NoTrack": 0, "Flagged": 0} for s in STORES}
today = datetime.now(timezone.utc).date()

logger.info(f"Processing {len(raw_data)} RMAs...")

# Show database stats
if len(raw_data) > 0:
    st.info(f"📊 Database contains {len(raw_data)} active RMAs")
else:
    st.warning("⚠️ Database is empty. Click '🔄 Sync All Data' to fetch RMAs from ReturnGO.")

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
        
        def show_btn(label, stat, key, help_text, counts):
            ts = get_last_sync(store['url'], stat)
            st.markdown(
                f"<div class='sync-time'>Updated: {ts[11:19] if ts else '-'}</div>",
                unsafe_allow_html=True,
            )
            if st.button(f"{label}\n{counts[stat]}", key=key, help=help_text):
                handle_filter_click(store['url'], stat)
        
        show_btn("Pending", "Pending", f"p_{i}", f"Sync and view {c['Pending']} pending RMAs for {store['name']}", c)
        show_btn("Approved", "Approved", f"a_{i}", f"Sync and view {c['Approved']} approved RMAs for {store['name']}", c)
        show_btn("Received", "Received", f"r_{i}", f"Sync and view {c['Received']} received RMAs for {store['name']}", c)
        
        if st.button(f"No Track\n{c['NoTrack']}", key=f"n_{i}", help=f"Filter {c['NoTrack']} approved RMAs without tracking numbers"):
            handle_filter_click(store['url'], "NoTrack")
        if st.button(f"🚩 Flagged\n{c['Flagged']}", key=f"f_{i}", help=f"Filter {c['Flagged']} RMAs with 'flagged' in comments"):
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
        display_df = df_view[["No", "Store Name", "RMA ID", "Order", "Status", "TrackingNumber", 
                             "TrackingStatus", "Created", "Updated", "Days", 
                             "Store URL", "DisplayTrack", "shipment_id", "full_data", "_rma_id_text"]].copy()
        
        # Columns to display in table (10 columns)
        table_display_cols = ["No", "Store Name", "RMA ID", "Order", "Status", "TrackingNumber", 
                              "TrackingStatus", "Created", "Updated", "Days"]
        
        # Styling function - must return styles for exactly 10 columns
        def highlight_missing_tracking(row):
            # Access the full display_df using the index
            full_row = display_df.loc[row.name]
            if full_row.get("Status") == "Approved" and not full_row.get("DisplayTrack"):
                return ["background-color: rgba(220, 38, 38, 0.35); color: #fee2e2;"] * len(row)
            return [""] * len(row)
        
        styled_table = display_df[table_display_cols].style.apply(
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
        selection = sel_event.get("selection", {})
        sel_rows = selection.get("rows", [])
        if sel_rows:
            idx = int(sel_rows[0])
            show_rma_actions_dialog(display_df.iloc[idx])
    else:
        st.info("No matching records.")
else:
    st.info("No data available. Click 'Sync All Data' to load RMAs.")
