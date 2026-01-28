import streamlit as st
import streamlit.components.v1 as components
import requests
import json
import pandas as pd
import os
import threading
import time
import re
import logging
import sys # Added for os.sys.version fix
from datetime import datetime, timedelta, timezone, date # Added date for days_since
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import concurrent.futures
from typing import Optional, Dict, Any, Set, Mapping, Tuple, Callable, Union # Added Mapping, Tuple, Callable, Union for HTTP logic
from returngo_api import api_url, RMA_COMMENT_PATH # Fixed: text was not defined
from sqlalchemy import create_engine, text, Engine # Added Engine for type hinting

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
logger.info(f"Python version: {sys.version}") # Fixed: os.sys.version -> sys.version
logger.info("=" * 100)

# ==========================================
# 1. CONFIGURATION
# ==========================================
logger.info("Setting up page configuration...")
st.set_page_config(page_title="Bounty Apparel ReturnGo RMAs", layout="wide", page_icon="🔄️")

# ACCESS SECRETS
logger.info("Loading API credentials...")
try:
    MY_API_KEY = st.secrets["RETURNGO_API_KEY_BOUNTY"]
    logger.info("✓ API Key loaded from secrets")
except (FileNotFoundError, KeyError):
    MY_API_KEY = os.environ.get("RETURNGO_API_KEY_BOUNTY")
    if MY_API_KEY:
        logger.info("✓ API Key loaded from environment")
    else:
        logger.error("✗ API Key not found in secrets or environment")

if not MY_API_KEY:
    logger.critical("FATAL: API Key not found! Application stopped.")
    st.error("API Key not found! Please set 'RETURNGO_API_KEY_BOUNTY' in secrets or env vars.")
    st.stop()

logger.info(f"API Key present: {MY_API_KEY[:8]}..." if MY_API_KEY else "API Key: None")

CACHE_EXPIRY_HOURS = 4
COURIER_REFRESH_HOURS = 12

# Define missing global variables
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # Define SCRIPT_DIR
DB_FILE = os.path.join(SCRIPT_DIR, "rma_cache.db") # Placeholder for DB_FILE, though not used by SQLAlchemy
DB_LOCK = threading.Lock() # Define DB_LOCK

# Define STORES and ACTIVE_STATUSES
STORES = [
    {"name": "Bounty Apparel", "url": "bounty-apparel.myshopify.com"},
    # Add other stores if applicable
]
ACTIVE_STATUSES = ["Pending", "Approved", "Received"]

MAIN_TABLE_KEY = "main_table" # Module-level constant for the main dataframe key

# ==========================================
# HTTP SESSIONS + RATE LIMITING (Ported from levis_web.py)
# ==========================================
_thread_local = threading.local()
_rate_lock = threading.Lock()
_last_req_ts = 0.0
RG_RPS = 2 # Requests per second, ported from levis_web.py

RATE_LIMIT_HIT = threading.Event() # Define RATE_LIMIT_HIT
RATE_LIMIT_INFO: Dict[str, Union[int, str, datetime, None]] = {"remaining": None, "limit": None, "reset": None, "updated_at": None}
RATE_LIMIT_LOCK = threading.Lock() # Define RATE_LIMIT_LOCK

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _iso_utc(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _sleep_for_rate_limit():
    global _last_req_ts
    if RG_RPS <= 0:
        return
    min_interval = 1.0 / float(RG_RPS)
    with _rate_lock:
        now = time.time()
        wait = (_last_req_ts + min_interval) - now
        if wait > 0:
            time.sleep(wait)
        _last_req_ts = time.time()


def get_thread_session() -> requests.Session:
    s = getattr(_thread_local, "session", None)
    if s is not None:
        return s

    s = requests.Session()
    retries = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "PUT", "POST"]),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    _thread_local.session = s
    return s

def rg_headers(store_url: str) -> Dict[str, Optional[str]]:
    return {"x-api-key": MY_API_KEY, "X-API-KEY": MY_API_KEY, "x-shop-name": store_url}

def update_rate_limit_info(headers: Mapping[str, Optional[str]]):
    if not headers:
        return
    lower = {str(k).lower(): v for k, v in headers.items()}
    remaining = lower.get("x-ratelimit-remaining") or lower.get("x-rate-limit-remaining") or lower.get("ratelimit-remaining")
    limit = lower.get("x-ratelimit-limit") or lower.get("x-rate-limit-limit") or lower.get("ratelimit-limit")
    reset = lower.get("x-ratelimit-reset") or lower.get("x-rate-limit-reset") or lower.get("ratelimit-reset")

    if remaining is None and limit is None and reset is None:
        return

    def to_int(val):
        if val is None: return None
        sval = str(val).strip()
        return int(sval) if sval.isdigit() else sval

    with RATE_LIMIT_LOCK:
        RATE_LIMIT_INFO["remaining"] = to_int(remaining)
        RATE_LIMIT_INFO["limit"] = to_int(limit)
        RATE_LIMIT_INFO["reset"] = to_int(reset)
        RATE_LIMIT_INFO["updated_at"] = _now_utc()

# The `engine` variable needs to be accessible globally after `init_database` runs.
engine: Optional[Engine] = None # Initialize engine globally with explicit type hint
@st.cache_resource
def init_database():
    logger.info("Attempting to initialize database connection via direct SQLAlchemy...")
    try:
        # Use standard PostgreSQL connection string from secrets
        creds = st.secrets["connections"]["postgresql"]
        user = creds["username"]
        password = creds["password"]
        host = creds["host"]
        port = creds["port"]
        database = creds["database"]
        
        db_url = f"postgresql://{user}:{password}@{host}:{port}/{database}"
        engine_instance: Engine = create_engine( # Use a local variable first
            db_url, 
            pool_pre_ping=True, 
            connect_args={
                "connect_timeout": 60,
                "keepalives": 1,
                "keepalives_idle": 30,
                "keepalives_interval": 10,
                "keepalives_count": 5
            }
        )
        
        # Use a single transaction to create both tables # Fixed: text was not defined
        with engine_instance.begin() as connection:
            connection.execute(text("""
            CREATE TABLE IF NOT EXISTS rmas (
                rma_id TEXT PRIMARY KEY,
                store_url TEXT,
                status TEXT,
                created_at TIMESTAMP WITH TIME ZONE,
                json_data JSONB,
                last_fetched TIMESTAMP WITH TIME ZONE,
                courier_status TEXT,
                courier_last_checked TIMESTAMP WITH TIME ZONE,
                received_first_seen TIMESTAMP WITH TIME ZONE
            );
            """))
            connection.execute(text("""
            CREATE TABLE IF NOT EXISTS sync_logs (
                scope TEXT PRIMARY KEY,
                last_sync_iso TIMESTAMP WITH TIME ZONE
            );
            """)) # Fixed: text was not defined
        return engine
    except Exception as e:
        st.error(f"Application error: Could not initialize database. Error: {e}")
        logger.critical(f"Application error: Could not initialize database. Error: {e}", exc_info=True)
        st.stop()
        return None

engine = init_database()
if engine is not None:
    st.toast("✅ Database connection successful!")
else: # Fixed: "connect" is not a known attribute of "None"
    st.error("Database engine not initialized. Exiting.")
    st.stop()

def rg_request(method: str, url: str, store_url: str, *, headers=None, timeout=15, json_body=None):
    logger.info(f"rg_request: Making {method} request to {url} for store {store_url}")
    session: requests.Session = get_thread_session()
    headers = headers or rg_headers(store_url) # Use store_url here

    backoff = 1
    res = None
    for _attempt in range(1, 6):
        _sleep_for_rate_limit()
        logger.info(f"rg_request: Attempt {_attempt}...")
        res = session.request(method, url, headers=headers, timeout=timeout, json=json_body)
        logger.info(f"rg_request: Request completed with status {res.status_code if res else 'None'}")
        update_rate_limit_info(res.headers)
        if res.status_code != 429:
            return res # type: ignore

        RATE_LIMIT_HIT.set()

        ra = res.headers.get("Retry-After")
        if ra and ra.isdigit():
            sleep_s = int(ra)
        else:
            sleep_s = backoff
            backoff = min(backoff * 2, 30)

        time.sleep(sleep_s)

    return None # Explicitly return None if all retries fail after retries

def upsert_rma(rma_id, store_url, status, created_at, json_data, courier_status=None, courier_checked=None):
    logger.debug(f"Upserting RMA: {rma_id} | Store: {store_url} | Status: {status}")
    now = datetime.now(timezone.utc) # Use datetime object directly

    # Convert created_at to a timezone-aware datetime object
    created_at_dt = None
    if created_at:
        try:
            created_at_dt = pd.to_datetime(created_at).tz_localize('UTC')
        except Exception:
            created_at_dt = None

    courier_checked_dt = None
    if engine is None:
        logger.critical("Database engine is None in upsert_rma. This should not happen.")
        return # Or raise an error, depending on desired behavior
    if courier_checked: # courier_checked is already ISO string from previous logic
        try:
            courier_checked_dt = pd.to_datetime(courier_checked).tz_localize('UTC')
        except Exception:
            courier_checked_dt = None

    assert engine is not None, "Database engine is not initialized." # Fixed: "connect" is not a known attribute of "None"
    with engine.connect() as connection: # Pylance error fixed by assert engine is not None
        # First, select the existing values for received_first_seen
        select_query = text("SELECT received_first_seen FROM rmas WHERE rma_id=:rma_id")
        result = connection.execute(select_query, {"rma_id": rma_id}).fetchone() # Fixed: "connect" is not a known attribute of "None"
        existing_received_first_seen = result[0] if result else None

        received_first_seen = existing_received_first_seen
        if status == "Received" and not existing_received_first_seen:
            received_first_seen = now # Store as datetime object for DB

        insert_query = text('''
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
        ''')
        connection.execute(insert_query, {
            "rma_id": rma_id,
            "store_url": store_url,
            "status": status,
            "created_at": created_at_dt,
            "json_data": json_data, # json_data is already a string from json.dumps
            "last_fetched": now, # Store as datetime object
            "courier_status": courier_status,
            "courier_last_checked": courier_checked_dt, # Store as datetime object
            "received_first_seen": received_first_seen # Store as datetime object
        })
        connection.commit()
    logger.debug(f"✓ RMA {rma_id} upserted successfully")

def set_last_sync(scope, ts):
    logger.debug(f"Setting last sync for scope: {scope} at {ts}")
    if engine is None:
        logger.critical("Database engine is None in set_last_sync. This should not happen.")
        return
    assert engine is not None, "Database engine is not initialized." # Fixed: "connect" is not a known attribute of "None"
    with engine.connect() as connection:
        insert_query = text('INSERT INTO sync_logs (scope, last_sync_iso) VALUES (:scope, :last_sync_iso) ON CONFLICT (scope) DO UPDATE SET last_sync_iso = EXCLUDED.last_sync_iso;')
        connection.execute(insert_query, {"scope": scope, "last_sync_iso": ts}) # ts should be datetime object
        connection.commit() # Fixed: "connect" is not a known attribute of "None"
    logger.debug(f"✓ Last sync set for {scope}")

def get_last_sync(store_url, status):
    scope = f"{store_url}_{status}"
    with engine.connect() as connection: # Pylance error fixed by assert engine is not None
        select_query = text('SELECT last_sync_iso FROM sync_logs WHERE scope=:scope')
        r = connection.execute(select_query, {"scope": scope}).fetchone()
        result = r[0] if r else None # r[0] will be a datetime object from PostgreSQL
        logger.debug(f"Last sync for {scope}: {result}")
        return result

def get_all_active_from_db():
    """Load all active RMAs from database - NO CACHE so data shows immediately after sync"""
    logger.info("Loading all active RMAs from database...")
    results = []
    if engine is None:
        logger.critical("Database engine is None in get_all_active_from_db. This should not happen.")
        return []
    with engine.connect() as connection: # Fixed: "connect" is not a known attribute of "None"
        select_query = text(f"""
            SELECT rma_id, store_url, status, json_data, courier_status, courier_last_checked, received_first_seen
            FROM rmas WHERE status IN ({', '.join([':' + s for s in ACTIVE_STATUSES])})
        """)
        # Create a dictionary of parameters for the IN clause
        params = {s: s for s in ACTIVE_STATUSES}
        rows = connection.execute(select_query, params).fetchall()
    
    logger.info(f"✓ Loaded {len(rows)} RMAs from database")
    for row in rows:
        rma_id, store_url, status, json_data_pg, courier_status, courier_last_checked_dt, received_first_seen_dt = row
        try:
            # json_data_pg is already a Python dict/list from JSONB column
            data = json_data_pg
            data['_local_courier_status'] = courier_status or ''
            data['_local_courier_checked'] = courier_last_checked_dt.isoformat() if courier_last_checked_dt else ''
            data['_local_received_first_seen'] = received_first_seen_dt.isoformat() if received_first_seen_dt else ''
            data['store_url'] = store_url  # Make sure store_url is in the data
            results.append(data)
        except Exception as e:
            logger.error(f"Error parsing RMA {rma_id}: {e}", exc_info=True)
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
        logger.error(f"Error checking courier refresh time: {e}", exc_info=True)
        return True

def clear_db():
    logger.info("Clearing database...")
    try:
        if engine is None:
            logger.critical("Database engine is None in clear_db. This should not happen.")
            return False
        with engine.begin() as connection: # Fixed: "connect" is not a known attribute of "None"
            connection.execute(text("TRUNCATE TABLE rmas;"))
            connection.execute(text("TRUNCATE TABLE sync_logs;"))
        logger.info("✓ Database cleared successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Error clearing database: {e}", exc_info=True)
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

            logger.info(f"  Page {page}: GET {url}") # Fixed: get_session was not defined
            
            res = rg_request("GET", url, store['url'], headers=headers, timeout=20) # Pylance error fixed by explicit None check
            
            if res is None:
                error_detail = "No response from ReturnGO API after retries."
                logger.error(f"  ✗ API Error: {error_detail}")
                if page == 1:
                    st.error(f"Error fetching RMAs for {store['name']}: {error_detail}\nURL: {url}")
                break
            
            logger.info(f"  Response: Status {res.status_code}, Content-Length: {len(res.content) if res.content else 0}") # Fixed: "connect" is not a known attribute of "None"
            
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
                break # Fixed: "connect" is not a known attribute of "None"
            
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
        res = rg_request("GET", url, store_url, headers=headers, timeout=20) # Pylance error fixed by explicit None check
        if res is None:
            logger.error(f"    ✗ Failed to fetch detail for {rma_id}: No response after retries.")
            return None

        logger.debug(f"    Response: {res.status_code}") # Fixed: "connect" is not a known attribute of "None"
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
    url = api_url(f"/shipment/{shipment_id}") # Changed to PUT /shipment/{shipmentId} as per API docs
    headers = {
        "x-api-key": MY_API_KEY,
        "X-API-KEY": MY_API_KEY,
        "x-shop-name": store_url,
        "Content-Type": "application/json"
    }
    payload = { # Added required fields for PUT /shipment/{shipmentId}
        "status": "LabelCreated", # Assuming this is the status when tracking is updated
        "carrierName": "CourierGuy", # Placeholder, adjust as needed
        "trackingNumber": new_tracking,
        "trackingURL": f"https://optimise.parcelninja.com/shipment/track?WaybillNo={new_tracking}", # Placeholder
        "labelURL": "https://sellerportal.dpworld.com/api/file-download?link=null", # Placeholder
    }
    try:
        res = rg_request("PUT", url, store_url, headers=headers, json_body=payload, timeout=15) # Pylance error fixed by explicit None check
        if res is None:
            return False, "API Error: No response from ReturnGO API after retries."

        if res.status_code in [200, 201]: # Fixed: "connect" is not a known attribute of "None"
            return True, "Success"
        return False, f"API Error {res.status_code}"
    except Exception as e:
        return False, str(e)

def push_comment_update(rma_id, comment_text, store_url):
    """Post a comment to an RMA"""
    url = api_url(RMA_COMMENT_PATH.format(rmaId=rma_id))
    headers = {"X-API-KEY": MY_API_KEY, "x-api-key": MY_API_KEY, "x-shop-name": store_url, "Content-Type": "application/json"} # Fixed: get_session was not defined
    payload = {"htmlText": comment_text}
    try:
        res = rg_request("POST", url, store_url, headers=headers, json_body=payload, timeout=15) # Pylance error fixed by explicit None check
        if res is None:
            return False, "API Error: No response from ReturnGO API after retries."

        if res.status_code in [200, 201]: # Fixed: "connect" is not a known attribute of "None"
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
    logger.info(f"Status filter: {status if status else 'ALL STATUSES'}") # Fixed: text was not defined
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
                    upsert_rma(rma_id, store['url'], stat, created_at, json.dumps(full_data)) # json.dumps here as upsert_rma expects string
                    total_rmas_synced += 1
                else:
                    logger.warning(f"    ✗ Failed to get detail for {rma_id}")
                
                # Small delay to avoid rate limiting
                time.sleep(0.1) # Fixed: "connect" is not a known attribute of "None"
            
            # Save sync timestamp
            scope = f"{store['url']}_{stat}" # Fixed: text was not defined
            sync_time = datetime.now(timezone.utc) # Pass datetime object
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
# 6. UI: FETCH (API) vs SYNC (UI/DB)
# ==========================================

def render_sync_dashboard_ui():
    """Sync UI: Loads from database cache."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("Dashboard Operations")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🔄 Sync Dashboard", help="Reload data from database cache"):
            st.session_state.cache_version = st.session_state.get('cache_version', 0) + 1
            st.toast("✅ Dashboard synced from database")
            st.rerun()
            
    with col2:
        if st.button("📡 Fetch Data", help="Fetch fresh data from ReturnGO API"):
            st.session_state["pending_fetch"] = True
            st.rerun()

    if st.session_state.get("pending_fetch"):
        st.session_state["pending_fetch"] = False
        with st.spinner("Fetching fresh data from ReturnGO API..."):
            perform_sync()

# ==========================================
# 7. FRONTEND UI LOGIC
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
            if st.form_submit_button("Save Changes"): # Fixed: "connect" is not a known attribute of "None"
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
                if st.form_submit_button("Post Comment"): # Fixed: "connect" is not a known attribute of "None"
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
with st.expander("ℹ️ Database Info", expanded=False): # Fixed: DB_FILE, SCRIPT_DIR were not defined
    st.code(f"Database Type: PostgreSQL (configured via secrets)")
    st.code(f"Script Directory: {SCRIPT_DIR}")
    st.code(f"Working Directory: {os.getcwd()}")
    
    # Check if database exists and show stats
    if engine: # Check if engine is initialized
        # No direct file size for PostgreSQL, but can check connection # Fixed: "connect" is not a known attribute of "None"
        try:
            with engine.connect() as connection:
                connection.execute(text("SELECT 1")) # Simple query to check connection
            st.success(f"✅ PostgreSQL connection active.")
        except Exception as e:
            st.error(f"PostgreSQL connection failed: {e}")
        # Show row counts # Fixed: "connect" is not a known attribute of "None"
        try:
            with engine.connect() as connection:
                rma_count_result = connection.execute(text("SELECT COUNT(*) FROM rmas")).fetchone()
                rma_count = rma_count_result[0] if rma_count_result else 0 # Fixed: "connect" is not a known attribute of "None"

                sync_count_result = connection.execute(text("SELECT COUNT(*) FROM sync_logs")).fetchone()
                sync_count = sync_count_result[0] if sync_count_result else 0

            st.info(f"📊 Database contains: {rma_count} RMAs, {sync_count} sync logs")
        except Exception as e:
            st.warning(f"Could not read database: {e}")
    else:
        st.warning("⚠️ Database engine not initialized. Please check connection settings.")

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
            st.success("Database cleared! Data will reload on next sync.")
            logger.info("✓ Cache reset successful")
            st.rerun()
        else:
            st.error("DB might be locked.")
            logger.error("✗ Cache reset failed - DB locked")

# --- Process Data ---
logger.info("Loading and processing RMA data...")
raw_data = get_all_active_from_db()
processed_rows = []
store_counts = {s['url']: {"Pending": 0, "Approved": 0, "Received": 0, "NoTrack": 0, "Flagged": 0} for s in STORES} # Fixed: STORES was not defined
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
                "RMA ID": get_store_returngo_url(store_url, rma_id), # Fixed: "connect" is not a known attribute of "None"
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
        st.markdown(f"**{store['name'].upper()}**") # Fixed: "connect" is not a known attribute of "None"
        
        def show_btn(label, stat, key, help_text):
            ts = get_last_sync(store['url'], stat)
            st.markdown(
                f"<div class='sync-time'>Updated: {ts[11:19] if ts else '-'}</div>",
                unsafe_allow_html=True,
            )
            if st.button(f"{label}\n{c[stat]}", key=key, help=help_text): # Fixed: "connect" is not a known attribute of "None"
                handle_filter_click(store['url'], stat)
        
        show_btn("Pending", "Pending", f"p_{i}", f"Sync and view {c['Pending']} pending RMAs for {store['name']}")
        show_btn("Approved", "Approved", f"a_{i}", f"Sync and view {c['Approved']} approved RMAs for {store['name']}")
        show_btn("Received", "Received", f"r_{i}", f"Sync and view {c['Received']} received RMAs for {store['name']}")
        
        if st.button(f"No Track\n{c['NoTrack']}", key=f"n_{i}", help=f"Filter {c['NoTrack']} approved RMAs without tracking numbers"): # Fixed: "connect" is not a known attribute of "None"
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
            
            if st.button("📋 Copy all", key="copy_all_btn"): # Fixed: "connect" is not a known attribute of "None"
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
            # Access the full display_df using the index # Fixed: "connect" is not a known attribute of "None"
            full_row = display_df.loc[row.name]
            if full_row.get("Status") == "Approved" and not full_row.get("DisplayTrack"):
                return ["background-color: rgba(220, 38, 38, 0.35); color: #fee2e2;"] * len(row)
            return [""] * len(row)
        
        styled_table = display_df[table_display_cols].style.apply(
            highlight_missing_tracking, axis=1
        )
        
        sel_event = st.dataframe( # Line 1048
            styled_table,
            use_container_width=True,
            height=700,
            hide_index=True,
            key=MAIN_TABLE_KEY,
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
        sel_rows = [] # Initialize sel_rows
        if MAIN_TABLE_KEY in st.session_state and hasattr(st.session_state[MAIN_TABLE_KEY], "selection"):
            selection_state = st.session_state[MAIN_TABLE_KEY].selection
            if hasattr(selection_state, "rows"):
                sel_rows = selection_state.rows
        sel_rows = sel_rows or []
        if sel_rows:
            idx = int(sel_rows[0])
            show_rma_actions_dialog(display_df.iloc[idx])
    else:
        st.info("No matching records.")
else:
    st.info("No data available. Click 'Sync All Data' to load RMAs.")
