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
import sys
from datetime import datetime, timedelta, timezone, date
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import concurrent.futures
from typing import Optional, Dict, Any, Set, Mapping, Tuple, Callable, Union
from returngo_api import api_url, RMA_COMMENT_PATH
from sqlalchemy import create_engine, text, Engine

# ==========================================
# LOGGING SETUP
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s',
    handlers=[logging.StreamHandler()],
    force=True
)
logger = logging.getLogger(__name__)

logger.info("=" * 100)
logger.info("RMA WEB APPLICATION STARTING")
logger.info(f"Timestamp: {datetime.now().isoformat()}")
logger.info(f"Python version: {sys.version}")
logger.info("=" * 100)

# ==========================================
# 1. CONFIGURATION
# ==========================================
logger.info("Setting up page configuration...")
st.set_page_config(page_title="Bounty Apparel ReturnGo RMAs", layout="wide", page_icon="🔄️")

# Custom CSS for better UI
st.markdown("""
<style>
    /* Store filter buttons styling */
    .store-section {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
    }
    
    .store-header {
        font-size: 16px;
        font-weight: 600;
        margin-bottom: 10px;
        color: #ffffff;
        text-align: center;
    }
    
    .sync-time {
        font-size: 11px;
        color: rgba(255, 255, 255, 0.6);
        text-align: center;
        margin-bottom: 5px;
    }
    
    /* Row count styling */
    .rows-count {
        font-size: 14px;
        padding: 8px;
        color: rgba(255, 255, 255, 0.8);
    }
    
    .rows-count .value {
        font-weight: 600;
        color: #ffffff;
    }
    
    /* Button container */
    .stButton button {
        width: 100%;
        white-space: pre-wrap;
        height: auto;
        min-height: 45px;
    }
    
    /* Filter info banner */
    .filter-banner {
        background: rgba(59, 130, 246, 0.1);
        border-left: 4px solid #3b82f6;
        padding: 12px 16px;
        margin: 10px 0;
        border-radius: 4px;
    }
    
    .filter-banner-text {
        color: #ffffff;
        font-size: 14px;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(SCRIPT_DIR, "rma_cache.db")
DB_LOCK = threading.Lock()

STORES = [
    {"name": "Diesel", "url": "diesel-dev-south-africa.myshopify.com"},
    {"name": "Hurley", "url": "hurley-dev-south-africa.myshopify.com"},
    {"name": "Jeep Apparel", "url": "jeep-apparel-dev-south-africa.myshopify.com"},
    {"name": "Reebok", "url": "reebok-dev-south-africa.myshopify.com"},
    {"name": "Superdry", "url": "superdry-dev-south-africa.myshopify.com"}
]

ACTIVE_STATUSES = ["Pending", "Approved", "Received"]
MAIN_TABLE_KEY = "main_table"

# ==========================================
# HTTP SESSIONS + RATE LIMITING
# ==========================================
_thread_local = threading.local()
_rate_lock = threading.Lock()
_last_req_ts = 0.0
RG_RPS = 2
RATE_LIMIT_HIT = threading.Event()
RATE_LIMIT_INFO: Dict[str, Union[int, str, datetime, None]] = {
    "remaining": None, "limit": None, "reset": None, "updated_at": None
}
RATE_LIMIT_LOCK = threading.Lock()

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

engine: Optional[Engine] = None

@st.cache_resource
def init_database() -> Optional[Engine]:
    """Initializes a robust, pooled database connection using settings from Streamlit secrets."""
    logger.info("Initializing database connection from Streamlit secrets...")
    try:
        creds = st.secrets["connections"]["postgresql"]
        dialect = creds.get("dialect", "postgresql")
        driver = creds.get("driver")
        user = creds["username"]
        password = creds["password"]
        host = creds["host"]
        port = creds["port"]
        database = creds["database"]

        db_url = f"{dialect}{f'+{driver}' if driver else ''}://{user}:{password}@{host}:{port}/{database}"
        connect_args = {"timeout": 60}

        if driver == "pg8000" and creds.get("sslmode") == "require":
            connect_args["ssl_context"] = True
        elif "sslmode" in creds:
            db_url += f"?sslmode={creds['sslmode']}"

        logger.info(f"Connecting to database: {dialect} on {host}:{port}/{database}")

        engine_instance = create_engine(
            db_url,
            pool_pre_ping=True,
            connect_args=connect_args,
        )

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
            """))
        
        logger.info("Database tables verified and connection is ready.")
        return engine_instance

    except KeyError as e:
        logger.critical(f"Database configuration error: Missing key {e}")
        st.error(f"Application error: Database configuration is missing required key: {e}")
        st.stop()
        return None
    except Exception as e:
        logger.critical(f"Fatal error during database initialization: {e}", exc_info=True)
        st.error(f"Application error: Could not connect to the database. Details: {e}")
        st.stop()
        return None

engine = init_database()
if engine is not None:
    st.toast("✅ Database connection successful!")
else:
    st.error("Database engine not initialized. Exiting.")
    st.stop()

def rg_request(method: str, url: str, store_url: str, *, headers=None, timeout=15, json_body=None):
    logger.info(f"rg_request: Making {method} request to {url} for store {store_url}")
    session: requests.Session = get_thread_session()
    headers = headers or rg_headers(store_url)
    backoff = 1
    res = None
    
    for _attempt in range(1, 6):
        _sleep_for_rate_limit()
        logger.info(f"rg_request: Attempt {_attempt}...")
        res = session.request(method, url, headers=headers, timeout=timeout, json=json_body)
        logger.info(f"rg_request: Request completed with status {res.status_code if res else 'None'}")
        update_rate_limit_info(res.headers)
        
        if res.status_code != 429:
            return res
        
        RATE_LIMIT_HIT.set()
        ra = res.headers.get("Retry-After")
        if ra and ra.isdigit():
            sleep_s = int(ra)
        else:
            sleep_s = backoff
            backoff = min(backoff * 2, 30)
        time.sleep(sleep_s)
    
    return None

def upsert_rma(rma_id, store_url, status, created_at, json_data, courier_status=None, courier_checked=None):
    logger.debug(f"Upserting RMA: {rma_id} | Store: {store_url} | Status: {status}")
    now = datetime.now(timezone.utc)
    
    created_at_dt = None
    if created_at:
        try:
            created_at_dt = pd.to_datetime(created_at).tz_localize('UTC')
        except Exception:
            created_at_dt = None
    
    courier_checked_dt = None
    if engine is None:
        logger.critical("Database engine is None in upsert_rma.")
        return
    
    if courier_checked:
        try:
            courier_checked_dt = pd.to_datetime(courier_checked).tz_localize('UTC')
        except Exception:
            courier_checked_dt = None
    
    with engine.connect() as connection:
        select_query = text("SELECT received_first_seen FROM rmas WHERE rma_id=:rma_id")
        result = connection.execute(select_query, {"rma_id": rma_id}).fetchone()
        existing_received_first_seen = result[0] if result else None
        
        received_first_seen = existing_received_first_seen
        if status == "Received" and not existing_received_first_seen:
            received_first_seen = now
        
        insert_query = text('''
            INSERT INTO rmas (rma_id, store_url, status, created_at, json_data, last_fetched, courier_status, courier_last_checked, received_first_seen)
            VALUES (:rma_id, :store_url, :status, :created_at, :json_data, :last_fetched, :courier_status, :courier_last_checked, :received_first_seen)
            ON CONFLICT(rma_id) DO UPDATE SET
                store_url=excluded.store_url,
                status=excluded.status,
                created_at=excluded.created_at,
                json_data=excluded.json_data,
                last_fetched=excluded.last_fetched,
                courier_status=COALESCE(excluded.courier_status, rmas.courier_status),
                courier_last_checked=COALESCE(excluded.courier_last_checked, rmas.courier_last_checked),
                received_first_seen=excluded.received_first_seen
        ''')
        
        connection.execute(insert_query, {
            "rma_id": rma_id,
            "store_url": store_url,
            "status": status,
            "created_at": created_at_dt,
            "json_data": json_data,
            "last_fetched": now,
            "courier_status": courier_status,
            "courier_last_checked": courier_checked_dt,
            "received_first_seen": received_first_seen
        })
        connection.commit()
    
    logger.debug(f"✓ RMA {rma_id} upserted successfully")

def set_last_sync(scope, ts):
    logger.debug(f"Setting last sync for scope: {scope} at {ts}")
    if engine is None:
        logger.critical("Database engine is None in set_last_sync.")
        return
    
    with engine.connect() as connection:
        insert_query = text('INSERT INTO sync_logs (scope, last_sync_iso) VALUES (:scope, :last_sync_iso) ON CONFLICT (scope) DO UPDATE SET last_sync_iso = EXCLUDED.last_sync_iso;')
        connection.execute(insert_query, {"scope": scope, "last_sync_iso": ts})
        connection.commit()
    
    logger.debug(f"✓ Last sync set for {scope}")

def get_last_sync(store_url, status):
    scope = f"{store_url}_{status}"
    if engine is None:
        logger.critical("Database engine is None in get_last_sync.")
        return None
    
    with engine.connect() as connection:
        select_query = text('SELECT last_sync_iso FROM sync_logs WHERE scope=:scope')
        r = connection.execute(select_query, {"scope": scope}).fetchone()
        result = r[0] if r else None
        logger.debug(f"Last sync for {scope}: {result}")
        return result

def get_all_active_from_db():
    """Load all active RMAs from database"""
    logger.info("Loading all active RMAs from database...")
    results = []
    
    if engine is None:
        logger.critical("Database engine is None in get_all_active_from_db.")
        return []
    
    with engine.connect() as connection:
        select_query = text(f"""
            SELECT rma_id, store_url, status, json_data, courier_status, courier_last_checked, received_first_seen
            FROM rmas WHERE status IN ({', '.join([':' + s for s in ACTIVE_STATUSES])})
        """)
        params = {s: s for s in ACTIVE_STATUSES}
        rows = connection.execute(select_query, params).fetchall()
    
    logger.info(f"✓ Loaded {len(rows)} RMAs from database")
    
    for row in rows:
        rma_id, store_url, status, json_data_pg, courier_status, courier_last_checked_dt, received_first_seen_dt = row
        try:
            data = json_data_pg
            data['_local_courier_status'] = courier_status or ''
            data['_local_courier_checked'] = courier_last_checked_dt.isoformat() if courier_last_checked_dt else ''
            data['_local_received_first_seen'] = received_first_seen_dt.isoformat() if received_first_seen_dt else ''
            data['store_url'] = store_url
            results.append(data)
        except Exception as e:
            logger.error(f"Error parsing RMA {rma_id}: {e}", exc_info=True)
    
    logger.info(f"✓ Successfully parsed {len(results)} RMAs")
    return results

def should_refresh_courier(rma_data):
    """Check if courier status needs refreshing"""
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
            logger.critical("Database engine is None in clear_db.")
            return False
        
        with engine.begin() as connection:
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
    max_pages = 100
    
    headers = {
        "x-api-key": MY_API_KEY,
        "X-API-KEY": MY_API_KEY,
        "x-shop-name": store['url']
    }
    logger.debug(f"Headers prepared: x-shop-name={store['url']}, API key present")
    
    while page <= max_pages:
        try:
            base_url = f"{api_url('/rmas')}?status={status}&pagesize=500"
            url = f"{base_url}&cursor={cursor}" if cursor else base_url
            logger.info(f"  Page {page}: GET {url}")
            
            res = rg_request("GET", url, store['url'], headers=headers, timeout=20)
            
            if res is None:
                error_detail = "No response from ReturnGO API after retries."
                logger.error(f"  ✗ API Error: {error_detail}")
                if page == 1:
                    st.error(f"Error fetching RMAs for {store['name']}: {error_detail}\nURL: {url}")
                break
            
            logger.info(f"  Response: Status {res.status_code}, Content-Length: {len(res.content) if res.content else 0}")
            
            if res.status_code != 200:
                error_detail = ""
                try:
                    error_detail = res.text[:500] if res.text else "(empty response)"
                except:
                    pass
                logger.error(f"  ✗ API Error: {res.status_code}")
                logger.error(f"  Response body: {error_detail}")
                
                if page == 1:
                    st.error(f"Error fetching RMAs for {store['name']}: {res.status_code}\nURL: {url}\nResponse: {error_detail}")
                break
            
            data = res.json()
            rmas = data.get("rmas", [])
            
            logger.info(f"  ✓ Received {len(rmas)} RMAs on page {page}")
            
            if not rmas:
                logger.info(f"  No more RMAs, stopping pagination")
                break
            
            all_rmas.extend(rmas)
            
            cursor = data.get("next_cursor")
            if cursor:
                logger.debug(f"  Next cursor: {cursor[:50]}...")
            else:
                logger.info(f"  No next cursor, pagination complete")
                break
                
            page += 1
        except Exception as e:
            logger.error(f"  ✗ Exception on page {page}: {str(e)}", exc_info=True)
            if page == 1:
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
        res = rg_request("GET", url, store_url, headers=headers, timeout=20)
        if res is None:
            logger.error(f"    ✗ Failed to fetch detail for {rma_id}: No response after retries.")
            return None
        
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
    url = api_url(f"/shipment/{shipment_id}")
    headers = {
        "x-api-key": MY_API_KEY,
        "X-API-KEY": MY_API_KEY,
        "x-shop-name": store_url,
        "Content-Type": "application/json"
    }
    payload = {
        "status": "LabelCreated",
        "carrierName": "CourierGuy",
        "trackingNumber": new_tracking,
        "trackingURL": f"https://optimise.parcelninja.com/shipment/track?WaybillNo={new_tracking}",
        "labelURL": "https://sellerportal.dpworld.com/api/file-download?link=null",
    }
    
    try:
        res = rg_request("PUT", url, store_url, headers=headers, json_body=payload, timeout=15)
        if res is None:
            return False, "API Error: No response from ReturnGO API after retries."
        if res.status_code in [200, 201]:
            return True, "Success"
        return False, f"API Error {res.status_code}"
    except Exception as e:
        return False, str(e)

def push_comment_update(rma_id, comment_text, store_url):
    """Post a comment to an RMA"""
    url = api_url(RMA_COMMENT_PATH.format(rmaId=rma_id))
    headers = {
        "X-API-KEY": MY_API_KEY,
        "x-api-key": MY_API_KEY,
        "x-shop-name": store_url,
        "Content-Type": "application/json"
    }
    payload = {"htmlText": comment_text}
    
    try:
        res = rg_request("POST", url, store_url, headers=headers, json_body=payload, timeout=15)
        if res is None:
            return False, "API Error: No response from ReturnGO API after retries."
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
            
            rma_list = fetch_rma_list(store, stat)
            logger.info(f"  Retrieved {len(rma_list)} RMAs for {stat}")
            
            for idx, rma_summary in enumerate(rma_list):
                rma_id = rma_summary.get('rmaId')
                if not rma_id:
                    logger.warning(f"    Skipping RMA with no ID")
                    continue
                
                sub_progress_text = f"Syncing {store['name']} - {stat}... ({idx+1}/{len(rma_list)} RMAs)"
                status_text.text(sub_progress_text)
                
                full_data = fetch_rma_detail(rma_id, store['url'])
                if full_data:
                    created_at = rma_summary.get('createdAt', '')
                    upsert_rma(rma_id, store['url'], stat, created_at, json.dumps(full_data))
                    total_rmas_synced += 1
                else:
                    logger.warning(f"    ✗ Failed to get detail for {rma_id}")
                
                time.sleep(0.1)
            
            scope = f"{store['url']}_{stat}"
            sync_time = datetime.now(timezone.utc)
            set_last_sync(scope, sync_time)
            logger.info(f"  ✓ Sync timestamp saved for {scope}")
    
    progress_placeholder.empty()
    status_text.empty()
    
    logger.info("=" * 100)
    logger.info(f"SYNC OPERATION COMPLETE: {total_rmas_synced} RMAs synced")
    logger.info("=" * 100)
    
    st.toast(f"✅ Sync Complete! {total_rmas_synced} RMAs synced.", icon="🔄")
    st.session_state['show_toast'] = True
    time.sleep(0.5)
    
    logger.info("Triggering rerun...")
    st.rerun()

# ==========================================
# 5. UTILITY FUNCTIONS
# ==========================================
def days_since(date_str, today=None):
    """Calculate days since a given date"""
    if not date_str or date_str == "N/A":
        return 999999
    try:
        if today is None:
            today = datetime.now(timezone.utc).date()
        date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00')).date()
        return (today - date_obj).days
    except Exception:
        return 999999

def get_store_returngo_url(store_url, rma_id):
    """Generate the ReturnGO URL for a specific RMA"""
    return f"https://app.returngo.ai/dashboard/returns?filter_status=open&rmaid={rma_id}"

def extract_date_from_events(events, event_name):
    """Extract date from events list"""
    if not events:
        return "N/A"
    for evt in events:
        if evt.get('eventName') == event_name:
            date_val = evt.get('eventDate', 'N/A')
            return str(date_val)[:10] if date_val != "N/A" else "N/A"
    return "N/A"

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

# Database info expander
with st.expander("ℹ️ Database Info", expanded=False):
    st.code(f"Database Type: PostgreSQL (configured via secrets)")
    st.code(f"Script Directory: {SCRIPT_DIR}")
    st.code(f"Working Directory: {os.getcwd()}")
    
    if engine:
        try:
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            st.success(f"✅ PostgreSQL connection active.")
        except Exception as e:
            st.error(f"PostgreSQL connection failed: {e}")
        
        try:
            with engine.connect() as connection:
                rma_count_result = connection.execute(text("SELECT COUNT(*) FROM rmas")).fetchone()
                rma_count = rma_count_result[0] if rma_count_result else 0
                sync_count_result = connection.execute(text("SELECT COUNT(*) FROM sync_logs")).fetchone()
                sync_count = sync_count_result[0] if sync_count_result else 0
            st.info(f"📊 Database contains: {rma_count} RMAs, {sync_count} sync logs")
        except Exception as e:
            st.warning(f"Could not read database: {e}")
    else:
        st.warning("⚠️ Database engine not initialized. Please check connection settings.")

# Main header and controls
col1, col2 = st.columns([3, 1])
with col1:
    st.title("ReturnGo RMAs 🔄️")
    search_query = st.text_input("🔍 Search Order, RMA, or Tracking", placeholder="Type to search...")
    if search_query:
        logger.info(f"Search query: {search_query}")

with col2:
    if st.button("🔄 Sync All Data", type="primary", help="Fetch latest data from ReturnGO API", use_container_width=True):
        logger.info("USER ACTION: Sync All Data button clicked")
        perform_sync()
    
    if st.button("🔄 Update Display", type="secondary", help="Refresh display from database", use_container_width=True):
        logger.info("USER ACTION: Update Display button clicked")
        st.cache_data.clear()
        st.cache_resource.clear()
        st.session_state.pop("main_table", None)
        st.session_state['show_update_toast'] = True
        logger.info("Caches cleared, triggering rerun")
        st.rerun()
    
    if st.button("🗑️ Reset Cache", type="secondary", help="⚠️ Delete ALL cached data", use_container_width=True):
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

st.divider()

# --- Process Data ---
logger.info("Loading and processing RMA data...")
raw_data = get_all_active_from_db()
processed_rows = []
store_counts = {s['url']: {
    "Pending": 0, 
    "Approved": 0, 
    "Received": 0, 
    "NoTrack": 0, 
    "Flagged": 0
} for s in STORES}

today = datetime.now(timezone.utc).date()
logger.info(f"Processing {len(raw_data)} RMAs...")

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
    events = summary.get('events', [])
    
    status = summary.get('status', 'Unknown')
    rma_id = summary.get('rmaId', 'N/A')
    order_name = summary.get('order_name', 'N/A')
    resolution_type = summary.get('resolutionType', 'N/A')
    
    track_nums = [s.get('trackingNumber') for s in shipments if s.get('trackingNumber')]
    track_str = ", ".join(track_nums) if track_nums else ""
    shipment_id = shipments[0].get('shipmentId') if shipments else None
    local_status = rma.get('_local_courier_status', '')
    
    track_link_url = f"https://optimise.parcelninja.com/shipment/track?WaybillNo={track_nums[0]}&ref={track_nums[0]}" if track_nums else ""
    
    # Extract dates from events
    created_at = summary.get('createdAt')
    if not created_at:
        created_at = extract_date_from_events(events, 'RMA_CREATED')
    
    approved_date = extract_date_from_events(events, 'RMA_APPROVED')
    received_date = extract_date_from_events(events, 'RMA_RECEIVED')
    
    requested_date = str(created_at)[:10] if created_at and created_at != "N/A" else "N/A"
    
    u_at = rma.get('lastUpdated')
    days_since_requested = days_since(requested_date, today=today)
    
    # Check if resolution actioned
    resolution_actioned = "No"
    if any(evt.get('eventName') in ['REFUND_ISSUED', 'EXCHANGE_CREATED'] for evt in events):
        resolution_actioned = "Yes"
    
    # Check for tracking update
    is_tracking_updated = "No"
    if track_str:
        is_tracking_updated = "Yes"
    
    # Check for failures
    failures = []
    for c in comments:
        comment_text = c.get('htmlText', '').lower()
        if 'refund failed' in comment_text or 'refund failure' in comment_text:
            failures.append("Refund")
        if 'upload failed' in comment_text or 'upload failure' in comment_text:
            failures.append("Upload")
        if 'shipment failed' in comment_text or 'shipment failure' in comment_text:
            failures.append("Shipment")
    failures_str = ", ".join(list(set(failures))) if failures else ""
    
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
                "Current Status": status,
                "Store URL": store_url,
                "Tracking Number": track_link_url,
                "Tracking Status": local_status,
                "Requested date": requested_date,
                "Approved date": approved_date,
                "Received date": received_date,
                "Days since requested": days_since_requested,
                "resolutionType": resolution_type,
                "Resolution actioned": resolution_actioned,
                "Is_tracking_updated": is_tracking_updated,
                "failures": failures_str,
                "DisplayTrack": track_str,
                "shipment_id": shipment_id,
                "full_data": rma,
                "_rma_id_text": rma_id,
                "is_nt": is_nt,
                "is_fg": is_fg
            })

# --- Store Filter Buttons (Improved Layout) ---
st.markdown("### 📊 Store Filters")

# Create 5 columns for the 5 stores
cols = st.columns(len(STORES))

for i, store in enumerate(STORES):
    c = store_counts[store['url']]
    
    with cols[i]:
        # Store section container
        st.markdown(f'<div class="store-section">', unsafe_allow_html=True)
        st.markdown(f'<div class="store-header">{store["name"].upper()}</div>', unsafe_allow_html=True)
        
        def show_btn(label, stat, key, help_text):
            ts = get_last_sync(store['url'], stat)
            st.markdown(
                f"<div class='sync-time'>Updated: {ts.strftime('%H:%M:%S') if ts else '-'}</div>",
                unsafe_allow_html=True,
            )
            if st.button(f"{label}\n{c[stat]}", key=key, help=help_text, use_container_width=True):
                handle_filter_click(store['url'], stat)
        
        # Status buttons
        show_btn("⏳ Pending", "Pending", f"p_{i}", f"Sync and view {c['Pending']} pending RMAs for {store['name']}")
        show_btn("✅ Approved", "Approved", f"a_{i}", f"Sync and view {c['Approved']} approved RMAs for {store['name']}")
        show_btn("📦 Received", "Received", f"r_{i}", f"Sync and view {c['Received']} received RMAs for {store['name']}")
        
        # Special filter buttons
        if st.button(f"🚫 No Track\n{c['NoTrack']}", key=f"n_{i}", help=f"Filter {c['NoTrack']} approved RMAs without tracking", use_container_width=True):
            handle_filter_click(store['url'], "NoTrack")
        
        if st.button(f"🚩 Flagged\n{c['Flagged']}", key=f"f_{i}", help=f"Filter {c['Flagged']} flagged RMAs", use_container_width=True):
            handle_filter_click(store['url'], "Flagged")
        
        st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# --- Active Filter Display ---
cur = st.session_state.filter_state
if cur['store'] or cur['status'] != "All":
    store_name = next((s['name'] for s in STORES if s['url'] == cur['store']), "All Stores") if cur['store'] else "All Stores"
    filter_text = f"🔍 Active Filter: {store_name} - {cur['status']}"
    st.markdown(f'<div class="filter-banner"><p class="filter-banner-text">{filter_text}</p></div>', unsafe_allow_html=True)
    
    if st.button("Clear Filter", key="clear_filter_btn"):
        st.session_state.filter_state = {"store": None, "status": "All"}
        st.rerun()

# --- Filter and Display Data ---
df_view = pd.DataFrame(processed_rows)

if not df_view.empty:
    if cur['store']:
        df_view = df_view[df_view['Store URL'] == cur['store']]
    
    f_stat = cur['status']
    if f_stat == "Pending":
        df_view = df_view[df_view['Current Status'] == 'Pending']
    elif f_stat == "Approved":
        df_view = df_view[df_view['Current Status'] == 'Approved']
    elif f_stat == "Received":
        df_view = df_view[df_view['Current Status'] == 'Received']
    elif f_stat == "NoTrack":
        df_view = df_view[df_view['is_nt'] == True]
    elif f_stat == "Flagged":
        df_view = df_view[df_view['is_fg'] == True]
    
    if not df_view.empty:
        # Sort by days since requested
        df_view = df_view.sort_values(by="Days since requested", ascending=False).reset_index(drop=True)
        df_view['No'] = range(1, len(df_view) + 1)
        df_view['No'] = df_view['No'].astype(str)
        
        # Show row count and actions
        total_rows = len(df_view)
        count_col, action_col = st.columns([5, 3], vertical_alignment="center")
        
        with count_col:
            st.markdown(
                f"<div class='rows-count'>Rows in table: <span class='value'>{total_rows}</span></div>",
                unsafe_allow_html=True,
            )
        
        with action_col:
            # Copy all button
            display_cols = [
                "No", "Store Name", "RMA ID", "Order", "Current Status",
                "Tracking Number", "Tracking Status", "Requested date",
                "Approved date", "Received date", "Days since requested",
                "resolutionType", "Resolution actioned", "Is_tracking_updated", "failures"
            ]
            
            tsv_rows = [display_cols] + df_view[display_cols].astype(str).values.tolist()
            tsv_text = "\n".join(["\t".join(row) for row in tsv_rows])
            
            if st.button("📋 Copy all", key="copy_all_btn", use_container_width=True):
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
        
        # Display Table
        display_df = df_view[[
            "No", "Store Name", "RMA ID", "Order", "Current Status",
            "Tracking Number", "Tracking Status", "Requested date",
            "Approved date", "Received date", "Days since requested",
            "resolutionType", "Resolution actioned", "Is_tracking_updated", "failures",
            "Store URL", "DisplayTrack", "shipment_id", "full_data", "_rma_id_text"
        ]].copy()
        
        # Columns to display (matching levis_web2.txt)
        table_display_cols = [
            "No", "RMA ID", "Order", "Current Status",
            "Tracking Number", "Tracking Status", "Requested date",
            "Approved date", "Received date", "Days since requested",
            "resolutionType", "Resolution actioned", "Is_tracking_updated", "failures"
        ]
        
        # Styling function
        def highlight_missing_tracking(row):
            full_row = display_df.loc[row.name]
            if full_row.get("Current Status") == "Approved" and not full_row.get("DisplayTrack"):
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
            key=MAIN_TABLE_KEY,
            on_select="rerun",
            selection_mode="single-row",
            column_config={
                "No": st.column_config.TextColumn("No", width="small"),
                "RMA ID": st.column_config.LinkColumn("RMA ID", display_text=r"rmaid=([^&]+)", width="medium"),
                "Order": st.column_config.TextColumn("Order", width="medium"),
                "Current Status": st.column_config.TextColumn("Current Status", width="small"),
                "Tracking Number": st.column_config.LinkColumn("Tracking Number", display_text=r"ref=(.*)", width="medium"),
                "Tracking Status": st.column_config.TextColumn("Tracking Status", width="medium"),
                "Requested date": st.column_config.TextColumn("Requested date", width="small"),
                "Approved date": st.column_config.TextColumn("Approved date", width="small"),
                "Received date": st.column_config.TextColumn("Received date", width="small"),
                "Days since requested": st.column_config.NumberColumn("Days since requested", width="small", format="%d"),
                "resolutionType": st.column_config.TextColumn("Resolution Type", width="small"),
                "Resolution actioned": st.column_config.TextColumn("Resolution Actioned", width="small"),
                "Is_tracking_updated": st.column_config.TextColumn("Tracking Updated", width="small"),
                "failures": st.column_config.TextColumn("Failures", width="medium")
            }
        )
        
        # Handle row selection
        sel_rows = []
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
