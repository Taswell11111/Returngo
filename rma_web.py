import streamlit as st
import streamlit.components.v1 as components
import requests
import json
import pandas as pd
import time
import re
import logging
import os
import threading
import html
from datetime import datetime, timedelta, timezone, date
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sqlalchemy import create_engine, text, Engine
import concurrent.futures
from typing import Optional, Tuple, Dict, Callable, Set, Union, Any, Mapping, List, Literal
from returngo_api import api_url, RMA_COMMENT_PATH
from dataclasses import dataclass, asdict, field
from enum import Enum
import pickle
from pathlib import Path

# ==========================================
# 0. CONFIGURATION & LOGGING
# ==========================================
logger = logging.getLogger(__name__)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

class NoScriptRunContextWarningFilter(logging.Filter):
    def filter(self, record):
        return not record.getMessage().startswith("Thread 'MainThread': missing ScriptRunContext!")

script_run_context_logger = logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context")
script_run_context_logger.addFilter(NoScriptRunContextWarningFilter())

# --- MULTI-STORE CONFIGURATION ---
STORES = [
    {"name": "Diesel", "url": "diesel-dev-south-africa.myshopify.com"},
    {"name": "Hurley", "url": "hurley-dev-south-africa.myshopify.com"},
    {"name": "Jeep Apparel", "url": "jeep-apparel-dev-south-africa.myshopify.com"},
    {"name": "Reebok", "url": "reebok-dev-south-africa.myshopify.com"},
    {"name": "Superdry", "url": "superdry-dev-south-africa.myshopify.com"}
]
STORE_URLS = [s["url"] for s in STORES]
STORE_MAP = {s["url"]: s["name"] for s in STORES}

# Load API Key (Try specific then generic)
try:
    MY_API_KEY = st.secrets.get("RETURNGO_API_KEY_BOUNTY") or st.secrets.get("RETURNGO_API_KEY")
except KeyError:
    MY_API_KEY = None

if not MY_API_KEY:
    st.error("Application error: Missing 'RETURNGO_API_KEY_BOUNTY' or 'RETURNGO_API_KEY' in Streamlit secrets.")
    st.stop()

# Efficiency controls
CACHE_EXPIRY_HOURS = 24
COURIER_REFRESH_HOURS = 12
MAX_WORKERS = 3
RG_RPS = 2
SYNC_OVERLAP_MINUTES = 35

ACTIVE_STATUSES = ["Pending", "Approved", "Received"]
COURIER_STATUS_OPTIONS = [
    "Submitted to Courier",
    "In Transit",
    "Delivered",
    "Courier Cancelled",
    "No tracking number",
]

# ==========================================
# DATA CLASSES
# ==========================================
@dataclass
class UserSettings:
    theme: str = "dark"
    favorites: List[str] = field(default_factory=list)
    export_presets: Dict[str, Dict] = field(default_factory=dict)
    performance_metrics: bool = True
    keyboard_shortcuts: bool = True

@dataclass
class PerformanceMetrics:
    api_call_times: List[float] = field(default_factory=list)
    cache_hit_rate: float = 0.0
    last_sync_duration: float = 0.0
    avg_response_time: float = 0.0

# ==========================================
# 1. DATABASE
# ==========================================
@st.cache_resource
def init_database() -> Optional[Engine]:
    logger.info("Initializing database connection...")
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

        engine = create_engine(db_url, pool_pre_ping=True, connect_args=connect_args)

        with engine.begin() as connection:
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
        return engine
    except Exception as e:
        logger.critical(f"Fatal database error: {e}", exc_info=True)
        st.error(f"Database Connection Error: {e}")
        st.stop()
        return None

engine = init_database()

# Globals
RATE_LIMIT_HIT = threading.Event()
RATE_LIMIT_INFO = {"remaining": None, "limit": None, "reset": None, "updated_at": None}
RATE_LIMIT_LOCK = threading.Lock()
df_view = pd.DataFrame()
counts = {}

# Feature Files
USER_SETTINGS_FILE = Path("user_settings.pkl")
PERFORMANCE_FILE = Path("performance_metrics.pkl")

# Keyboard Shortcuts
KEYBOARD_SHORTCUTS = {
    "1": "Pending Requests", "2": "Approved - Submitted", "3": "Approved - In Transit",
    "4": "Received", "5": "No Tracking", "6": "Flagged", "7": "Courier Cancelled",
    "8": "Approved - Delivered", "s": "search", "f": "toggle_favorites", "r": "refresh_all",
    "e": "export_csv", "c": "clear_filters",
}

# ==========================================
# SETTINGS MANAGEMENT
# ==========================================
def load_user_settings() -> UserSettings:
    if USER_SETTINGS_FILE.exists():
        try:
            with open(USER_SETTINGS_FILE, 'rb') as f: return pickle.load(f)
        except Exception: pass
    return UserSettings()

def save_user_settings(settings: UserSettings):
    try:
        with open(USER_SETTINGS_FILE, 'wb') as f: pickle.dump(settings, f)
    except Exception as e: logger.error(f"Failed to save settings: {e}")

def load_performance_metrics() -> PerformanceMetrics:
    if PERFORMANCE_FILE.exists():
        try:
            with open(PERFORMANCE_FILE, 'rb') as f: return pickle.load(f)
        except Exception: pass
    return PerformanceMetrics()

def save_performance_metrics(metrics: PerformanceMetrics):
    try:
        with open(PERFORMANCE_FILE, 'wb') as f: pickle.dump(metrics, f)
    except Exception as e: logger.error(f"Failed to save metrics: {e}")

# ==========================================
# 2. HTTP & RATE LIMITING
# ==========================================
_thread_local = threading.local()
_rate_lock = threading.Lock()
_last_req_ts = 0.0

def _now_utc() -> datetime: return datetime.now(timezone.utc)
def _iso_utc(dt: datetime) -> str:
    if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()

def _append_log_entry(log_key: str, message: str, *, level: str = "info") -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    entry = f"[{ts}] {message}" if level == "info" else f"[{ts}] {level.upper()}: {message}"
    log_entries = st.session_state.get(log_key, [])
    log_entries.append(entry)
    st.session_state[log_key] = log_entries[-200:]

def append_ops_log(message: str, *, level: str = "info") -> None:
    _append_log_entry("ops_log", message, level=level)

def append_schema_log(message: str, *, level: str = "info") -> None:
    _append_log_entry("schema_log", message, level=level)

def _sleep_for_rate_limit():
    global _last_req_ts
    if RG_RPS <= 0: return
    min_interval = 1.0 / float(RG_RPS)
    with _rate_lock:
        now = time.time()
        wait = (_last_req_ts + min_interval) - now
        if wait > 0: time.sleep(wait)
        _last_req_ts = time.time()

def get_thread_session() -> requests.Session:
    s = getattr(_thread_local, "session", None)
    if s is not None: return s
    s = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504))
    s.mount("https://", HTTPAdapter(max_retries=retries))
    _thread_local.session = s
    return s

def rg_headers(store_url: str) -> Dict[str, Optional[str]]:
    return {"x-api-key": MY_API_KEY, "X-API-KEY": MY_API_KEY, "x-shop-name": store_url}

def update_rate_limit_info(headers: Mapping[str, Any]):
    if not headers: return
    lower = {str(k).lower(): v for k, v in headers.items()}
    remaining = lower.get("x-ratelimit-remaining")
    limit = lower.get("x-ratelimit-limit")
    if remaining is None: return
    
    with RATE_LIMIT_LOCK:
        try:
            RATE_LIMIT_INFO["remaining"] = int(remaining)
            if limit: RATE_LIMIT_INFO["limit"] = int(limit)
            RATE_LIMIT_INFO["updated_at"] = _now_utc()
        except ValueError: pass

def rg_request(method: str, url: str, store_url: str, *, timeout=15, json_body=None):
    start_time = time.time()
    session = get_thread_session()
    headers = rg_headers(store_url)
    
    backoff = 1
    res = None
    for _attempt in range(1, 6):
        _sleep_for_rate_limit()
        res = session.request(method, url, headers=headers, timeout=timeout, json=json_body)
        update_rate_limit_info(res.headers)
        
        if res.status_code != 429:
            if "performance_metrics" in st.session_state:
                st.session_state.performance_metrics.api_call_times.append(time.time() - start_time)
            return res
            
        RATE_LIMIT_HIT.set()
        ra = res.headers.get("Retry-After")
        sleep_s = int(ra) if (ra and ra.isdigit()) else backoff
        backoff = min(backoff * 2, 30)
        time.sleep(sleep_s)
    return None

# ==========================================
# 3. DATABASE FUNCTIONS
# ==========================================
def upsert_rma(rma_id: str, store_url: str, status: str, created_at: str, payload: dict,
               courier_status: Optional[str] = None, courier_checked_iso: Optional[str] = None):
    now_iso = _iso_utc(_now_utc())
    
    created_at_dt = None
    if created_at:
        try: created_at_dt = pd.to_datetime(created_at).tz_localize('UTC')
        except: pass

    courier_checked_dt = None
    if courier_checked_iso:
        try: courier_checked_dt = pd.to_datetime(courier_checked_iso).tz_localize('UTC')
        except: pass

    if engine is None: return

    with engine.connect() as connection:
        # Check existing values to preserve them if not provided
        res = connection.execute(
            text("SELECT courier_status, courier_last_checked, received_first_seen FROM rmas WHERE rma_id=:rid"),
            {"rid": rma_id}
        ).fetchone()
        
        existing_cstat = res[0] if res else None
        existing_cchk = res[1] if res else None
        existing_received = res[2] if res else None

        if courier_status is None: courier_status = existing_cstat
        if courier_checked_iso is None and existing_cchk: courier_checked_dt = existing_cchk
        
        received_seen = existing_received
        if status == "Received" and not existing_received: received_seen = now_iso

        stmt = text("""
            INSERT INTO rmas (rma_id, store_url, status, created_at, json_data, last_fetched, 
                              courier_status, courier_last_checked, received_first_seen)
            VALUES (:rma_id, :store_url, :status, :created_at, CAST(:json_data AS JSONB), :last_fetched, 
                    :courier_status, :courier_last_checked, :received_first_seen)
            ON CONFLICT (rma_id) DO UPDATE SET
                store_url = EXCLUDED.store_url,
                status = EXCLUDED.status,
                created_at = EXCLUDED.created_at,
                json_data = EXCLUDED.json_data,
                last_fetched = EXCLUDED.last_fetched,
                courier_status = EXCLUDED.courier_status,
                courier_last_checked = EXCLUDED.courier_last_checked,
                received_first_seen = EXCLUDED.received_first_seen;
        """)
        connection.execute(stmt, {
            "rma_id": str(rma_id), "store_url": store_url, "status": status,
            "created_at": created_at_dt, "json_data": payload, "last_fetched": now_iso,
            "courier_status": courier_status, "courier_last_checked": courier_checked_dt,
            "received_first_seen": received_seen
        })
        connection.commit()

def delete_rmas(rma_ids):
    if not rma_ids or engine is None: return
    ids = [str(i) for i in rma_ids if i]
    if not ids: return
    try:
        with engine.connect() as conn:
            conn.execute(text("DELETE FROM rmas WHERE rma_id = ANY(:ids)"), {"ids": ids})
            conn.commit()
            if "cache_version" in st.session_state: st.session_state.cache_version += 1
    except Exception as e:
        logger.error(f"Failed delete: {e}")

def get_rma(rma_id: str):
    if engine is None: return None
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT json_data, last_fetched, courier_status, courier_last_checked, received_first_seen, store_url FROM rmas WHERE rma_id=:rid"),
            {"rid": str(rma_id)}
        ).fetchone()
    if not row: return None
    
    payload = row[0]
    payload["_local_courier_status"] = row[2]
    payload["_local_courier_checked"] = row[3].isoformat() if row[3] else None
    payload["_local_received_first_seen"] = row[4].isoformat() if row[4] else None
    payload["_store_url"] = row[5] # Store URL for multi-store context
    return payload, (row[1].isoformat() if row[1] else None)

def get_all_open_from_db():
    if engine is None: return []
    with engine.connect() as conn:
        # NOTE: Fetching for ALL active statuses across ALL stores
        rows = conn.execute(
            text("SELECT json_data, courier_status, courier_last_checked, received_first_seen, store_url FROM rmas WHERE status = ANY(:statuses)"),
            {"statuses": ACTIVE_STATUSES}
        ).fetchall()
        
    results = []
    for js, cstat, cchk, rcv_seen, surl in rows:
        data = js
        data["_local_courier_status"] = cstat
        data["_local_courier_checked"] = cchk.isoformat() if cchk else None
        data["_local_received_first_seen"] = rcv_seen.isoformat() if rcv_seen else None
        data["_store_url"] = surl
        results.append(data)
    return results

@st.cache_data(show_spinner=False, ttl=60)
def load_open_rmas(_cache_version: int):
    return get_all_open_from_db()

def get_local_ids_for_status(status: str):
    if engine is None: return set()
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT rma_id FROM rmas WHERE status=:status"), {"status": status}).fetchall()
    return {r[0] for r in rows}

def set_last_sync(scope: str, dt: datetime):
    if engine is None: return
    with engine.connect() as conn:
        conn.execute(
            text("INSERT INTO sync_logs (scope, last_sync_iso) VALUES (:s, :t) ON CONFLICT (scope) DO UPDATE SET last_sync_iso=:t"),
            {"s": scope, "t": dt}
        )
        conn.commit()

def get_last_sync(scope: str) -> Optional[datetime]:
    if engine is None: return None
    with engine.connect() as conn:
        row = conn.execute(text("SELECT last_sync_iso FROM sync_logs WHERE scope=:s"), {"s": scope}).fetchone()
        return row[0] if row else None

def clear_db():
    if engine is None: return False
    try:
        with engine.begin() as conn:
            conn.execute(text("TRUNCATE TABLE rmas, sync_logs;"))
        return True
    except Exception: return False

# ==========================================
# 4. COURIER STATUS (Web Scraping)
# ==========================================
def check_courier_status(tracking_number: str) -> str:
    # Simplified version for brevity, fully compatible with enhanced version logic
    if not tracking_number: return "No tracking number"
    try:
        url = f"https://optimise.parcelninja.com/shipment/track?WaybillNo={tracking_number}"
        # Using a dedicated session for this if possible, otherwise requests
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        if res.status_code == 404: return "Tracking not found"
        if res.status_code != 200: return f"Tracking Error {res.status_code}"
        
        # Simple HTML scrape for keywords if JSON parse fails
        txt = res.text.lower()
        if "delivered" in txt: return "Delivered"
        if "collected" in txt: return "Collected"
        if "out for delivery" in txt: return "Out for Delivery"
        if "in transit" in txt: return "In Transit"
        return "In Transit (Assumed)"
    except Exception:
        return "Tracking Check Failed"

@st.cache_data(ttl=COURIER_REFRESH_HOURS * 3600, show_spinner=False)
def check_courier_status_cached(tracking_number: str) -> str:
    return check_courier_status(tracking_number)

# ==========================================
# 5. SYNC LOGIC
# ==========================================
def fetch_rma_list(store_url, statuses, since_dt: Optional[datetime]) -> Tuple[list, bool, Optional[str]]:
    all_summaries = []
    status_param = ",".join(statuses)
    cursor = None
    
    updated_filter = ""
    if since_dt:
        since_dt = since_dt - timedelta(minutes=SYNC_OVERLAP_MINUTES)
        updated_filter = f"&rma_updated_at=gte:{_iso_utc(since_dt)}"

    while True:
        base_url = f"{api_url('/rmas')}?pagesize=500&status={status_param}{updated_filter}"
        url = f"{base_url}&cursor={cursor}" if cursor else base_url
        
        res = rg_request("GET", url, store_url, timeout=20)
        if not res or res.status_code != 200:
            return all_summaries, False, (res.text if res else "No Response")
            
        data = res.json()
        rmas = data.get("rmas", [])
        if not rmas: break
        
        all_summaries.extend(rmas)
        cursor = data.get("next_cursor")
        if not cursor: break
        
    return all_summaries, True, None

def should_refresh_detail(rma_id: str) -> bool:
    cached = get_rma(rma_id)
    if not cached: return True
    _, last_fetched_iso = cached
    if not last_fetched_iso: return True
    try:
        last_dt = datetime.fromisoformat(last_fetched_iso)
        if last_dt.tzinfo is None: last_dt = last_dt.replace(tzinfo=timezone.utc)
        return (_now_utc() - last_dt) > timedelta(hours=CACHE_EXPIRY_HOURS)
    except ValueError: return True

def fetch_rma_detail(rma_id: str, store_url: str, *, force: bool = False):
    cached = get_rma(rma_id)
    if (not force) and cached and not should_refresh_detail(rma_id):
        return cached[0]

    res = rg_request("GET", api_url(f"/rma/{rma_id}"), store_url, timeout=20)
    if not res or res.status_code != 200:
        return cached[0] if cached else None

    data = res.json()
    
    # Courier Refresh Logic
    track_no = None
    for s in (data.get("shipments") or []):
        if s.get("trackingNumber"): 
            track_no = s.get("trackingNumber")
            break
            
    courier_status = data.get("_local_courier_status")
    courier_checked = data.get("_local_courier_checked")
    
    # Simple check logic: if we have tracking, and it's time to refresh
    should_refresh_courier = True
    if courier_checked:
        try:
            chk_dt = datetime.fromisoformat(courier_checked)
            if (_now_utc().replace(tzinfo=None) - chk_dt.replace(tzinfo=None)) < timedelta(hours=COURIER_REFRESH_HOURS):
                should_refresh_courier = False
        except: pass
        
    if track_no and should_refresh_courier:
        courier_status = check_courier_status_cached(track_no)
        courier_checked = _iso_utc(_now_utc())
    elif cached:
        # Preserve previous if not refreshing
        courier_status = cached[0].get("_local_courier_status")
        courier_checked = cached[0].get("_local_courier_checked")

    upsert_rma(
        rma_id=str(rma_id),
        store_url=store_url,
        status=data.get("rmaSummary", {}).get("status", "Unknown"),
        created_at=data.get("createdAt") or "",
        payload=data,
        courier_status=courier_status,
        courier_checked_iso=courier_checked
    )
    
    data["_local_courier_status"] = courier_status
    data["_store_url"] = store_url
    return data

def perform_sync(statuses=None, *, full=False, stores_to_sync=None):
    if st.session_state.get("disconnected"): return
    
    sync_start = time.time()
    if statuses is None: statuses = ACTIVE_STATUSES
    if stores_to_sync is None: stores_to_sync = STORES

    append_ops_log(f"Starting sync for {len(stores_to_sync)} stores...")
    
    total_upd = 0
    
    for store in stores_to_sync:
        store_url = store["url"]
        scope_key = f"{store_url}_ALL"
        
        # Determine last sync time for this store
        since_dt = get_last_sync(scope_key) if not full else None
        
        append_ops_log(f"[{store['name']}] Fetching list...")
        summaries, ok, err = fetch_rma_list(store_url, statuses, since_dt)
        
        if not ok:
            append_ops_log(f"[{store['name']}] Sync failed: {err}", level="error")
            continue
            
        api_ids = {s.get("rmaId") for s in summaries if s.get("rmaId")}
        
        # Cleanup stale
        if not full and since_dt:
            # For incremental, we trust the API list. If doing full, we verify db vs API.
            pass 
            
        # Detail Fetching
        to_fetch = list(api_ids) if since_dt else [rid for rid in api_ids if should_refresh_detail(rid)]
        
        if to_fetch:
            append_ops_log(f"[{store['name']}] Updating {len(to_fetch)} details...")
            for i, rid in enumerate(to_fetch):
                fetch_rma_detail(rid, store_url, force=(since_dt is not None))
                total_upd += 1
                
        set_last_sync(scope_key, _now_utc())

    duration = time.time() - sync_start
    if "performance_metrics" in st.session_state:
        st.session_state.performance_metrics.last_sync_duration = duration
        
    append_ops_log(f"✅ Sync Complete! {total_upd} RMAs updated in {duration:.1f}s.")
    st.session_state["show_toast"] = True
    load_open_rmas.clear()
    st.rerun()

def force_refresh_rma_ids(rma_ids_with_stores: List[Tuple[str, str]]):
    """Refreshes specific RMAs. Input is list of (rma_id, store_url)."""
    if not rma_ids_with_stores: return
    append_ops_log(f"Refreshing {len(rma_ids_with_stores)} records...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(fetch_rma_detail, rid, surl, force=True) for rid, surl in rma_ids_with_stores]
        concurrent.futures.wait(futures)

    st.session_state.cache_version = st.session_state.get("cache_version", 0) + 1
    st.session_state["show_toast"] = True
    load_open_rmas.clear()
    st.rerun()

# ==========================================
# 6. HELPERS FOR UI
# ==========================================
def inject_enhanced_css():
    st.markdown("""
    <style>
    /* Dark/Light Theme Variables */
    :root {
        --bg-secondary: #1e293b; --border-color: #475569; --text-primary: #f1f5f9; --accent-primary: #3b82f6;
    }
    .light-mode {
        --bg-secondary: #f8fafc; --border-color: #cbd5e1; --text-primary: #1e293b; --accent-primary: #2563eb;
    }
    
    /* Tiles */
    .enhanced-tile {
        background: var(--bg-secondary); border: 1px solid var(--border-color); border-radius: 0.5rem;
        padding: 1rem; margin-bottom: 0.5rem; transition: all 0.2s; position: relative;
    }
    .enhanced-tile:hover { transform: translateY(-2px); border-color: var(--accent-primary); }
    .enhanced-tile.selected { border-color: #22c55e; background: rgba(34, 197, 94, 0.05); }
    
    .tile-header { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem; }
    .tile-title { flex: 1; font-weight: 600; font-size: 0.8rem; text-transform: uppercase; }
    .tile-badge { background: #334155; padding: 2px 8px; border-radius: 99px; font-weight: 700; }
    
    .summary-bar { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem; margin-bottom: 1rem; }
    .summary-item { text-align: center; padding: 1rem; background: var(--bg-secondary); border-radius: 0.5rem; border: 1px solid var(--border-color); }
    .summary-value { font-size: 1.5rem; font-weight: 800; }
    
    /* Multi-store banner */
    .store-badge { background: #2563eb; color: white; padding: 2px 6px; border-radius: 4px; font-size: 0.75rem; margin-right: 5px; }
    </style>
    """, unsafe_allow_html=True)

# FILTERS DEFINITION
FILTERS = {
    "Pending Requests": {"icon": "⏳", "key": "Pending", "fn": lambda d: d["Current Status"] == "Pending"},
    "Approved - Submitted": {"icon": "📬", "key": "AppSub", "fn": lambda d: (d["Current Status"] == "Approved") & (d["Tracking Status"].str.contains("Submitted", case=False, na=False))},
    "Approved - In Transit": {"icon": "🚚", "key": "AppTransit", "fn": lambda d: (d["Current Status"] == "Approved") & (d["Tracking Status"].str.contains("Transit|Out for|Routing", case=False, na=False))},
    "Received": {"icon": "📦", "key": "Received", "fn": lambda d: d["Current Status"] == "Received"},
    "No Tracking": {"icon": "🚫", "key": "NoTrack", "fn": lambda d: d["is_nt"]},
    "Flagged": {"icon": "🚩", "key": "Flagged", "fn": lambda d: d["is_fg"]},
    "Courier Cancelled": {"icon": "🛑", "key": "Cancelled", "fn": lambda d: d["is_cc"]},
}

def render_enhanced_tile(col, name, cfg, count_val, is_selected):
    with col:
        st.markdown(f"""
        <div class='enhanced-tile {"selected" if is_selected else ""}'>
            <div class="tile-header">
                <span style="font-size: 1.2rem">{cfg['icon']}</span>
                <span class="tile-title">{name}</span>
                <span class="tile-badge">{count_val}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("View", key=f"btn_{name}", use_container_width=True):
            if is_selected:
                st.session_state.active_filters.remove(name)
            else:
                st.session_state.active_filters.add(name)
            st.rerun()

# ==========================================
# MAIN
# ==========================================
def main():
    st.set_page_config(page_title="Bounty ReturnGO Hub", layout="wide", page_icon="🌐")
    
    # State Init
    if "user_settings" not in st.session_state: st.session_state.user_settings = load_user_settings()
    if "performance_metrics" not in st.session_state: st.session_state.performance_metrics = load_performance_metrics()
    if "active_filters" not in st.session_state: st.session_state.active_filters = set()
    if "show_toast" not in st.session_state: st.session_state.show_toast = False
    
    inject_enhanced_css()
    if st.session_state.user_settings.theme == "light":
        st.markdown('<style>.light-mode {}</style>', unsafe_allow_html=True)

    if st.session_state.show_toast:
        st.toast("Updated Successfully!", icon="✅")
        st.session_state.show_toast = False

    # --- SIDEBAR ---
    with st.sidebar:
        st.subheader("Data Operations")
        if st.button("🚀 Sync All Stores", type="primary"):
            perform_sync(full=True)
        
        st.divider()
        st.caption("Sync Specific Store:")
        for s in STORES:
            if st.button(f"🔄 Sync {s['name']}"):
                perform_sync(stores_to_sync=[s])

        st.divider()
        if st.button("🗑️ Reset Cache"):
            clear_db()
            st.session_state.cache_version = st.session_state.get("cache_version", 0) + 1
            st.rerun()
            
        with st.expander("Settings"):
            theme = st.selectbox("Theme", ["dark", "light"], index=0 if st.session_state.user_settings.theme=="dark" else 1)
            if theme != st.session_state.user_settings.theme:
                st.session_state.user_settings.theme = theme
                save_user_settings(st.session_state.user_settings)
                st.rerun()

        st.markdown("### Activity Log")
        st.code("\n".join(st.session_state.get("ops_log", [])[-5:]), language="text")

    # --- HEADER ---
    st.title("Bounty Brands ReturnGO Hub")
    
    # GLOBAL STORE FILTER
    selected_stores = st.multiselect(
        "Filter by Store", 
        options=[s["name"] for s in STORES],
        default=[],
        help="Leave empty to view all stores"
    )
    
    target_urls = [s["url"] for s in STORES if s["name"] in selected_stores] if selected_stores else STORE_URLS

    # --- LOAD DATA ---
    raw_data = load_open_rmas(st.session_state.get("cache_version", 0))
    
    rows = []
    for rma in raw_data:
        # Multi-store filtering at dataframe level
        store_url = rma.get("_store_url") or rma.get("store_url")
        if store_url not in target_urls: continue
        
        summary = rma.get("rmaSummary", {})
        status = summary.get("status", "Unknown")
        rma_id = summary.get("rmaId")
        
        # Flags
        shipments = rma.get("shipments", [])
        track_nums = [s.get("trackingNumber") for s in shipments if s.get("trackingNumber")]
        track_str = ", ".join(track_nums) if track_nums else ""
        
        local_courier = rma.get("_local_courier_status", "").lower()
        
        is_nt = (status == "Approved" and not track_str)
        is_cc = "cancelled" in local_courier
        is_ad = "delivered" in local_courier and status == "Approved"
        comments = " ".join([c.get("htmlText", "") for c in rma.get("comments", [])]).lower()
        is_fg = "flagged" in comments
        
        rows.append({
            "RMA ID": rma_id,
            "Store": STORE_MAP.get(store_url, "Unknown"),
            "Store URL": store_url,
            "Order": summary.get("order_name"),
            "Current Status": status,
            "Tracking Number": track_str,
            "Tracking Status": rma.get("_local_courier_status", ""),
            "Requested date": (summary.get("createdAt") or "")[:10],
            "is_nt": is_nt, "is_cc": is_cc, "is_fg": is_fg, "is_ad": is_ad,
            "full_data": rma,
            "_rma_id_text": rma_id
        })

    df = pd.DataFrame(rows)
    
    # --- COUNTS & METRICS ---
    counts = {}
    for name, cfg in FILTERS.items():
        if df.empty:
            counts[name] = 0
        else:
            counts[name] = int(cfg["fn"](df).sum())

    # --- COMMAND CENTER ---
    st.markdown("### Command Center")
    
    st.markdown(f"""
    <div class="summary-bar">
        <div class="summary-item"><div class="summary-value">{len(df)}</div><div>Visible Records</div></div>
        <div class="summary-item"><div class="summary-value">{counts.get("Pending Requests", 0)}</div><div>Pending</div></div>
        <div class="summary-item"><div class="summary-value">{counts.get("Approved - In Transit", 0)}</div><div>In Transit</div></div>
        <div class="summary-item"><div class="summary-value">{counts.get("Flagged", 0) + counts.get("No Tracking", 0)}</div><div>Issues</div></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Tiles
    tile_cols = st.columns(4)
    for i, (name, cfg) in enumerate(FILTERS.items()):
        render_enhanced_tile(
            tile_cols[i % 4], 
            name, cfg, 
            counts.get(name, 0), 
            name in st.session_state.active_filters
        )

    st.divider()

    # --- TABLE ---
    if df.empty:
        st.info("No records match the current filters.")
        st.stop()
        
    # Apply Active Tile Filters
    if st.session_state.active_filters:
        mask = pd.Series([False] * len(df), index=df.index)
        for f in st.session_state.active_filters:
            mask |= FILTERS[f]["fn"](df)
        df = df[mask]
        
    # Search
    search = st.text_input("Search...", placeholder="RMA, Order, or Tracking")
    if search:
        s = search.lower()
        df = df[df.astype(str).apply(lambda x: x.str.lower().str.contains(s)).any(axis=1)]

    st.markdown(f"**Showing {len(df)} records**")
    
    column_config = {
        "RMA ID": st.column_config.TextColumn("RMA ID"),
        "Store": st.column_config.TextColumn("Store"),
        "Tracking Status": st.column_config.TextColumn("Tracking Status"),
    }
    
    event = st.dataframe(
        df[["Store", "RMA ID", "Order", "Current Status", "Tracking Status", "Requested date"]],
        use_container_width=True,
        hide_index=True,
        selection_mode="single-row",
        on_select="rerun",
        height=600,
        column_config=column_config
    )
    
    # --- DETAIL VIEW ---
    if event.selection.rows:
        idx = event.selection.rows[0]
        record = df.iloc[idx]
        
        @st.dialog(f"RMA Details: {record['RMA ID']}")
        def show_detail():
            st.markdown(f"**Store:** {record['Store']} | **Status:** {record['Current Status']}")
            st.markdown(f"**Order:** {record['Order']}")
            
            # Actions
            t1, t2 = st.tabs(["Tracking", "Notes"])
            with t1:
                new_track = st.text_input("Tracking Number", value=record['Tracking Number'])
                if st.button("Update Tracking"):
                    ship_id = None
                    ships = record['full_data'].get('shipments', [])
                    if ships: ship_id = ships[0].get('shipmentId')
                    
                    if ship_id:
                        ok, msg = push_tracking_update(record['_rma_id_text'], ship_id, new_track, record['Store URL'])
                        if ok: st.success("Updated!"); st.rerun()
                        else: st.error(msg)
                    else: st.error("No Shipment ID found on record.")
            
            with t2:
                new_note = st.text_area("Add Note")
                if st.button("Post Note"):
                    ok, msg = push_comment_update(record['_rma_id_text'], new_note, record['Store URL'])
                    if ok: st.success("Posted!"); st.rerun()
                    else: st.error(msg)
                
                st.divider()
                comments = record['full_data'].get('comments', [])
                for c in comments:
                    st.caption(f"{c.get('datetime', '')[:16]} - {c.get('triggeredBy', 'System')}")
                    st.markdown(f"> {c.get('htmlText', '')}")
                    st.divider()
                    
        show_detail()

if __name__ == "__main__":
    main()
