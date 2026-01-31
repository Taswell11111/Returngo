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
from dataclasses import dataclass, asdict
from enum import Enum
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

# Suppress the "missing ScriptRunContext" warning
class NoScriptRunContextWarningFilter(logging.Filter):
    def filter(self, record):
        return not record.getMessage().startswith("Thread 'MainThread': missing ScriptRunContext!")

script_run_context_logger = logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context")
script_run_context_logger.addFilter(NoScriptRunContextWarningFilter())

# ==========================================
# NEW: Data classes for enhanced features
# ==========================================
@dataclass
class UserSettings:
    theme: str = "dark"  # dark, light, auto
    favorites: List[str] = None
    export_presets: Dict[str, Dict] = None
    performance_metrics: bool = True
    keyboard_shortcuts: bool = True
    
    def __post_init__(self):
        if self.favorites is None:
            self.favorites = []
        if self.export_presets is None:
            self.export_presets = {}

@dataclass
class PerformanceMetrics:
    api_call_times: List[float] = None
    cache_hit_rate: float = 0.0
    last_sync_duration: float = 0.0
    avg_response_time: float = 0.0
    
    def __post_init__(self):
        if self.api_call_times is None:
            self.api_call_times = []

class Theme(Enum):
    DARK = "dark"
    LIGHT = "light"
    AUTO = "auto"

# ==========================================
# 1a. CONFIGURATION
# ==========================================
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

        engine = create_engine(
            db_url,
            pool_pre_ping=True,
            connect_args=connect_args,
        )

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
        logger.info("Database tables verified and connection is ready.")
        return engine

    except KeyError as e:
        logger.critical(f"Database configuration error: Missing key {e} in [connections.postgresql] secrets.")
        st.error(f"Application error: Database configuration is missing required key: {e}. Please check your secrets.toml file.")
        st.stop()
        return None
    except Exception as e:
        logger.critical(f"Fatal error during database initialization: {e}", exc_info=True)
        st.error(f"Application error: Could not connect to the database. Details: {e}")
        st.stop()
        return None
        
# Initialize the database connection engine at the module level
engine = init_database()

# Load API secrets
try:
    MY_API_KEY = st.secrets["RETURNGO_API_KEY"]
    PARCEL_NINJA_TOKEN = st.secrets.get("PARCEL_NINJA_TOKEN")
except KeyError:
    st.error("Application error: Missing 'RETURNGO_API_KEY' in Streamlit secrets.")
    st.stop()
    MY_API_KEY = None
    PARCEL_NINJA_TOKEN = None
STORE_URL = "levis-sa.myshopify.com"

# Efficiency controls
CACHE_EXPIRY_HOURS = 24
COURIER_REFRESH_HOURS = 12
MAX_WORKERS = 3
RG_RPS = 2
SYNC_OVERLAP_MINUTES = 35

ACTIVE_STATUSES = ["Pending", "Approved", "Received"]
PRIMARY_STATUS_TILES = {"Pending", "Approved", "Received", "No Tracking", "Flagged"}
COURIER_STATUS_OPTIONS = [
    "Submitted to Courier",
    "In Transit",
    "Delivered",
    "Courier Cancelled",
    "No tracking number",
]

RATE_LIMIT_HIT = threading.Event()
RATE_LIMIT_INFO: Dict[str, Union[int, str, datetime, None]] = {"remaining": None, "limit": None, "reset": None, "updated_at": None}
RATE_LIMIT_LOCK = threading.Lock()
PARCEL_NINJA_TOKEN: Optional[str] = None
df_view = pd.DataFrame()
counts: Dict[str, int] = {}

# ==========================================
# NEW: Enhanced feature initializations
# ==========================================
USER_SETTINGS_FILE = Path("user_settings.pkl")
PERFORMANCE_FILE = Path("performance_metrics.pkl")
KEYBOARD_SHORTCUTS = {
    "1": "Pending Requests",
    "2": "Approved - Submitted",
    "3": "Approved - In Transit",
    "4": "Received",
    "5": "No Tracking",
    "6": "Flagged",
    "7": "Courier Cancelled",
    "8": "Approved - Delivered",
    "9": "Resolution Actioned",
    "0": "No Resolution Actioned",
    "s": "search",
    "f": "toggle_favorites",
    "r": "refresh_all",
    "e": "export_csv",
    "c": "clear_filters",
}

def load_user_settings() -> UserSettings:
    """Load user settings from file or create default."""
    if USER_SETTINGS_FILE.exists():
        try:
            with open(USER_SETTINGS_FILE, 'rb') as f:
                return pickle.load(f)
        except Exception:
            pass
    return UserSettings()

def save_user_settings(settings: UserSettings):
    """Save user settings to file."""
    try:
        with open(USER_SETTINGS_FILE, 'wb') as f:
            pickle.dump(settings, f)
    except Exception as e:
        logger.error(f"Failed to save user settings: {e}")

def load_performance_metrics() -> PerformanceMetrics:
    """Load performance metrics from file."""
    if PERFORMANCE_FILE.exists():
        try:
            with open(PERFORMANCE_FILE, 'rb') as f:
                return pickle.load(f)
        except Exception:
            pass
    return PerformanceMetrics()

def save_performance_metrics(metrics: PerformanceMetrics):
    """Save performance metrics to file."""
    try:
        with open(PERFORMANCE_FILE, 'wb') as f:
            pickle.dump(metrics, f)
    except Exception as e:
        logger.error(f"Failed to save performance metrics: {e}")

# ==========================================
# 2. HTTP SESSIONS + RATE LIMITING
# ==========================================
_thread_local = threading.local()
_rate_lock = threading.Lock()
_last_req_ts = 0.0


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _iso_utc(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _append_log_entry(log_key: str, message: str, *, level: str = "info", limit: int = 200) -> None:
    if not message:
        return
    ts = datetime.now().strftime("%H:%M:%S")
    entry = f"[{ts}] {message}"
    if level and level != "info":
        entry = f"[{ts}] {level.upper()}: {message}"
    log_entries: List[str] = st.session_state.get(log_key, [])
    log_entries.append(entry)
    st.session_state[log_key] = log_entries[-limit:]


def append_ops_log(message: str, *, level: str = "info") -> None:
    _append_log_entry("ops_log", message, level=level)


def append_schema_log(message: str, *, level: str = "info") -> None:
    _append_log_entry("schema_log", message, level=level)


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


def get_parcel_session() -> requests.Session:
    s = getattr(_thread_local, "parcel_session", None)
    if s is not None:
        return s

    s = requests.Session()
    retries = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=0.7,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    _thread_local.parcel_session = s
    return s


def rg_headers() -> Dict[str, Optional[str]]:
    return {"x-api-key": MY_API_KEY, "X-API-KEY": MY_API_KEY, "x-shop-name": STORE_URL}


def update_rate_limit_info(headers: Mapping[str, Optional[str]]):
    if not headers:
        return
    lower = {str(k).lower(): v for k, v in headers.items()}
    remaining = (
        lower.get("x-ratelimit-remaining")
        or lower.get("x-rate-limit-remaining")
        or lower.get("ratelimit-remaining")
    )
    limit = (
        lower.get("x-ratelimit-limit")
        or lower.get("x-rate-limit-limit")
        or lower.get("ratelimit-limit")
    )
    reset = (
        lower.get("x-ratelimit-reset")
        or lower.get("x-rate-limit-reset")
        or lower.get("ratelimit-reset")
    )

    if remaining is None and limit is None and reset is None:
        return

    def to_int(val):
        if val is None:
            return None
        sval = str(val).strip()
        return int(sval) if sval.isdigit() else sval

    with RATE_LIMIT_LOCK:
        RATE_LIMIT_INFO["remaining"] = to_int(remaining)
        RATE_LIMIT_INFO["limit"] = to_int(limit)
        RATE_LIMIT_INFO["reset"] = to_int(reset)
        RATE_LIMIT_INFO["updated_at"] = _now_utc()


def rg_request(method: str, url: str, *, headers=None, timeout=15, json_body=None):
    start_time = time.time()
    logger.info(f"rg_request: Making {method} request to {url}")
    session: requests.Session = get_thread_session()
    headers = headers or rg_headers()

    backoff = 1
    res = None
    for _attempt in range(1, 6):
        _sleep_for_rate_limit()
        logger.info(f"rg_request: Attempt {_attempt}...")
        res = session.request(method, url, headers=headers, timeout=timeout, json=json_body)
        logger.info(f"rg_request: Request completed with status {res.status_code if res else 'None'}")
        update_rate_limit_info(res.headers)
        if res.status_code != 429:
            end_time = time.time()
            # Track performance metrics
            if "performance_metrics" in st.session_state:
                st.session_state.performance_metrics.api_call_times.append(end_time - start_time)
                if len(st.session_state.performance_metrics.api_call_times) > 100:
                    st.session_state.performance_metrics.api_call_times = st.session_state.performance_metrics.api_call_times[-100:]
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


# ==========================================
# 3. DATABASE
# ==========================================
def upsert_rma(rma_id: str, status: str, created_at: str, payload: dict,
               courier_status: Optional[str] = None, courier_checked_iso: Optional[str] = None):
    now_iso = _iso_utc(_now_utc())
    
    created_at_dt = None
    if created_at:
        try:
            created_at_dt = pd.to_datetime(created_at).tz_localize('UTC')
        except Exception:
            created_at_dt = None

    courier_checked_dt = None
    if courier_checked_iso:
        try:
            courier_checked_dt = pd.to_datetime(courier_checked_iso).tz_localize('UTC')
        except Exception:
            courier_checked_dt = None

    if engine is None:
        logger.error("Database engine not initialized, cannot upsert RMA %s", rma_id)
        return
        
    with engine.connect() as connection:
        select_query = text("SELECT courier_status, courier_last_checked, received_first_seen FROM rmas WHERE rma_id=:rma_id")
        result = connection.execute(select_query, {"rma_id": rma_id}).fetchone()

        existing_cstat = result[0] if result else None
        existing_cchk = result[1] if result else None
        existing_received = result[2] if result else None

        if courier_status is None:
            courier_status = existing_cstat
        if courier_checked_iso is None and existing_cchk:
            courier_checked_dt = existing_cchk
        
        received_seen = existing_received
        if status == "Received" and not existing_received:
            received_seen = now_iso

        insert_query = text("""
        INSERT INTO rmas (rma_id, store_url, status, created_at, json_data, last_fetched, courier_status, courier_last_checked, received_first_seen)
        VALUES (:rma_id, :store_url, :status, :created_at, CAST(:json_data AS JSONB), :last_fetched, :courier_status, :courier_last_checked, :received_first_seen)
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
        
        connection.execute(insert_query, {
            "rma_id": str(rma_id),
            "store_url": STORE_URL,
            "status": status,
            "created_at": created_at_dt,
            "json_data": payload,
            "last_fetched": now_iso,
            "courier_status": courier_status,
            "courier_last_checked": courier_checked_dt,
            "received_first_seen": received_seen
        })
        connection.commit()


def delete_rmas(rma_ids):
    if not rma_ids:
        return
    if isinstance(rma_ids, (str, bytes)):
        rma_ids = [rma_ids]

    seen: Set[str] = set()
    normalized_ids = []
    for i in (rma_ids or []):
        if i is None:
            continue
        s = str(i)
        if s == "":
            continue
        if s in seen:
            continue
        seen.add(s)
        normalized_ids.append(s)
    if not normalized_ids:
        return
    
    if engine is None:
        return
        
    try:
        with engine.connect() as connection:
            delete_query = text("DELETE FROM rmas WHERE rma_id = ANY(:rma_ids) AND store_url=:store_url")
            connection.execute(delete_query, {"rma_ids": normalized_ids, "store_url": STORE_URL})
            connection.commit()
            if "cache_version" in st.session_state:
                st.session_state.cache_version += 1
            logger.info("Deleted %s RMAs from cache", len(normalized_ids))
    except Exception as e:
        logger.error("Failed to delete RMAs: %s", e)
        st.error(f"Failed to delete RMAs: {e}")
        raise


def get_rma(rma_id: str):
    if engine is None:
        return None

    with engine.connect() as connection:
        query = text("""
            SELECT json_data, last_fetched, courier_status, courier_last_checked, received_first_seen 
            FROM rmas WHERE rma_id=:rma_id
        """)
        row = connection.execute(query, {"rma_id": str(rma_id)}).fetchone()

    if not row:
        return None

    payload = row[0]
    payload["_local_courier_status"] = row[2]
    payload["_local_courier_checked"] = row[3].isoformat() if row[3] else None
    payload["_local_received_first_seen"] = row[4].isoformat() if row[4] else None
    return payload, row[1].isoformat() if row[1] else None


def get_all_open_from_db():
    if engine is None:
        return []

    with engine.connect() as connection:
        query = text(f"""
            SELECT json_data, courier_status, courier_last_checked, received_first_seen
            FROM rmas
            WHERE store_url=:store_url AND status = ANY(:statuses)
        """)
        rows = connection.execute(query, {"store_url": STORE_URL, "statuses": ACTIVE_STATUSES}).fetchall()

    results = []
    for js, cstat, cchk, rcv_seen in rows:
        data = js
        data["_local_courier_status"] = cstat
        data["_local_courier_checked"] = cchk.isoformat() if cchk else None
        data["_local_received_first_seen"] = rcv_seen.isoformat() if rcv_seen else None
        results.append(data)
    return results


@st.cache_data(show_spinner=False, ttl=60)
def load_open_rmas(_cache_version: int):
    return get_all_open_from_db()


def get_local_ids_for_status(status: str):
    if engine is None:
        return set()
    with engine.connect() as connection:
        query = text("SELECT rma_id FROM rmas WHERE store_url=:store_url AND status=:status")
        rows = connection.execute(query, {"store_url": STORE_URL, "status": status}).fetchall()
    return {r[0] for r in rows}


def set_last_sync(scope: str, dt: datetime):
    if engine is None:
        return
    with engine.connect() as connection:
        query = text("""
            INSERT INTO sync_logs (scope, last_sync_iso) 
            VALUES (:scope, :last_sync_iso)
            ON CONFLICT (scope) DO UPDATE SET last_sync_iso = :last_sync_iso;
        """)
        connection.execute(query, {"scope": scope, "last_sync_iso": dt})
        connection.commit()


def get_last_sync(scope: str) -> Optional[datetime]:
    if engine is None:
        return None
        
    with engine.connect() as connection:
        query = text("SELECT last_sync_iso FROM sync_logs WHERE scope=:scope")
        row = connection.execute(query, {"scope": scope}).fetchone()
        if row and row[0]:
            try:
                return row[0]
            except Exception:
                return None
        return None

def clear_db():
    if engine is None:
        return False
    try:
        with engine.begin() as connection:
            connection.execute(text("TRUNCATE TABLE rmas, sync_logs;"))
        return True
    except Exception as e:
        logger.error(f"Failed to clear PostgreSQL tables: {e}")
        st.error(f"Failed to clear PostgreSQL tables: {e}")
        return False

# ==========================================
# 4. COURIER STATUS
# ==========================================
def _extract_json_object(html: str, start_pos: int) -> Optional[str]:
    depth = 0
    in_string = False
    escape = False

    for i in range(start_pos, len(html)):
        char = html[i]

        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return html[start_pos : i + 1]

        if depth < 0:
            return None

    return None


def _extract_json_array(html: str, start_pos: int) -> Optional[str]:
    depth = 0
    in_string = False
    escape = False

    for i in range(start_pos, len(html)):
        char = html[i]

        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return html[start_pos : i + 1]

        if depth < 0:
            return None

    return None


def check_courier_status_web_scraping(tracking_number: str) -> str:
    if not tracking_number:
        logger.debug("check_courier_status_web_scraping: No tracking number provided")
        return "No tracking number"

    try:
        url = f"https://optimise.parcelninja.com/shipment/track?WaybillNo={tracking_number}"
        logger.debug("Fetching courier status for %s from %s", tracking_number, url)
        session = get_parcel_session()
        res = session.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)

        if res.status_code == 404:
            logger.warning("Tracking not found for %s (404)", tracking_number)
            return "Tracking not found (404)"
        if res.status_code in (401, 403):
            return "Tracking blocked/unauthorised (401/403)"
        if res.status_code == 429:
            return "Tracking rate limited (429)"
        if 500 <= res.status_code <= 599:
            return "Tracking service error (5xx)"
        if res.status_code != 200:
            return f"Tracking error ({res.status_code})"

        html = res.text

        def format_event_date(date_str: str) -> str:
            if not date_str:
                return ""
            try:
                dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                return dt.strftime("%Y-%m-%d %H:%M")
            except (ValueError, TypeError):
                return str(date_str)[:16]

        def format_event(event: dict) -> str:
            status = (
                event.get("status")
                or event.get("description")
                or event.get("event")
                or event.get("statusDescription")
                or "Unknown"
            )
            date_str = (
                event.get("date")
                or event.get("eventDate")
                or event.get("timestamp")
                or event.get("datetime")
                or ""
            )
            formatted_date = format_event_date(date_str)
            return f"{status}\t{formatted_date}" if formatted_date else status

        def extract_events(data: dict) -> Optional[list]:
            paths = [
                ["shipment", "events"],
                ["shipment", "checkpoints"],
                ["tracking", "events"],
                ["tracking", "checkpoints"],
                ["data", "events"],
                ["data", "checkpoints"],
                ["events"],
                ["checkpoints"],
            ]

            for path in paths:
                obj = data
                for key in path:
                    obj = obj.get(key) if isinstance(obj, dict) else None
                    if obj is None:
                        break
                if isinstance(obj, list) and obj:
                    return obj
            return None

        var_pattern = re.compile(
            r"(?:window\.__INITIAL_STATE__|(?:var|let|const)\s+(?:trackingData|shipmentData|initialData))\s*=\s*(\{)",
            re.IGNORECASE,
        )

        for match in var_pattern.finditer(html):
            json_start = match.start(1)
            json_str = _extract_json_object(html, json_start)
            if not json_str:
                continue
            try:
                data = json.loads(json_str)
                events = extract_events(data)
                if events:
                    return format_event(events[-1])
            except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
                continue

        array_pattern = re.compile(r'"(?:events|checkpoints)"\s*:\s*(\[)', re.IGNORECASE)
        for match in array_pattern.finditer(html):
            array_start = match.start(1)
            array_str = _extract_json_array(html, array_start)
            if not array_str:
                continue
            try:
                events = json.loads(array_str)
                if isinstance(events, list) and events:
                    return format_event(events[-1])
            except (json.JSONDecodeError, TypeError, ValueError):
                continue

        clean_html = re.sub(r"<(script|style).*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
        rows = re.findall(r"<tr[^>]*>.*?</tr>", clean_html, flags=re.IGNORECASE | re.DOTALL)
        if not rows:
            return "No tracking events found"

        def clean_text(value: str) -> str:
            text = re.sub(r"<[^>]+>", " ", value)
            return re.sub(r"\s+", " ", text).strip()

        for row_html in rows:
            cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row_html, flags=re.IGNORECASE | re.DOTALL)
            cleaned_cells = [clean_text(c) for c in cells if clean_text(c)]
            if len(cleaned_cells) >= 2 and re.search(r"\d", cleaned_cells[0]):
                logger.info("Found tracking status for %s: %s", tracking_number, cleaned_cells[0])
                return "\t".join(cleaned_cells[:2])

        for row_html in rows:
            cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row_html, flags=re.IGNORECASE | re.DOTALL)
            cleaned_cells = [clean_text(c) for c in cells if clean_text(c)]
            if cleaned_cells:
                return "\t".join(cleaned_cells[:2]) if len(cleaned_cells) >= 2 else cleaned_cells[0]

        status_keywords = ["collected", "delivered", "transit", "delayed", "exception", "returned"]
        date_pattern = re.compile(r"\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}")
        for keyword in status_keywords:
            match = re.search(rf"({keyword}[^<>]*?({date_pattern.pattern})?)", clean_html, re.IGNORECASE)
            if match:
                logger.info("Fallback status extraction for %s: %s", tracking_number, match.group(1))
                return match.group(1).strip()

        logger.warning("No tracking events found for %s in HTML response", tracking_number)
        return "No tracking events found"

    except requests.Timeout:
        logger.warning("Timeout checking tracking for %s", tracking_number)
        return "Tracking check timed out"
    except requests.ConnectionError as exc:
        logger.error("Connection error for %s: %s", tracking_number, exc)
        return "Tracking service unavailable"
    except requests.RequestException as exc:
        logger.error("Request failed for %s: %s", tracking_number, exc, exc_info=True)
        return "Tracking request failed (ERR_REQ)"
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON from ParcelNinja for %s: %s", tracking_number, exc)
        return "Tracking data invalid (ERR_JSON)"
    except Exception:
        logger.exception("Unexpected tracking error for %s", tracking_number)
        return "Tracking check failed (ERR_UNKNOWN)"


@st.cache_data(ttl=COURIER_REFRESH_HOURS * 3600, show_spinner=False)
def check_courier_status(tracking_number: str) -> str:
    if not tracking_number:
        return "No tracking number"
    return check_courier_status_web_scraping(tracking_number)


# ==========================================
# 5. RETURNGO FETCHING
# ==========================================
def fetch_rma_list(statuses, since_dt: Optional[datetime]) -> Tuple[list, bool, Optional[str]]:
    all_summaries = []
    cursor = None
    status_param = ",".join(statuses)
    had_success = False
    last_error = None

    updated_filter = ""
    if since_dt is not None:
        since_dt = since_dt - timedelta(minutes=SYNC_OVERLAP_MINUTES)
        updated_filter = f"&rma_updated_at=gte:{_iso_utc(since_dt)}"

    while True:
        base_url = f"{api_url('/rmas')}?pagesize=500&status={status_param}{updated_filter}"
        url = f"{base_url}&cursor={cursor}" if cursor else base_url

        res = rg_request("GET", url, timeout=20)
        if res is None:
            last_error = "No response from ReturnGO"
            break
        if res.status_code != 200:
            err_text = (res.text or "").strip()
            if len(err_text) > 200:
                err_text = f"{err_text[:200]}..."
            last_error = f"{res.status_code}: {err_text}" if err_text else str(res.status_code)
            break

        had_success = True
        try:
            data = res.json() if res.content else {}
        except ValueError:
            last_error = "Invalid JSON response from ReturnGO"
            break
        rmas = data.get("rmas", []) or []
        if not rmas:
            break

        all_summaries.extend(rmas)
        cursor = data.get("next_cursor")
        if not cursor:
            break

    return all_summaries, had_success, last_error


def should_refresh_detail(rma_id: str) -> bool:
    cached = get_rma(rma_id)
    if not cached:
        return True
    _, last_fetched_iso = cached
    if not last_fetched_iso:
        return True
    try:
        last_dt = datetime.fromisoformat(last_fetched_iso)
    except (ValueError, TypeError):
        return True
    if last_dt.tzinfo is None:
        last_dt = last_dt.replace(tzinfo=timezone.utc)
    return (_now_utc() - last_dt) > timedelta(hours=CACHE_EXPIRY_HOURS)


@st.cache_data(ttl=COURIER_REFRESH_HOURS * 3600, show_spinner=False)
def should_refresh_courier(rma_payload: dict) -> bool:
    cached_checked = rma_payload.get("_local_courier_checked")
    if not cached_checked:
        return True
    try:
        last_chk = datetime.fromisoformat(cached_checked)
        if last_chk.tzinfo is None:
            last_chk = last_chk.replace(tzinfo=timezone.utc)
        return (_now_utc() - last_chk) > timedelta(hours=COURIER_REFRESH_HOURS)
    except (ValueError, TypeError):
        return True


def maybe_refresh_courier(rma_payload: dict) -> Tuple[Optional[str], Optional[str]]:
    shipments = rma_payload.get("shipments", []) or []
    track_no = None
    for s in shipments:
        if s.get("trackingNumber"):
            track_no = s.get("trackingNumber")
            break
    if not track_no:
        return "No tracking number", None

    cached_status = rma_payload.get("_local_courier_status")
    cached_checked = rma_payload.get("_local_courier_checked")
    
    status = check_courier_status(track_no)
    
    if status != cached_status or cached_checked is None:
        checked_iso = _iso_utc(_now_utc())
    else:
        checked_iso = cached_checked
    return status, checked_iso


def fetch_rma_detail(rma_id: str, *, force: bool = False):
    cached = get_rma(rma_id)
    if (not force) and cached and not should_refresh_detail(rma_id):
        return cached[
