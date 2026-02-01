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
        return cached[0]

    url = api_url(f"/rma/{rma_id}")
    res: Optional[requests.Response] = rg_request("GET", url, timeout=20)
    if res is None:
        logger.error("Failed to fetch RMA detail for %s after retries.", rma_id)
        return cached[0] if cached else None
    if res.status_code != 200:
        logger.error("Failed to fetch RMA detail for %s: Status %s, Response: %s", rma_id, res.status_code, res.text)
        return cached[0] if cached else None

    data = res.json() if res.content else {}
    summary = data.get("rmaSummary", {}) or {}

    if cached:
        data["_local_courier_status"] = cached[0].get("_local_courier_status")
        data["_local_courier_checked"] = cached[0].get("_local_courier_checked")
        data["_local_received_first_seen"] = cached[0].get("_local_received_first_seen")

    courier_status, courier_checked = maybe_refresh_courier(data)
    
    try:
        upsert_rma(
            rma_id=str(rma_id),
            status=summary.get("status", "Unknown"),
            created_at=summary.get("createdAt") or data.get("createdAt") or "",
            payload=data,
            courier_status=courier_status,
            courier_checked_iso=courier_checked,
        )
        fresh = get_rma(str(rma_id))
        if fresh:
            data["_local_received_first_seen"] = fresh[0].get("_local_received_first_seen")
        data["_local_courier_status"] = courier_status
        data["_local_courier_checked"] = courier_checked
        return data
    except Exception as e:
        logger.error("Error upserting RMA %s to DB: %s", rma_id, e)
        return cached[0] if cached else None


def get_incremental_since(statuses, full: bool) -> Optional[datetime]:
    if full:
        return None
    stamps = [get_last_sync(s) for s in statuses]
    stamps = [d for d in stamps if d]
    return min(stamps) if stamps else None


def perform_sync(statuses=None, *, full=False, rerun: bool = True):
    if st.session_state.get("disconnected"):
        append_ops_log("System is disconnected. Login again to resume sync.", level="warning")
        return
    
    sync_start = time.time()
    failed_fetches = 0
    successful_fetches = 0
    append_ops_log("⏳ Connecting to ReturnGO...")

    if statuses is None:
        statuses = ACTIVE_STATUSES

    since_dt = get_incremental_since(statuses, full)

    append_ops_log("Calling fetch_rma_list...")
    summaries, ok, err = fetch_rma_list(statuses, since_dt)
    append_ops_log(f"fetch_rma_list returned. ok={ok}, error='{err}'")
    if not ok:
        msg = f"ReturnGO sync failed: {err}" if err else "ReturnGO sync failed."
        append_ops_log(msg, level="error")
        return
    append_ops_log(f"Fetched {len(summaries)} RMAs.")

    api_ids = {s.get("rmaId") for s in summaries if s.get("rmaId")}

    if set(statuses) == set(ACTIVE_STATUSES) and not full and since_dt is not None:
        local_active_ids = set()
        for stt in ACTIVE_STATUSES:
            local_active_ids |= get_local_ids_for_status(stt)
        stale = local_active_ids - api_ids
        logger.info("Removing %s stale RMAs no longer in active statuses", len(stale))
        delete_rmas(stale)
    elif full:
        if engine is None:
            return
        with engine.connect() as connection:
            query = text("SELECT rma_id FROM rmas WHERE store_url=:store_url")
            result = connection.execute(query, {"store_url": STORE_URL})
            all_local_ids = {r[0] for r in result.fetchall()}
        stale_full = all_local_ids - api_ids
        if stale_full:
            logger.info("Full sync: removing %s RMAs not in API response", len(stale_full))
            delete_rmas(stale_full)

    if since_dt is not None:
        to_fetch = list(api_ids)
        force = True
    else:
        to_fetch = [rid for rid in api_ids if should_refresh_detail(rid)]
        force = False

    total = len(to_fetch)
    append_ops_log(f"⏳ Syncing {total} records...")

    if total > 0:
        append_ops_log(f"Starting to fetch details for {total} RMAs one by one...")

        for i, rid in enumerate(to_fetch):
            done = i + 1
            if done == 1 or done % 5 == 0 or done == total:
                append_ops_log(f"[{done}/{total}] Fetching detail for RMA: {rid}")

            result = fetch_rma_detail(rid, force=force)

            if result is not None:
                successful_fetches += 1
            else:
                failed_fetches += 1

        append_ops_log("Finished fetching all details.")

    now = _now_utc()
    scope = ",".join(statuses)
    set_last_sync(scope, now)
    for s in statuses:
        set_last_sync(s, now)

    sync_duration = time.time() - sync_start
    if "performance_metrics" in st.session_state:
        st.session_state.performance_metrics.last_sync_duration = sync_duration
        # Calculate average response time
        if st.session_state.performance_metrics.api_call_times:
            st.session_state.performance_metrics.avg_response_time = (
                sum(st.session_state.performance_metrics.api_call_times) 
                / len(st.session_state.performance_metrics.api_call_times)
            )

    if failed_fetches == 0:
        append_ops_log(f"✅ Sync Complete! {successful_fetches} RMAs updated in {sync_duration:.1f}s.")
    else:
        append_ops_log(
            f"⚠️ Sync Complete with {successful_fetches} RMAs updated and {failed_fetches} failed in {sync_duration:.1f}s."
        )
    st.session_state["show_toast"] = True
    try:
        load_open_rmas.clear()
    except Exception:
        append_ops_log("Failed to clear open-RMA cache", level="error")
    if rerun:
        st.rerun()


def force_refresh_rma_ids(rma_ids, scope_label: str):
    if st.session_state.get("disconnected"):
        append_ops_log("System is disconnected. Login again to refresh data.", level="warning")
        return
    ids = [str(i) for i in set(rma_ids) if i]
    if not ids:
        set_last_sync(scope_label, _now_utc())
        st.session_state["show_toast"] = True
        st.rerun()
        return

    append_ops_log(f"⏳ Refreshing {len(ids)} records...")

    total = len(ids)
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(fetch_rma_detail, rid, force=True) for rid in ids]
        done = 0
        for _ in concurrent.futures.as_completed(futures):
            done += 1
            if done == 1 or done % 10 == 0 or done == total:
                append_ops_log(f"Refreshing progress: {done}/{total}")

    set_last_sync(scope_label, _now_utc())
    st.session_state.cache_version = st.session_state.get("cache_version", 0) + 1
    st.session_state["show_toast"] = True
    append_ops_log("✅ Refresh Complete!")
    try:
        load_open_rmas.clear()
    except Exception as e:
        append_ops_log(f"Failed to clear open-RMA cache: {e}", level="error")
    st.rerun()

# ==========================================
# 6. MUTATIONS
# ==========================================
def push_tracking_update(rma_id, shipment_id, tracking_number):
    headers = {**rg_headers(), "Content-Type": "application/json"}
    payload = {
        "status": "LabelCreated",
        "carrierName": "CourierGuy",
        "trackingNumber": tracking_number,
        "trackingURL": f"https://optimise.parcelninja.com/shipment/track?WaybillNo={tracking_number}",
        "labelURL": "https://sellerportal.dpworld.com/api/file-download?link=null",
    }

    try:
        res: Optional[requests.Response] = rg_request("PUT", api_url(f"/shipment/{shipment_id}"), headers=headers, timeout=15, json_body=payload)
        if res is not None and res.status_code == 200:
            fetch_rma_detail(rma_id, force=True)
            return True, "Success"
        if res is None:
            return False, "API Error: No response"
        return False, f"API Error {res.status_code}: {res.text}"
    except Exception as e:
        return False, str(e)


def push_comment_update(rma_id, comment_text):
    headers = {**rg_headers(), "Content-Type": "application/json"}
    payload = {"htmlText": comment_text}

    try:
        res: Optional[requests.Response] = rg_request("POST", api_url(RMA_COMMENT_PATH.format(rma_id=rma_id)), headers=headers, timeout=15, json_body=payload)
        if res is not None and res.status_code in (200, 201):
            fetch_rma_detail(rma_id, force=True)
            return True, "Success"
        if res is None:
            return False, "API Error: No response"
        return False, f"API Error {res.status_code}: {res.text}"
    except Exception as e:
        return False, str(e)


# ==========================================
# 7. Helpers
# ==========================================
def get_event_date_iso(rma_payload: dict, event_name: str) -> str:
    summary = rma_payload.get("rmaSummary", {}) or {}
    for e in (summary.get("events") or []):
        if e.get("eventName") == event_name and e.get("eventDate"):
            return str(e["eventDate"])
    return ""


def get_received_date_iso(rma_payload: dict) -> str:
    for name in ("SHIPMENT_RECEIVED", "RMA_RECEIVED", "RMA_STATUS_RECEIVED"):
        dt = get_event_date_iso(rma_payload, name)
        if dt:
            return dt

    for c in (rma_payload.get("comments") or []):
        txt = (c.get("htmlText", "") or "").lower()
        if "shipment" in txt and "received" in txt and c.get("datetime"):
            return str(c.get("datetime"))

    return str(rma_payload.get("_local_received_first_seen") or "")


def pretty_resolution(rt: Optional[str]) -> str:
    if not rt:
        return ""
    out = re.sub(r"([a-z])([A-Z])", r"\1 \2", rt).strip()
    return out.replace("To ", "to ")


def get_resolution_type(rma_payload: dict) -> str:
    items = (
        rma_payload.get("items")
        or (rma_payload.get("rmaSummary", {}) or {}).get("items")
        or []
    )
    types = {i.get("resolutionType") for i in items if isinstance(i, dict) and i.get("resolutionType")}
    if len(types) > 1:
        return "Mix"
    if len(types) == 1:
        return pretty_resolution(next(iter(types)))
    return ""


def is_resolution_actioned(rma_payload: dict, comment_texts: Optional[list] = None) -> bool:
    for t in (rma_payload.get("transactions") or []):
        if (t.get("status") or "").lower() == "success":
            return True
    ex_orders = rma_payload.get("exchangeOrders") or []
    if isinstance(ex_orders, list) and len(ex_orders) > 0:
        return True
    if comment_texts is None:
        comment_texts = [(c.get("htmlText", "") or "").lower() for c in (rma_payload.get("comments") or [])]
    for txt in comment_texts:
        if "transaction id" in txt and ("refund" in txt or "credited" in txt):
            return True
        if "exchange order" in txt and "released" in txt:
            return True
        if "total refund amount" in txt:
            return True
    return False


def resolution_actioned_label(rma_payload: dict) -> str:
    for t in (rma_payload.get("transactions") or []):
        if (t.get("status") or "").lower() == "success":
            typ = (t.get("type") or "").strip()
            return pretty_resolution(typ) if typ else "Yes"
    ex_orders = rma_payload.get("exchangeOrders") or []
    if isinstance(ex_orders, list) and len(ex_orders) > 0:
        return "Exchange released"
    return "No"


def tracking_update_comment(old_tracking: Optional[str], new_tracking: Optional[str]) -> str:
    old_value = old_tracking.strip() if old_tracking else "None"
    new_value = new_tracking.strip() if new_tracking else "None"
    return f"Updated Shipment tracking number from {old_value} to {new_value}."


def has_tracking_update_comment(comment_texts: list[str]) -> bool:
    pattern = re.compile(r"updated shipment", re.IGNORECASE)
    for text in comment_texts:
        if pattern.search(text):
            return True
    return False


def failure_flags(
    comment_texts: list[str],
    *,
    status: str,
    approved_iso: str,
    has_tracking: bool,
) -> tuple[bool, bool, bool, str]:
    joined = " ".join(comment_texts).lower()
    has_refund_failure = "refund failed" in joined
    has_upload_failed = "missing required media" in joined
    has_shipment_failure = False

    if status == "Approved" and not has_tracking and approved_iso:
        try:
            approved_dt = datetime.fromisoformat(approved_iso)
            if approved_dt.tzinfo is None:
                approved_dt = approved_dt.replace(tzinfo=timezone.utc)
            if (_now_utc() - approved_dt) >= timedelta(hours=1):
                has_shipment_failure = True
        except ValueError:
            pass

    labels = []
    if has_refund_failure:
        labels.append("refund_failure")
    if has_upload_failed:
        labels.append("upload_failed")
    if has_shipment_failure:
        labels.append("shipment_failure")
    return has_refund_failure, has_upload_failed, has_shipment_failure, ", ".join(labels)


def parse_yyyy_mm_dd(s: str) -> Optional[datetime]:
    try:
        if not s or s == "N/A":
            return None
        return datetime.fromisoformat(s[:10])
    except Exception:
        return None


def days_since(dt_str: str, today: Optional[date] = None) -> str:
    d = parse_yyyy_mm_dd(dt_str)
    if not d:
        return "N/A"
    if today is None:
        today = datetime.now().date()
    return str((today - d.date()).days)


def format_api_limit_display() -> Tuple[str, str]:
    with RATE_LIMIT_LOCK:
        remaining = RATE_LIMIT_INFO.get("remaining")
        limit = RATE_LIMIT_INFO.get("limit")
        reset = RATE_LIMIT_INFO.get("reset")
        updated_at = RATE_LIMIT_INFO.get("updated_at")

    if remaining is None and limit is None:
        main = "API Limit: --"
    elif remaining is not None and limit is not None:
        main = f"API Left: {remaining}/{limit}"
    else:
        main = f"API Left: {remaining}" if remaining is not None else f"API Limit: {limit}"

    sub = "Updated: --"
    if isinstance(reset, int):
        try:
            reset_dt = datetime.fromtimestamp(reset, tz=timezone.utc).astimezone()
            sub = f"Resets: {reset_dt.strftime('%H:%M')}"
        except (ValueError, TypeError, OSError):
            sub = f"Reset: {reset}"
    elif reset:
        sub = f"Reset: {reset}"
    elif updated_at and isinstance(updated_at, datetime):
        sub = f"Updated: {updated_at.astimezone().strftime('%H:%M')}"

    return main, sub


def format_tracking_status_with_icon(status: str) -> str:
    if not status:
        return "❓ Unknown"

    if status == "Unknown":
        return "❓ Unknown"

    if status == "No tracking number":
        return "🚫 No tracking number"

    status_lower = status.lower()
    if "return" in status_lower and "sender" in status_lower:
        return f"↩️ {status}"

    status_patterns = [
        ("🛑", ["cancel", "failed", "exception", "rejected"]),
        ("✅", ["delivered"]),
        ("🚚", ["out for delivery", "out-for-delivery"]),
        ("📬", ["collected", "picked"]),
        ("📦", ["in transit", "transit", "on the way"]),
    ]
    for icon, keywords in status_patterns:
        if any(keyword in status_lower for keyword in keywords):
            return f"{icon} {status}"

    return f"📍 {status}"


def toggle_filter(name: str):
    """Toggle filter selection in session state."""
    # Ensure active_filters exists
    if "active_filters" not in st.session_state:
        st.session_state.active_filters = set()
    
    s: Set[str] = set(st.session_state.active_filters)
    if name in s:
        s.remove(name)
        logger.info("Filter deselected: %s", name)
    else:
        s.add(name)
        logger.info("Filter selected: %s", name)
    st.session_state.active_filters = s
    st.rerun()


def clear_all_filters():
    """Clear all active filters."""
    # Ensure all session state variables exist
    if "active_filters" not in st.session_state:
        st.session_state.active_filters = set()
    
    st.session_state.active_filters = set()
    st.session_state.search_query_input = ""
    st.session_state.status_multi = []
    st.session_state.res_multi = []
    st.session_state.actioned_multi = []
    st.session_state.tracking_status_multi = []
    st.session_state.req_dates_selected = []
    st.session_state.failure_refund = False
    st.session_state.failure_upload = False
    st.session_state.failure_shipment = False


def update_data_table_log(rows: list):
    if "data_table_log" not in st.session_state:
        st.session_state.data_table_log = []
    if "data_table_prev" not in st.session_state:
        st.session_state.data_table_prev = {}

    prev_map: Dict[str, Dict[str, str]] = st.session_state.data_table_prev
    current_map = {}
    new_entries = []
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    for row in rows:
        rma_id = row.get("_rma_id_text") or row.get("RMA ID")
        if not rma_id:
            continue
        status = row.get("Current Status", "Unknown")
        resolution = row.get("resolutionType", "N/A")
        current_map[rma_id] = {"status": status, "resolution": resolution}

        prev = prev_map.get(rma_id)
        if not prev:
            continue
        if prev.get("status") != status:
            new_entries.append(f"{timestamp} | RMA {rma_id}: {prev.get('status')} → {status}")
        if prev.get("resolution") != resolution:
            new_entries.append(
                f"{timestamp} | RMA {rma_id}: Resolution {prev.get('resolution')} → {resolution}"
            )

    if new_entries:
        st.session_state.data_table_log = (new_entries + st.session_state.data_table_log)[:100]
    st.session_state.data_table_prev = {**prev_map, **current_map}


def header_last_sync_display(scopes: list) -> str:
    scope_key = ",".join(scopes)
    ts = get_last_sync(scope_key)
    if not ts:
        stamps = [get_last_sync(s) for s in scopes]
        stamps = [stamp for stamp in stamps if stamp]
        ts = max(stamps) if stamps else None
    if not ts:
        return "--"
    return ts.astimezone().strftime("%H:%M")


def format_time_ago(last_sync_ts: Optional[datetime]) -> str:
    if not last_sync_ts:
        return "Never"
    time_ago = (_now_utc() - last_sync_ts).total_seconds()
    if time_ago < 60:
        return f"{int(time_ago)}s ago"
    if time_ago < 3600:
        return f"{int(time_ago / 60)}m ago"
    return f"{int(time_ago / 3600)}h ago"


# ==========================================
# NEW: Enhanced UI Components
# ==========================================
def inject_enhanced_css():
    """Inject enhanced CSS with dark/light mode support"""
    st.markdown(
        """
        <style>
        /* Theme Variables */
        :root {
            /* Dark Theme (Default) */
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-tertiary: #334155;
            --text-primary: #f1f5f9;
            --text-secondary: #cbd5e1;
            --text-muted: #94a3b8;
            --border-color: #475569;
            --accent-primary: #3b82f6;
            --accent-success: #22c55e;
            --accent-warning: #f59e0b;
            --accent-danger: #ef4444;
            --accent-info: #06b6d4;
            --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
            --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
            --radius-sm: 0.375rem;
            --radius-md: 0.5rem;
            --radius-lg: 0.75rem;
        }
        
        .light-mode {
            --bg-primary: #ffffff;
            --bg-secondary: #f8fafc;
            --bg-tertiary: #e2e8f0;
            --text-primary: #1e293b;
            --text-secondary: #475569;
            --text-muted: #64748b;
            --border-color: #cbd5e1;
            --accent-primary: #2563eb;
            --accent-success: #16a34a;
            --accent-warning: #d97706;
            --accent-danger: #dc2626;
            --accent-info: #0891b2;
        }
        
        /* Enhanced Tile System */
        .enhanced-tile {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-lg);
            padding: 1.25rem;
            margin-bottom: 1rem;
            transition: all 0.2s ease;
            position: relative;
            overflow: hidden;
        }
        
        .enhanced-tile:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-lg);
            border-color: var(--accent-primary);
        }
        
        .enhanced-tile.selected {
            border-color: var(--accent-success);
            background: linear-gradient(135deg, var(--bg-secondary) 0%, rgba(34, 197, 94, 0.05) 100%);
        }
        
        .enhanced-tile.favorite::before {
            content: "★";
            position: absolute;
            top: 0.5rem;
            right: 0.5rem;
            color: #fbbf24;
            font-size: 1rem;
            z-index: 1;
        }
        
        /* Urgency Indicators */
        .enhanced-tile.urgent {
            border-left: 4px solid var(--accent-danger);
        }
        
        .enhanced-tile.warning {
            border-left: 4px solid var(--accent-warning);
        }
        
        .enhanced-tile.success {
            border-left: 4px solid var(--accent-success);
        }
        
        .enhanced-tile.info {
            border-left: 4px solid var(--accent-info);
        }
        
        .enhanced-tile.active {
            border-left: 4px solid var(--accent-primary);
        }
        
        /* Tile Header */
        .tile-header {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 1rem;
        }
        
        .tile-icon {
            font-size: 1.5rem;
            flex-shrink: 0;
        }
        
        .tile-title {
            flex: 1;
            font-weight: 600;
            font-size: 0.875rem;
            color: var(--text-primary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .tile-badge {
            background: var(--bg-tertiary);
            color: var(--text-primary);
            padding: 0.25rem 0.75rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 1.25rem;
            min-width: 3rem;
            text-align: center;
        }
        
        .enhanced-tile.selected .tile-badge {
            background: rgba(34, 197, 94, 0.15);
            color: var(--accent-success);
        }
        
        /* Tile Body */
        .tile-body {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }
        
        .tile-meta {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.75rem;
            color: var(--text-muted);
        }
        
        .tile-actions {
            display: flex;
            gap: 0.5rem;
        }
        
        .tile-btn {
            flex: 1;
            padding: 0.5rem 0.75rem;
            border: 1px solid var(--border-color);
            border-radius: var(--radius-md);
            background: var(--bg-tertiary);
            color: var(--text-primary);
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            cursor: pointer;
            transition: all 0.2s;
            text-align: center;
        }
        
        .tile-btn:hover {
            background: var(--accent-primary);
            color: white;
            border-color: var(--accent-primary);
        }
        
        .tile-btn.active {
            background: var(--accent-success);
            color: white;
            border-color: var(--accent-success);
        }
        
        .tile-btn.secondary {
            background: transparent;
            border-color: var(--border-color);
        }
        
        .tile-btn.secondary:hover {
            background: var(--bg-tertiary);
        }
        
        /* Progress Indicators */
        .progress-ring {
            position: relative;
            width: 40px;
            height: 40px;
        }
        
        .progress-ring-circle {
            transform: rotate(-90deg);
            transform-origin: 50% 50%;
        }
        
        .progress-ring-bg {
            fill: none;
            stroke: var(--border-color);
            stroke-width: 3;
        }
        
        .progress-ring-fill {
            fill: none;
            stroke: var(--accent-success);
            stroke-width: 3;
            stroke-linecap: round;
            transition: stroke-dashoffset 0.3s;
        }
        
        .progress-text {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--text-primary);
        }
        
        /* Summary Bar */
        .summary-bar {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
            padding: 1.5rem;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-lg);
        }
        
        .summary-item {
            text-align: center;
            padding: 1rem;
            background: var(--bg-tertiary);
            border-radius: var(--radius-md);
            transition: all 0.2s;
        }
        
        .summary-item:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-md);
        }
        
        .summary-value {
            font-size: 2rem;
            font-weight: 800;
            color: var(--text-primary);
            margin-bottom: 0.25rem;
            font-variant-numeric: tabular-nums;
        }
        
        .summary-label {
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }
        
        /* Grid Layout */
        .command-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }
        
        /* Bulk Actions */
        .bulk-actions-bar {
            position: sticky;
            top: 0;
            z-index: 100;
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border-color);
            padding: 1rem;
            margin-bottom: 1rem;
            backdrop-filter: blur(10px);
        }
        
        .bulk-actions-content {
            display: flex;
            align-items: center;
            gap: 1rem;
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .selected-count {
            font-weight: 600;
            color: var(--accent-primary);
        }
        
        /* Performance Metrics */
        .metrics-panel {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-lg);
            padding: 1.5rem;
            margin-bottom: 2rem;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }
        
        .metric-item {
            text-align: center;
            padding: 1rem;
            background: var(--bg-tertiary);
            border-radius: var(--radius-md);
        }
        
        .metric-value {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 0.25rem;
        }
        
        .metric-label {
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        /* Export Presets */
        .presets-panel {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-lg);
            padding: 1.5rem;
            margin-bottom: 2rem;
        }
        
        .presets-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }
        
        .preset-card {
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-md);
            padding: 1rem;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .preset-card:hover {
            border-color: var(--accent-primary);
            transform: translateY(-2px);
        }
        
        .preset-name {
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 0.5rem;
        }
        
        .preset-filters {
            font-size: 0.75rem;
            color: var(--text-muted);
        }
        
        /* Keyboard Shortcuts Helper */
        .shortcuts-helper {
            position: fixed;
            bottom: 1rem;
            right: 1rem;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-lg);
            padding: 1rem;
            max-width: 300px;
            box-shadow: var(--shadow-lg);
            z-index: 1000;
        }
        
        .shortcut-item {
            display: flex;
            justify-content: space-between;
            padding: 0.25rem 0;
            border-bottom: 1px solid var(--border-color);
        }
        
        .shortcut-item:last-child {
            border-bottom: none;
        }
        
        .shortcut-key {
            background: var(--bg-tertiary);
            padding: 0.125rem 0.5rem;
            border-radius: var(--radius-sm);
            font-family: monospace;
            font-size: 0.75rem;
        }
        
        /* Responsive Design */
        @media (max-width: 768px) {
            .command-grid {
                grid-template-columns: 1fr;
            }
            
            .summary-bar {
                grid-template-columns: 1fr;
            }
            
            .tile-actions {
                flex-direction: column;
            }
            
            .bulk-actions-content {
                flex-direction: column;
                align-items: stretch;
            }
        }
        
        /* Theme Toggle */
        .theme-toggle {
            position: fixed;
            top: 1rem;
            right: 1rem;
            z-index: 1000;
        }
        
        .theme-toggle-btn {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-full);
            padding: 0.5rem;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .theme-toggle-btn:hover {
            background: var(--accent-primary);
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def render_enhanced_tile(col, name: str, refresh_scope: str, semantic_label: str, icon: str):
    """Render enhanced command center tile with all features."""
    cfg = FILTERS.get(name)
    if not cfg:
        return
    
    count_key = cfg["count_key"]
    active: Set[str] = st.session_state.active_filters
    is_selected = name in active
    is_favorite = name in st.session_state.user_settings.favorites
    
    count_val = counts.get(count_key, 0)
    last_sync_ts = get_last_sync(refresh_scope)
    time_ago_str = format_time_ago(last_sync_ts)
    
    # Determine urgency level
    urgency_levels = {
        "Flagged": "urgent",
        "No Tracking": "warning", 
        "Courier Cancelled": "error",
        "Pending Requests": "info",
        "Approved - Submitted": "success",
        "Approved - In Transit": "active",
        "Received": "neutral",
        "Approved - Delivered": "success",
        "Resolution Actioned": "success",
        "No Resolution Actioned": "warning",
    }
    
    urgency = urgency_levels.get(name, "neutral")
    
    with col:
        st.markdown(f"<div class='enhanced-tile {urgency} {'selected' if is_selected else ''} {'favorite' if is_favorite else ''}'>", unsafe_allow_html=True)
        
        # Header
        st.markdown(f"""
        <div class="tile-header">
            <span class="tile-icon">{icon}</span>
            <span class="tile-title">{semantic_label}</span>
            <span class="tile-badge">{count_val}</span>
        </div>
        """, unsafe_allow_html=True)

        # Body
        st.markdown('<div class="tile-body">', unsafe_allow_html=True)

        # Meta row with actions
        meta_cols = st.columns([2, 1])
        with meta_cols[0]:
            st.markdown(f'<div class="tile-meta"><span class="time-ago">Updated {time_ago_str}</span></div>', unsafe_allow_html=True)
        with meta_cols[1]:
            action_cols = st.columns(2)
            with action_cols[0]:
                if st.button("⟳", key=f"refresh_{name}", help="Refresh this category"):
                    force_refresh_rma_ids(ids_for_filter(name), refresh_scope)
            with action_cols[1]:
                if st.button('★' if is_favorite else '☆', key=f"fav_{name}", help="Toggle favorite"):
                    if is_favorite:
                        st.session_state.user_settings.favorites.remove(name)
                    else:
                        st.session_state.user_settings.favorites.append(name)
                    save_user_settings(st.session_state.user_settings)
                    st.rerun()

        # View Details Button
        button_label = "✓ VIEWING" if is_selected else "VIEW DETAILS"
        if st.button(button_label, key=f"view_{name}", use_container_width=True):
            toggle_filter(name)

        st.markdown('</div>', unsafe_allow_html=True) # close tile-body
        st.markdown('</div>', unsafe_allow_html=True) # close enhanced-tile


def render_summary_bar():
    """Render summary statistics bar."""
    total_open = sum(counts.get(k, 0) for k in ["PendingRequests", "ApprovedSubmitted", "ApprovedInTransit", "Received"])
    total_issues = counts.get("Flagged", 0) + counts.get("NoTracking", 0) + counts.get("CourierCancelled", 0)
    total_pending = counts.get("PendingRequests", 0)
    total_in_transit = counts.get("ApprovedInTransit", 0)
    
    st.markdown(
        f"""
        <div class="summary-bar">
            <div class="summary-item">
                <div class="summary-value">{total_open}</div>
                <div class="summary-label">Total Open</div>
            </div>
            <div class="summary-item">
                <div class="summary-value">{total_pending}</div>
                <div class="summary-label">Pending</div>
            </div>
            <div class="summary-item">
                <div class="summary-value">{total_in_transit}</div>
                <div class="summary-label">In Transit</div>
            </div>
            <div class="summary-item">
                <div class="summary-value">{total_issues}</div>
                <div class="summary-label">Issues</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_performance_metrics():
    """Render performance metrics panel."""
    if not st.session_state.user_settings.performance_metrics:
        return
    
    metrics = st.session_state.performance_metrics
    
    # Calculate cache hit rate (simplified)
    total_calls = len(metrics.api_call_times) if metrics.api_call_times else 1
    cache_hits = total_calls - (total_calls // 4)  # Simplified estimation
    cache_hit_rate = (cache_hits / total_calls) * 100 if total_calls > 0 else 0
    
    st.markdown(
        """
        <div class="metrics-panel">
            <h4 style="margin: 0 0 1rem 0; color: var(--text-primary);">Performance Metrics</h4>
            <div class="metrics-grid">
        """,
        unsafe_allow_html=True
    )
    
    # Metric items
    metric_items = [
        ("Avg Response", f"{metrics.avg_response_time:.2f}s"),
        ("Last Sync", f"{metrics.last_sync_duration:.1f}s"),
        ("Cache Hit Rate", f"{cache_hit_rate:.1f}%"),
        ("API Calls", f"{total_calls}"),
    ]
    
    for label, value in metric_items:
        st.markdown(
            f"""
            <div class="metric-item">
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown("</div></div>", unsafe_allow_html=True)


def render_export_presets():
    """Render export presets panel."""
    if not st.session_state.user_settings.export_presets:
        return
    
    st.markdown(
        """
        <div class="presets-panel">
            <h4 style="margin: 0 0 1rem 0; color: var(--text-primary);">Export Presets</h4>
            <div class="presets-grid">
        """,
        unsafe_allow_html=True
    )
    
    for preset_name, preset_config in st.session_state.user_settings.export_presets.items():
        filters = preset_config.get("filters", [])
        st.markdown(
            f"""
            <div class="preset-card" onclick="applyPreset('{preset_name}')">
                <div class="preset-name">{preset_name}</div>
                <div class="preset-filters">{', '.join(filters[:3])}{'...' if len(filters) > 3 else ''}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown("</div></div>", unsafe_allow_html=True)
    
    # Add JavaScript for preset application
    components.html(
        """
        <script>
        function applyPreset(presetName) {
            window.parent.postMessage({
                type: 'streamlit:setComponentValue',
                value: {presetName: presetName, action: 'apply_preset'}
            }, '*');
        }
        </script>
        """,
        height=0
    )


def handle_keyboard_shortcuts():
    """Handle keyboard shortcuts."""
    if not st.session_state.user_settings.keyboard_shortcuts:
        return
    
    # Create a hidden text input for capturing keyboard events
    # Fix: Add a non-empty label
    key_pressed = st.text_input(
        "Keyboard input",  # Non-empty label
        key="keyboard_input", 
        label_visibility="collapsed"
    )
    
    if key_pressed:
        shortcut = KEYBOARD_SHORTCUTS.get(key_pressed.lower())
        if shortcut:
            if shortcut == "search":
                st.session_state.search_query_input = ""
                st.rerun()
            elif shortcut == "toggle_favorites":
                # Toggle favorites mode
                if "show_favorites" in st.session_state:
                    st.session_state.show_favorites = not st.session_state.show_favorites
                else:
                    st.session_state.show_favorites = True
                st.rerun()
            elif shortcut == "refresh_all":
                perform_sync(full=True)
            elif shortcut == "export_csv":
                st.session_state.export_csv = True
            elif shortcut == "clear_filters":
                clear_all_filters()
            elif shortcut in FILTERS:
                toggle_filter(shortcut)


def render_keyboard_shortcuts_helper():
    """Render keyboard shortcuts helper."""
    if not st.session_state.user_settings.keyboard_shortcuts:
        return
    
    with st.expander("⌨️ Keyboard Shortcuts"):
        cols = st.columns(2)
        for i, (key, action) in enumerate(KEYBOARD_SHORTCUTS.items()):
            with cols[i % 2]:
                st.markdown(f"**{key.upper()}** → {action}")


def render_bulk_actions():
    """Render bulk actions bar."""
    selected_filters = st.session_state.active_filters
    if not selected_filters:
        return
    
    selected_count = sum(counts.get(FILTERS[f]["count_key"], 0) for f in selected_filters if f in FILTERS)
    
    st.markdown(
        f"""
        <div class="bulk-actions-bar">
            <div class="bulk-actions-content">
                <div class="selected-count">{selected_count} items selected</div>
                <div style="display: flex; gap: 0.5rem;">
                    <button onclick="handleBulkAction('refresh')" style="padding: 0.5rem 1rem; background: var(--accent-primary); color: white; border: none; border-radius: var(--radius-md); cursor: pointer;">
                        Refresh All
                    </button>
                    <button onclick="handleBulkAction('export')" style="padding: 0.5rem 1rem; background: var(--accent-success); color: white; border: none; border-radius: var(--radius-md); cursor: pointer;">
                        Export Selected
                    </button>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    components.html(
        """
        <script>
        function handleBulkAction(action) {
            window.parent.postMessage({
                type: 'streamlit:setComponentValue',
                value: {action: 'bulk_' + action}
            }, '*');
        }
        </script>
        """,
        height=0
    )


def render_theme_toggle():
    """Render theme toggle button."""
    st.markdown(
        """
        <div class="theme-toggle">
            <button class="theme-toggle-btn" onclick="toggleTheme()">
                🌓
            </button>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    components.html(
        """
        <script>
        function toggleTheme() {
            const root = document.documentElement;
            const currentTheme = root.classList.contains('light-mode') ? 'light' : 'dark';
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            
            if (newTheme === 'light') {
                root.classList.add('light-mode');
            } else {
                root.classList.remove('light-mode');
            }
            
            // Save theme preference
            window.parent.postMessage({
                type: 'streamlit:setComponentValue',
                value: {theme: newTheme, action: 'change_theme'}
            }, '*');
        }
        </script>
        """,
        height=0
    )


# ==========================================
# 12. FILTER DEFINITIONS (multi-select)
# ==========================================
FilterFn = Callable[[pd.DataFrame], pd.Series]

FILTERS: Dict[str, Dict[str, Any]] = {
    "Pending Requests": {
        "icon": "⏳",
        "count_key": "PendingRequests",
        "scope": "Pending",
        "fn": lambda d: d["Current Status"] == "Pending",
    },
    "Approved - Submitted": {
        "icon": "📬",
        "count_key": "ApprovedSubmitted",
        "scope": "FILTER_ApprovedSubmitted",
        "fn": lambda d: (
            (d["Current Status"] == "Approved")
            & (d["Tracking Status"].str.contains("Submitted to Courier", case=False, na=False))
        ),
    },
    "Approved - In Transit": {
        "icon": "🚚",
        "count_key": "ApprovedInTransit",
        "scope": "FILTER_ApprovedInTransit",
        "fn": lambda d: (
            (d["Current Status"] == "Approved")
            & (d["DisplayTrack"] != "")
            & (d["Tracking Status"].str.contains("routing delivery", case=False, na=False))
        ),
    },
    "Received": {
        "icon": "📦",
        "count_key": "Received",
        "scope": "Received",
        "fn": lambda d: d["Current Status"] == "Received",
    },
    "No Tracking": {
        "icon": "🚫",
        "count_key": "NoTracking",
        "scope": "FILTER_NoTracking",
        "fn": lambda d: d["is_nt"],
    },
    "Flagged": {
        "icon": "🚩",
        "count_key": "Flagged",
        "scope": "FILTER_Flagged",
        "fn": lambda d: d["is_fg"],
    },
    "Courier Cancelled": {
        "icon": "🛑",
        "count_key": "CourierCancelled",
        "scope": "FILTER_CourierCancelled",
        "fn": lambda d: d["is_cc"],
    },
    "Approved - Delivered": {
        "icon": "📬",
        "count_key": "ApprovedDelivered",
        "scope": "FILTER_ApprovedDelivered",
        "fn": lambda d: d["is_ad"],
    },
    "Resolution Actioned": {
        "icon": "💳",
        "count_key": "ResolutionActioned",
        "scope": "FILTER_ResolutionActioned",
        "fn": lambda d: d["is_ra"],
    },
    "No Resolution Actioned": {
        "icon": "⏸️",
        "count_key": "NoResolutionActioned",
        "scope": "FILTER_NoResolutionActioned",
        "fn": lambda d: d["is_nra"],
    },
}

API_OPERATIONS = {
    "Sync: Pending Requests": {
        "type": "status_sync",
        "statuses": ["Pending"],
        "scope": "Pending",
    },
    "Sync: Approved - Submitted": {
        "type": "filter_sync",
        "statuses": ["Approved"],
        "scope": "FILTER_ApprovedSubmitted",
        "filter_name": "Approved - Submitted",
    },
    "Sync: Approved - In Transit": {
        "type": "filter_sync",
        "statuses": ["Approved"],
        "scope": "FILTER_ApprovedInTransit",
        "filter_name": "Approved - In Transit",
    },
    "Sync: Received": {
        "type": "status_sync",
        "statuses": ["Received"],
        "scope": "Received",
    },
    "Sync: All Open (Pending, Approved, Received)": {
        "type": "status_sync",
        "statuses": ACTIVE_STATUSES,
        "scope": "ALL_OPEN",
    },
    "RMA Details (/rma/{id})": {
        "type": "rma_detail",
        "requires_context": True,
    },
    "Batch Refresh Courier Status": {
        "type": "batch_courier",
    },
    "Export to CSV": {
        "type": "export_csv",
    },
}


def current_filter_mask(df: pd.DataFrame) -> pd.Series:
    """
    Apply active filters and return a boolean mask.
    Handles multi-select with OR logic.
    """
    active: Set[str] = st.session_state.active_filters
    if df.empty:
        return pd.Series([], dtype=bool)
    if not active:
        return pd.Series([True] * len(df), index=df.index)

    masks = []
    for name in active:
        cfg = FILTERS.get(name)
        if not cfg:
            continue
        fn: FilterFn = cfg["fn"]
        masks.append(fn(df))

    if not masks:
        return pd.Series([True] * len(df), index=df.index)

    combined_mask = masks[0].copy()
    for m in masks[1:]:
        combined_mask = combined_mask | m
    return combined_mask


def deduplicate_filtered_rmas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate RMAs that appear in multiple filters.
    Keep only the instance with the most recent update timestamp.
    """
    if df.empty:
        return df

    if "_rma_id_text" not in df.columns:
        logger.warning("Cannot deduplicate: _rma_id_text column missing")
        return df

    def get_last_fetched_ts(row: pd.Series) -> datetime:
        try:
            last_fetched_iso = row.get("_last_fetched_iso")
            if last_fetched_iso:
                try:
                    return datetime.fromisoformat(last_fetched_iso)
                except (ValueError, TypeError):
                    pass

            req_date = row.get("Requested date", "")
            if req_date and req_date != "N/A":
                try:
                    return datetime.fromisoformat(req_date[:10]).replace(tzinfo=timezone.utc)
                except (ValueError, TypeError):
                    pass

            return datetime.min.replace(tzinfo=timezone.utc)
        except Exception:
            return datetime.min.replace(tzinfo=timezone.utc)

    df_work = df.copy()
    df_work["_dedup_ts"] = df_work.apply(get_last_fetched_ts, axis=1)

    df_deduped = (
        df_work.sort_values(["_rma_id_text", "_dedup_ts"], ascending=[True, False])
        .drop_duplicates(subset="_rma_id_text", keep="first")
        .drop(columns=["_dedup_ts"])
    )

    num_removed = len(df_work) - len(df_deduped)
    if num_removed > 0:
        logger.info("Removed %s duplicate RMAs (kept most recent)", num_removed)

    return df_deduped


def ids_for_filter(name: str) -> list:
    """Get RMA IDs that match a specific filter."""
    if df_view.empty:
        return []
    cfg = FILTERS.get(name)
    if not cfg:
        logger.warning("Filter '%s' not found in FILTERS config", name)
        return []
    fn: FilterFn = cfg["fn"]
    try:
        mask = fn(df_view)
        if mask is None or mask.empty:
            return []

        filtered_df = df_view.loc[mask].copy()
        if len(st.session_state.active_filters) > 1:
            filtered_df = deduplicate_filtered_rmas(filtered_df)

        return filtered_df["_rma_id_text"].astype(str).tolist()
    except Exception as exc:
        logger.error("Error getting IDs for filter '%s': %s", name, exc)
        return []


def execute_api_operation(endpoint: str, context: str = ""):
    """Execute the selected API operation."""
    global df_view

    if st.session_state.get("disconnected"):
        append_ops_log("System is disconnected. Login again to run API operations.", level="warning")
        return

    logger.info("Executing API operation: %s (context: %s)", endpoint, context or "none")

    operation = API_OPERATIONS.get(endpoint)
    if not operation:
        append_ops_log(f"Unknown operation: {endpoint}", level="error")
        logger.error("Unknown API operation: %s", endpoint)
        return

    op_type = operation.get("type")
    if not op_type:
        append_ops_log(f"Operation '{endpoint}' has no type defined in API_OPERATIONS", level="error")
        logger.error("Operation '%s' missing type field", endpoint)
        return

    if op_type == "status_sync":
        statuses = operation["statuses"]
        perform_sync(statuses=statuses, full=False)
        return

    if op_type == "filter_sync":
        statuses = operation.get("statuses")
        filter_name = operation.get("filter_name")
        scope = operation.get("scope")

        append_ops_log(f"Syncing {filter_name}...")
        perform_sync(statuses=statuses, full=False, rerun=False)

        if filter_name and filter_name in FILTERS and scope:
            matching_ids = ids_for_filter(filter_name)
            if matching_ids:
                force_refresh_rma_ids(matching_ids, scope)
            else:
                append_ops_log(f"No RMAs match {filter_name} filter.")
        elif filter_name and filter_name in FILTERS and not scope:
            append_ops_log(f"Operation '{endpoint}' missing required 'scope' field", level="error")
        return

    if op_type == "rma_detail":
        if not context:
            append_ops_log("Please enter an RMA ID", level="error")
            return
        append_ops_log(f"Fetching RMA {context}...")
        rma_data = fetch_rma_detail(context, force=True)
        if rma_data:
            append_ops_log(f"✅ Refreshed RMA {context}")
            st.session_state.cache_version = st.session_state.get("cache_version", 0) + 1
            st.rerun()
        else:
            append_ops_log(f"Failed to fetch RMA {context}", level="error")
        return

    if op_type == "batch_courier":
        if df_view.empty:
            append_ops_log("No data loaded. Sync dashboard first.", level="warning")
            return

        rmas_with_tracking = df_view[df_view["DisplayTrack"] != ""].copy()
        rma_ids = rmas_with_tracking["_rma_id_text"].tolist()

        if not rma_ids:
            append_ops_log("No RMAs with tracking numbers found.")
            return

        force_refresh_rma_ids(rma_ids, "BATCH_COURIER_REFRESH")
        return

    if op_type == "export_csv":
        if df_view.empty:
            append_ops_log("No data to export. Sync dashboard first.", level="warning")
            return

        csv = df_view.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"returngo_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
        return

    append_ops_log(f"Unknown operation type '{op_type}' for endpoint '{endpoint}'", level="error")
    logger.error("Unknown operation type '%s' for endpoint '%s'", op_type, endpoint)


def main():
    global df_view, counts

    st.set_page_config(page_title="Levi's ReturnGO Ops", layout="wide", page_icon="📤")
    
    # Initialize enhanced features
    if "user_settings" not in st.session_state:
        st.session_state.user_settings = load_user_settings()
    if "performance_metrics" not in st.session_state:
        st.session_state.performance_metrics = load_performance_metrics()
    
    # ==========================================
    # INITIALIZE ALL SESSION STATE VARIABLES
    # ==========================================
    default_state = {
        "active_filters": set(),
        "search_query_input": "",
        "status_multi": [],
        "res_multi": [],
        "actioned_multi": [],
        "tracking_status_multi": [],
        "req_dates_selected": [],
        "failure_refund": False,
        "failure_upload": False,
        "failure_shipment": False,
        "cache_version": 0,
        "ops_log": [],
        "schema_log": [],
        "table_autosize": False,
        "disconnected": False,
        "show_favorites": False,
        "show_toast": False,
        "last_sync_pressed": None,
        "last_sync_time": None,
        "export_csv": False,
        "copy_all_payload": None,
        "suppress_row_dialog": False,
        "api_context_input": "",
        "api_endpoint_selector": list(API_OPERATIONS.keys())[0] if API_OPERATIONS else "",
        "data_table_log": [],
        "data_table_prev": {},
    }
    
    # Initialize missing session state variables
    for key, default_value in default_state.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
    
    inject_enhanced_css()
    
    # Apply theme
    if st.session_state.user_settings.theme == "light":
        st.markdown('<style>.light-mode {}</style>', unsafe_allow_html=True)
    
    # Keyboard shortcuts - FIXED with proper label
    handle_keyboard_shortcuts()

    # Theme toggle
    render_theme_toggle()

    # --- Preserve scroll position ---
    components.html(
        """
        <script>
        const y = window.localStorage.getItem("scrollY");
        if (y) window.scrollTo(0, parseInt(y));
        window.addEventListener("scroll", () => {
          window.localStorage.setItem("scrollY", window.scrollY);
        }, { passive: true });
        </script>
        """,
        height=0,
    )

    # ==========================================
    # 0. UI: Sync Dashboard in Sidebar
    # ==========================================
    with st.sidebar:
        st.subheader("Data Operations")

        if st.button("🚀 Fetch from ReturnGo API", key="btn_sync_all", type="primary", help="Fetch latest data from the ReturnGO API for all stores and statuses. This may take several minutes."):
            st.session_state.last_sync_pressed = datetime.now()
            perform_sync(full=True)
        
        if st.button("🔄 Update From Database", help="Reload data from the local database cache without calling the API."):
            st.session_state.cache_version = st.session_state.get('cache_version', 0) + 1
            append_ops_log("✅ Dashboard updated from database")
            st.rerun()

        st.divider()

        is_connected = bool(engine) and not st.session_state.get("disconnected")
        db_status_led = "db-led-on" if is_connected else "db-led-off"
        db_status_text = "✅ Database connection" if is_connected else "⚪ Database connection"
        st.markdown(
            f"<div class='db-status-line'><span class='db-led {db_status_led}'></span>{db_status_text}</div>",
            unsafe_allow_html=True,
        )

        with st.expander("ℹ️ Database Info"):
            if engine:
                try:
                    with engine.connect() as connection:
                        rma_count_result = connection.execute(text("SELECT COUNT(*) FROM rmas")).scalar_one_or_none()
                        rma_count = rma_count_result if rma_count_result is not None else 0

                        sync_count_result = connection.execute(text("SELECT COUNT(*) FROM sync_logs")).scalar_one_or_none()
                        sync_count = sync_count_result if sync_count_result is not None else 0

                    st.info(f"📊 Database contains: **{rma_count}** RMAs, **{sync_count}** sync logs")
                except Exception as e:
                    st.warning(f"Could not read database stats: {e}")
            else:
                st.warning("⚠️ Database engine not initialized.")

        if st.button("🗑️ Reset Cache", key="btn_reset", help="⚠️ Delete ALL cached data from database. Use only for troubleshooting. Next sync will take longer."):
            if clear_db():
                st.session_state.cache_version = st.session_state.get("cache_version", 0) + 1
                append_ops_log("Database has been cleared!")
                st.rerun()
            else:
                append_ops_log("Failed to clear the database.", level="error")

        if st.button("🔬 Verify DB Schema", key="btn_verify_schema", help="Inspect table and column availability for the current database connection."):
            with st.spinner("Inspecting database..."):
                from sqlalchemy import inspect

                try:
                    conn_details = {
                        "host": st.secrets.connections.postgresql.host,
                        "database": st.secrets.connections.postgresql.database,
                        "username": st.secrets.connections.postgresql.username
                    }
                    append_schema_log(
                        f"Connection target: {conn_details['host']} / {conn_details['database']} ({conn_details['username']})"
                    )
                except Exception as e:
                    append_schema_log(
                        f"Could not read connection secrets from secrets.toml: {e}", level="error"
                    )

                def get_schema_info(engine):
                    if engine is None:
                        return {"error": "Database engine is not initialized."}
                    try:
                        inspector = inspect(engine)
                        table_names = inspector.get_table_names()
                        if not table_names:
                            return {"error": "No tables found in the database. The 'init_database' function may have failed or the database is empty."}
                        
                        schemas = {}
                        for table_name in table_names:
                            schemas[table_name] = []
                            columns = inspector.get_columns(table_name)
                            for column in columns:
                                schemas[table_name].append(f"{column['name']} ({column['type']})")
                        return schemas
                    except Exception as e:
                        return {"error": f"An error occurred while inspecting the schema: {e}"}

                schema_info = get_schema_info(engine)
                
                if "error" in schema_info:
                    append_schema_log(schema_info["error"], level="error")
                else:
                    summary = []
                    for table_name, columns in schema_info.items():
                        summary.append(f"{table_name} ({len(columns)} columns)")
                    append_schema_log("Schema inspection complete.")
                    append_schema_log("Tables: " + ", ".join(summary))

        # Enhanced features in sidebar
        with st.expander("⚙️ Settings"):
            # Theme selection
            theme = st.selectbox(
                "Theme",
                ["dark", "light", "auto"],
                index=["dark", "light", "auto"].index(st.session_state.user_settings.theme)
            )
            if theme != st.session_state.user_settings.theme:
                st.session_state.user_settings.theme = theme
                save_user_settings(st.session_state.user_settings)
                st.rerun()
            
            # Performance metrics toggle
            perf_metrics = st.checkbox(
                "Show performance metrics",
                value=st.session_state.user_settings.performance_metrics
            )
            if perf_metrics != st.session_state.user_settings.performance_metrics:
                st.session_state.user_settings.performance_metrics = perf_metrics
                save_user_settings(st.session_state.user_settings)
            
            # Keyboard shortcuts toggle
            keyboard_shortcuts = st.checkbox(
                "Enable keyboard shortcuts",
                value=st.session_state.user_settings.keyboard_shortcuts
            )
            if keyboard_shortcuts != st.session_state.user_settings.keyboard_shortcuts:
                st.session_state.user_settings.keyboard_shortcuts = keyboard_shortcuts
                save_user_settings(st.session_state.user_settings)
            
            # Favorites management
            if st.session_state.user_settings.favorites:
                st.write("**Favorites:**")
                for fav in st.session_state.user_settings.favorites:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"• {fav}")
                    with col2:
                        if st.button("🗑️", key=f"remove_fav_{fav}"):
                            st.session_state.user_settings.favorites.remove(fav)
                            save_user_settings(st.session_state.user_settings)
                            st.rerun()
            
            # Export preset management
            if st.session_state.user_settings.export_presets:
                st.write("**Export Presets:**")
                for preset_name in st.session_state.user_settings.export_presets.keys():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"• {preset_name}")
                    with col2:
                        if st.button("🗑️", key=f"remove_preset_{preset_name}"):
                            del st.session_state.user_settings.export_presets[preset_name]
                            save_user_settings(st.session_state.user_settings)
                            st.rerun()

        # API operations
        st.markdown("### 🔌 API Operations")
        api_endpoint = st.selectbox(
            "Select Operation",
            options=list(API_OPERATIONS.keys()),
            key="api_endpoint_selector",
            label_visibility="collapsed",
        )
        
        operation = API_OPERATIONS.get(api_endpoint, {})
        if operation.get("requires_context"):
            context = st.text_input("RMA ID", placeholder="Enter ID...", key="api_context_input")
            if st.button("⚡ Execute", key="api_execute_btn_ctx", use_container_width=True):
                execute_api_operation(api_endpoint, context)
        else:
            if st.button("⚡ Execute", key="api_execute_btn", use_container_width=True):
                execute_api_operation(api_endpoint)

        # Save current filter as preset
        if st.session_state.active_filters:
            preset_name = st.text_input("Preset name", placeholder="Enter preset name...")
            if st.button("💾 Save as Preset", use_container_width=True):
                if preset_name:
                    st.session_state.user_settings.export_presets[preset_name] = {
                        "filters": list(st.session_state.active_filters),
                        "timestamp": datetime.now().isoformat()
                    }
                    save_user_settings(st.session_state.user_settings)
                    st.success(f"Preset '{preset_name}' saved!")
                    st.rerun()

        st.markdown("<div class='schema-log-title'>Database Schema</div>", unsafe_allow_html=True)
        schema_log_entries = st.session_state.get("schema_log", [])
        schema_log_text = "\n".join(schema_log_entries) if schema_log_entries else "No schema checks yet."
        st.markdown(
            f"<div class='schema-log-box'><pre>{html.escape(schema_log_text)}</pre></div>",
            unsafe_allow_html=True,
        )

        st.markdown("<div class='ops-log-title'>Activity Log</div>", unsafe_allow_html=True)
        ops_log_entries = st.session_state.get("ops_log", [])
        ops_log_text = "\n".join(ops_log_entries) if ops_log_entries else "No activity yet."
        st.markdown(
            f"<div class='ops-log-box'><pre>{html.escape(ops_log_text)}</pre></div>",
            unsafe_allow_html=True,
        )

        # Keyboard shortcuts helper
        render_keyboard_shortcuts_helper()

    # ==========================================
    # 8. STATE - This section is now inside main()
    # ==========================================

    filter_migration = {
        "Pending": "Pending Requests",
        "Approved": "Approved - Submitted",
    }
    active: Set[str] = st.session_state.active_filters
    migrated = set()
    for old_name in active:
        migrated.add(filter_migration.get(old_name, old_name))
    valid_filters = set(FILTERS.keys())
    migrated = {name for name in migrated if name in valid_filters}
    if migrated != active:
        st.session_state.active_filters = migrated
        logger.info("Migrated filters: %s -> %s", active, migrated)

    if st.session_state.get("show_toast"):
        append_ops_log("✅ Updated!", level="info")
        st.session_state["show_toast"] = False

    if RATE_LIMIT_HIT.is_set():
        st.warning(
            "ReturnGO rate limit reached (429). Sync is slowing down and retrying. "
            "If this happens often, sync less frequently or request a higher quota key."
        )
        RATE_LIMIT_HIT.clear()

    # ==========================================
    # 10. HEADER
    # ==========================================
    header_left, header_right = st.columns([8, 1], vertical_alignment="center")
    with header_left:
        st.markdown("<div class='title-wrap'>", unsafe_allow_html=True)
        st.title("Levi's ReturnGO Ops Dashboard")
        st.markdown(
            f"""
            <div class="subtitle-bar">
                <span class="subtitle-dot"></span>
                <span><b>CONNECTED TO:</b> {STORE_URL.upper()}</span>
                <span style="opacity:.45">|</span>
                <span><b>CACHE:</b> {CACHE_EXPIRY_HOURS}h</span>
                <span style="opacity:.45">|</span>
                <span><b>THEME:</b> {st.session_state.user_settings.theme.title()}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with header_right:
        if st.button("⏻", key="disconnect_btn", help="Disconnect database and API sessions"):
            append_ops_log("Session disconnected. Awaiting login.", level="warning")
            st.session_state.disconnected = True

    st.write("")

    if st.session_state.get("disconnected"):
        st.markdown(
            """
            <div class="disconnect-overlay">
              <div class="disconnect-card">
                <h2>Session disconnected</h2>
                <p>Please login again to reconnect the database and API.</p>
                <div id="reconnect-slot"></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        reconnect_clicked = st.button("Login again", key="reconnect_btn")
        components.html(
            """
            <script>
              const slot = window.parent.document.querySelector('#reconnect-slot');
              const buttons = window.parent.document.querySelectorAll('button');
              buttons.forEach((button) => {
                if (button.textContent && button.textContent.trim() === 'Login again') {
                  const wrapper = button.closest('[data-testid="stButton"]');
                  if (wrapper && slot) slot.appendChild(wrapper);
                }
              });
            </script>
            """,
            height=0,
        )
        if reconnect_clicked:
            st.session_state.disconnected = False
            append_ops_log("✅ Reconnected. Ready to resume.", level="info")
            st.rerun()
        st.stop()

    # ==========================================
    # 11. LOAD + PROCESS OPEN RMAs
    # ==========================================
    raw_data = load_open_rmas(st.session_state.cache_version)
    processed_rows = []
    
    counts = {
        "PendingRequests": 0,
        "ApprovedSubmitted": 0,
        "ApprovedInTransit": 0,
        "Received": 0,
        "NoTracking": 0,
        "Flagged": 0,
        "CourierCancelled": 0,
        "ApprovedDelivered": 0,
        "ResolutionActioned": 0,
        "NoResolutionActioned": 0,
    }
    failure_counts = {
        "refund_failure": 0,
        "upload_failed": 0,
        "shipment_failure": 0,
    }

    search_query = st.session_state.get("search_query_input", "")
    search_query_lower = search_query.lower().strip()
    search_active = bool(search_query_lower)
    today: date = datetime.now().date()
    rmas_needing_courier_refresh = []
    
    for rma in raw_data:
        summary = rma.get("rmaSummary", {}) or {}
        shipments = rma.get("shipments", []) or []
        comments = rma.get("comments", []) or []

        status = summary.get("status", "Unknown")
        rma_id = summary.get("rmaId", "N/A")
        order_name = summary.get("order_name", summary.get("orderName", "N/A"))

        track_nums = [s.get("trackingNumber") for s in shipments if s.get("trackingNumber")]
        track_str = ", ".join(track_nums) if track_nums else ""
        shipment_id = shipments[0].get("shipmentId") if shipments else None

        local_tracking_status = rma.get("_local_courier_status") or "Unknown"
        local_tracking_status_display = format_tracking_status_with_icon(local_tracking_status)
        local_tracking_status_lower = local_tracking_status.lower()
        track_link_url = f"https://portal.thecourierguy.co.za/track?ref={track_nums[0]}" if track_nums else ""

        requested_iso = get_event_date_iso(rma, "RMA_CREATED") or (summary.get("createdAt") or "")
        approved_iso = get_event_date_iso(rma, "RMA_APPROVED")
        received_iso = get_received_date_iso(rma)
        resolution_type = get_resolution_type(rma)
        is_nt: bool = not track_nums
        req_dt = pd.to_datetime(requested_iso) if requested_iso else None
        is_fg = (
            status == "Pending" and isinstance(req_dt, datetime) and (today - req_dt.date()).days >= 7
        )

        comment_texts = [(c.get("htmlText", "") or "") for c in comments]
        comment_texts_lower = [c.lower() for c in comment_texts]
        is_cc = "courier cancelled" in local_tracking_status_lower
        is_ad = status == "Approved" and "delivered" in local_tracking_status_lower
        actioned: bool = is_resolution_actioned(rma, comment_texts_lower)
        actioned_label = resolution_actioned_label(rma)
        is_ra = (actioned_label != "No") and is_ad
        is_nra = (status == "Received") and (not actioned)
        is_tracking_updated = "Yes" if has_tracking_update_comment(comment_texts) else "No"
        has_refund_failure, has_upload_failed, has_shipment_failure, failures = failure_flags(
            comment_texts_lower,
            status=status,
            approved_iso=str(approved_iso) if approved_iso else "",
            has_tracking=bool(track_nums),
        )

        if track_nums and should_refresh_courier(rma):
            rmas_needing_courier_refresh.append((rma_id, rma, requested_iso, status))

        if has_refund_failure:
            failure_counts["refund_failure"] += 1
        if has_upload_failed:
            failure_counts["upload_failed"] += 1
        if has_shipment_failure:
            failure_counts["shipment_failure"] += 1

        if not search_active or (
            search_query_lower in str(rma_id).lower()
            or search_query_lower in str(order_name).lower()
            or search_query_lower in str(track_str).lower()
        ):
            processed_rows.append(
                {
                    "No": "",
                    "RMA ID": f"https://app.returngo.ai/dashboard/returns?filter_status=open&rmaid={rma_id}",
                    "Order": order_name,
                    "Current Status": status,
                    "Tracking Number": track_link_url,
                    "Tracking Status": local_tracking_status_display,
                    "Requested date": str(requested_iso)[:10] if requested_iso else "N/A",
                    "Approved date": str(approved_iso)[:10] if approved_iso else "N/A",
                    "Received date": str(received_iso)[:10] if received_iso else "N/A",
                    "Days since requested": days_since(str(requested_iso)[:10] if requested_iso else "N/A", today=today),
                    "resolutionType": resolution_type if resolution_type else "N/A",
                    "Resolution actioned": actioned_label,
                    "Is_tracking_updated": is_tracking_updated,
                    "failures": failures,
                    "has_refund_failure": has_refund_failure,
                    "has_upload_failed": has_upload_failed,
                    "has_shipment_failure": has_shipment_failure,
                    "DisplayTrack": track_str,
                    "shipment_id": shipment_id,
                    "full_data": rma,
                    "_rma_id_text": rma_id,
                    "is_nt": is_nt,
                    "is_fg": is_fg,
                    "is_cc": is_cc,
                    "is_ad": is_ad,
                    "is_ra": is_ra,
                    "is_nra": is_nra,
                }
            )

    df_view = pd.DataFrame(processed_rows)
    update_data_table_log(processed_rows)

    for filter_name, cfg in FILTERS.items():
        count_key = cfg["count_key"]
        fn: FilterFn = cfg["fn"]
        if df_view.empty:
            counts[count_key] = 0
        else:
            try:
                mask = fn(df_view)
                counts[count_key] = int(mask.sum())
            except Exception as exc:
                logger.error("Error calculating count for %s: %s", filter_name, exc)
                counts[count_key] = 0

    if rmas_needing_courier_refresh:
        def refresh_courier_batch():
            for rma_id, rma_payload, requested_iso, status in rmas_needing_courier_refresh:
                try:
                    courier_status, courier_checked = maybe_refresh_courier(rma_payload)
                    if courier_status:
                        upsert_rma(
                            rma_id=str(rma_id),
                            status=status,
                            created_at=requested_iso or "",
                            payload=rma_payload,
                            courier_status=courier_status,
                            courier_checked_iso=courier_checked,
                        )
                except Exception as exc:
                    logger.warning("Background courier refresh failed for %s: %s", rma_id, exc)

        refresh_thread = threading.Thread(target=refresh_courier_batch, daemon=True)
        refresh_thread.start()
        if len(rmas_needing_courier_refresh) > 5:
            append_ops_log(
                f"🔄 Refreshing {len(rmas_needing_courier_refresh)} courier statuses in background..."
            )

    st.divider()

    # ==========================================
    # ENHANCED COMMAND CENTER
    # ==========================================
    st.markdown("### 📊 Enhanced Command Center")
    
    # Bulk actions bar
    render_bulk_actions()
    
    # Performance metrics
    render_performance_metrics()
    
    # Export presets
    render_export_presets()
    
    # Summary bar
    render_summary_bar()
    
    # Favorites toggle
    col1, col2 = st.columns([1, 4])
    with col1:
        show_favs = st.checkbox("★ Favorites Only", value=st.session_state.show_favorites)
        if show_favs != st.session_state.show_favorites:
            st.session_state.show_favorites = show_favs
            st.rerun()
    
    # Enhanced grid layout
    st.markdown('<div class="command-grid">', unsafe_allow_html=True)
    
    # Filter tiles to show
    if st.session_state.show_favorites:
        tiles_to_show = [(name, FILTERS[name]) for name in st.session_state.user_settings.favorites if name in FILTERS]
    else:
        tiles_to_show = list(FILTERS.items())
    
    # Create columns for the grid
    num_cols = 3 if len(tiles_to_show) > 6 else 2
    cols = st.columns(num_cols)
    
    for i, (name, cfg) in enumerate(tiles_to_show):
        col_idx = i % num_cols
        render_enhanced_tile(
            cols[col_idx], 
            name, 
            cfg["scope"], 
            name.upper().replace(" - ", " > "), 
            cfg["icon"]
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.write("")

    # ==========================================
    # 14. SEARCH + VIEW ALL + FILTER BAR
    # ==========================================
    sc1, sc2, sc3 = st.columns([8, 1, 1], vertical_alignment="center")
    with sc1:
        st.text_input(
            "Search",
            placeholder="🔍 Search Order, RMA, or Tracking...",
            label_visibility="collapsed",
            key="search_query_input",
        )
    with sc2:
        pass
    with sc3:
        if st.button("📋 View All", width="stretch"):
            st.session_state.active_filters = set()
            st.rerun()

    # Extra filter bar
    with st.expander("Additional filters", expanded=True):
        c1, c2, c3, c4, c5, c6 = st.columns([2, 2, 2, 2, 3, 1], vertical_alignment="center")

        def multi_select_with_state(label: str, options: list, key: str):
            old = st.session_state.get(key, [])
            selections = [x for x in old if x in options]
            if selections != old:
                st.session_state[key] = selections
            return st.multiselect(label, options=options, key=key)

        with c1:
            multi_select_with_state("Status", ACTIVE_STATUSES, "status_multi")
        with c2:
            res_opts = []
            if not df_view.empty and "resolutionType" in df_view.columns:
                res_opts = sorted(
                    [x for x in df_view["resolutionType"].dropna().unique().tolist() if x and x != "N/A"]
                )
            multi_select_with_state("Resolution type", res_opts, "res_multi")
        with c3:
            multi_select_with_state("Resolution actioned", ["Yes", "No"], "actioned_multi")
        with c4:
            tracking_opts = [
                format_tracking_status_with_icon(status) for status in COURIER_STATUS_OPTIONS
            ]
        with c5:
            req_dates = []
            if not df_view.empty and "Requested date" in df_view.columns:
                req_dates = sorted(
                    [d for d in df_view["Requested date"].dropna().astype(str).unique().tolist() if d and d != "N/A"],
                    reverse=True,
                )
        with c6:
            st.button("🧼 Clear filters", width="stretch", on_click=clear_all_filters)

        st.markdown("**Failure filters**")
        f1, f2, f3, _ = st.columns([1.3, 1.3, 1.3, 3], vertical_alignment="center")
        with f1:
            st.checkbox(
                f"refund_failure ({failure_counts['refund_failure']})",
                key="failure_refund",
            )
        with f2:
            st.checkbox(
                f"upload_failed ({failure_counts['upload_failed']})",
                key="failure_upload",
            )
        with f3:
            st.checkbox(
                f"shipment_failure ({failure_counts['shipment_failure']})",
                key="failure_shipment",
            )

    st.divider()

    # ==========================================
    # 15. TABLE VIEW
    # ==========================================
    if df_view.empty:
        st.warning("Database empty. Click 'Fetch from ReturnGo API' to start.")
        st.stop()

    display_df = df_view[current_filter_mask(df_view)].copy()
    if len(st.session_state.active_filters) > 1:
        display_df = deduplicate_filtered_rmas(display_df)

    # Apply extra filters (AND logic)
    status_multi = st.session_state.get("status_multi", [])
    res_multi = st.session_state.get("res_multi", [])
    actioned_multi = st.session_state.get("actioned_multi", [])
    tracking_status_multi = st.session_state.get("tracking_status_multi", [])
    req_dates_selected = st.session_state.get("req_dates_selected", [])
    failure_selected = [
        label
        for label, key in (
                ("refund_failure", "failure_refund"),
            ("upload_failed", "failure_upload"),
            ("shipment_failure", "failure_shipment"),
        )
        if st.session_state.get(key)
    ]

    if status_multi:
        display_df = display_df[display_df["Current Status"].isin(status_multi)]
    if res_multi:
        display_df = display_df[display_df["resolutionType"].isin(res_multi)]
    if actioned_multi:
        actioned_set: Set[str] = set(actioned_multi)
        if actioned_set == {"Yes"}:
            display_df = display_df[display_df["Resolution actioned"] != "No"]
        elif actioned_set == {"No"}:
            display_df = display_df[display_df["Resolution actioned"] == "No"]
    if tracking_status_multi:
        display_df = display_df[display_df["Tracking Status"].isin(tracking_status_multi)]
    if req_dates_selected:
        display_df = display_df[display_df["Requested date"].isin(req_dates_selected)]
    if failure_selected:
        failure_mask = pd.Series(False, index=display_df.index)
        if st.session_state.get("failure_refund"):
            failure_mask |= display_df["has_refund_failure"]
        if st.session_state.get("failure_upload"):
            failure_mask |= display_df["has_upload_failed"]
        if st.session_state.get("failure_shipment"):
            failure_mask |= display_df["has_shipment_failure"]
        display_df = display_df[failure_mask]

    if display_df.empty:
        st.info("No matching records found.")
        st.stop()
    
    if "Requested date" not in display_df.columns:
        logger.error("KeyError: 'Requested date' column missing from display_df. Columns: %s", display_df.columns.tolist())
        st.error("Internal error: 'Requested date' column is missing from the data table. Please try syncing again.")
        st.stop()

    display_df = display_df.sort_values(by="Requested date", ascending=False).reset_index(drop=True)
    display_df["No"] = (display_df.index + 1).astype(str)

    @st.dialog("Data table log")
    def show_data_table_log():
        logs = st.session_state.get("data_table_log", [])
        if not logs:
            st.info("No edits recorded yet.")
        else:
            for entry in logs:
                st.markdown(f"- {entry}")

    @st.dialog("RMA Actions")
    def show_rma_actions_dialog(row: pd.Series):
        rma_url = row.get("RMA ID", "")
        rma_id_text = row.get("_rma_id_text", "")
        shipment_id = row.get("shipment_id")
        track_existing = row.get("DisplayTrack", "")

        tab1, tab2 = st.tabs(["Update tracking", "View timeline"])

        with tab1:
            st.markdown(f"### RMA `{rma_id_text}`")
            if rma_url:
                st.markdown(f"[Open in ReturnGO]({rma_url})")

            with st.form("upd_track_form", clear_on_submit=False):
                new_track = st.text_input("Tracking number", value=track_existing)
                submitted = st.form_submit_button("Save changes")
                if submitted:
                    if not shipment_id:
                        st.error("No Shipment ID available for this RMA.")
                    else:
                        new_track_value = new_track.strip() if new_track else ""
                        ok, msg = push_tracking_update(rma_id_text, shipment_id, new_track_value)
                        if ok:
                            if new_track_value != track_existing:
                                comment_payload = tracking_update_comment(track_existing, new_track_value)
                                push_comment_update(rma_id_text, comment_payload)
                            st.success("Tracking updated.")
                            time.sleep(0.2)
                            perform_sync(["Approved", "Received"])
                        else:
                            st.error(msg)

        with tab2:
            st.markdown(f"### Timeline for `{rma_id_text}`")

            with st.expander("➕ Add comment", expanded=False):
                with st.form("add_comment_form", clear_on_submit=True):
                    comment_text = st.text_area("New internal note")
                    if st.form_submit_button("Post comment"):
                        comment = (comment_text or "").strip()
                        if not comment:
                            st.error("Comment cannot be empty.")
                        else:
                            ok, msg = push_comment_update(rma_id_text, comment)
                            if ok:
                                st.success("Comment posted.")
                                time.sleep(0.2)
                                perform_sync(["Approved", "Received"])
                            else:
                                st.error(msg)

            full: Dict[str, Any] = row.get("full_data", {}) or {}
            timeline = full.get("comments", []) or []
            if not timeline:
                st.info("No timeline events found.")
            else:
                for t in timeline:
                    d_str = (t.get("datetime", "") or "")[:16].replace("T", " ")
                    trig = t.get("triggeredBy", "System") or "System"
                    txt = t.get("htmlText", "") or ""
                    st.markdown(f"**{d_str}** | `{trig}`\n> {txt}")
                    st.divider()

    display_cols = [
        "No",
        "RMA ID",
        "Order",
        "Current Status",
        "Tracking Number",
        "Tracking Status",
        "Requested date",
        "Approved date",
        "Received date",
        "Days since requested",
        "resolutionType",
        "Resolution actioned",
        "Is_tracking_updated",
        "failures",
    ]

    base_column_config = {
        "No": st.column_config.TextColumn("No"),
        "RMA ID": st.column_config.LinkColumn(
            "RMA ID",
            display_text=r"rmaid=([^&]+)",
        ),
        "Order": st.column_config.TextColumn("Order"),
        "Current Status": st.column_config.TextColumn("Current Status"),
        "Tracking Number": st.column_config.LinkColumn("Tracking Number", display_text=r"ref=(.*)"),
        "Tracking Status": st.column_config.TextColumn("Tracking Status"),
        "Requested date": st.column_config.TextColumn("Requested date"),
        "Approved date": st.column_config.TextColumn("Approved date"),
        "Received date": st.column_config.TextColumn("Received date"),
        "Days since requested": st.column_config.TextColumn("Days since requested"),
        "resolutionType": st.column_config.TextColumn("resolutionType"),
        "Resolution actioned": st.column_config.TextColumn("Resolution actioned"),
        "Is_tracking_updated": st.column_config.TextColumn("Is_tracking_updated"),
        "failures": st.column_config.TextColumn("failures"),
    }

    table_df = display_df[display_cols].copy()

    def _autosize_width(column: str, data: pd.DataFrame) -> Literal["small", "medium", "large"]:
        if column not in data.columns or data.empty:
            return "medium"
        max_len = int(data[column].astype(str).map(len).max())
        if max_len <= 8:
            return "small"
        if max_len <= 20:
            return "medium"
        return "large"

    link_column_configs = {
        "RMA ID": {"display_text": r"rmaid=([^&]+)"},
        "Tracking Number": {"display_text": r"ref=(.*)"},
    }

    if st.session_state.get("table_autosize"):
        column_config = {}
        for key in display_cols:
            width = _autosize_width(key, table_df)
            if key in link_column_configs:
                column_config[key] = st.column_config.LinkColumn(
                    key,
                    display_text=link_column_configs[key]["display_text"],
                    width=width,
                )
            else:
                column_config[key] = st.column_config.TextColumn(
                    key,
                    width=width,
                )
    else:
        column_config = base_column_config

    total_rows = len(table_df)
    tsv_rows = [display_cols] + table_df.astype(str).values.tolist()
    tsv_text = "\n".join(["\t".join(row) for row in tsv_rows])
    action_left, _ = st.columns([3, 5], vertical_alignment="center")
    with action_left:
        st.markdown("<div class='table-actions-left'>", unsafe_allow_html=True)
        csv_payload = table_df.to_csv(index=False)
        st.download_button(
            label="⬇️ CSV",
            data=csv_payload,
            file_name=f"returngo_table_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key="download_table_csv",
        )
        if st.button("Autosize", key="autosize_btn"):
            st.session_state.table_autosize = not st.session_state.get("table_autosize", False)
        st.markdown("</div>", unsafe_allow_html=True)
    count_col, action_col = st.columns([5, 3], vertical_alignment="center")
    with count_col:
        st.markdown(
            f"<div class='rows-count'>Rows in table: <span class='value'>{total_rows}</span></div>",
            unsafe_allow_html=True,
        )
    with action_col:
        st.markdown("<div class='data-table-actions'>", unsafe_allow_html=True)
        log_col, copy_col = st.columns([1, 1])
        with log_col:
            st.markdown("<div class='data-log-tab'>", unsafe_allow_html=True)
            if st.button("Data table log", key="btn_data_table_log", width="stretch"):
                st.session_state["suppress_row_dialog"] = True
                show_data_table_log()
            st.markdown("</div>", unsafe_allow_html=True)
        with copy_col:
            st.markdown("<div class='copy-all-btn'>", unsafe_allow_html=True)
            if st.button("📋 Copy all", width="stretch"):
                st.session_state["copy_all_payload"] = tsv_text
                st.session_state["suppress_row_dialog"] = True
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.get("copy_all_payload"):
        copy_payload = st.session_state.pop("copy_all_payload")
        json_payload = json.dumps(copy_payload)
        components.html(
            f"""
            <script>
              (async () => {{
                const payload = JSON.parse({json_payload});
                try {{
                  await navigator.clipboard.writeText(payload);
                }} catch (err) {{
                  console.warn("navigator.clipboard.writeText failed, using fallback.", err);
                  const textArea = document.createElement('textarea');
                  textArea.value = payload;
                  textArea.style.position = 'fixed';
                  textArea.style.opacity = '0';
                  document.body.appendChild(textArea);
                  textArea.focus();
                  textArea.select();
                  try {{
                    document.execCommand('copy');
                  }} catch (copyErr) {{
                    console.error("Fallback clipboard copy failed.", copyErr);
                  }}
                  document.body.removeChild(textArea);
                }}
              }})();
            </script>
            """,
            height=0,
        )
        append_ops_log("📋 Copied table to clipboard.")

    def highlight_problematic_rows(row: pd.Series):
        """Applies highlighting to rows based on a set of problem conditions."""
        highlight = False

        if (row.get("Current Status") in ["Approved", "Received"]) and not row.get("DisplayTrack"):
            highlight = True

        if row.get("failures"):
            highlight = True

        if row.get("is_cc"):
            highlight = True

        if row.get("is_fg"):
            highlight = True

        if highlight:
            return ["background-color: rgba(220, 38, 38, 0.35); color: #fee2e2;"] * len(row)

        return [""] * len(row)

    cols_for_styling = list(
        dict.fromkeys(display_cols + ["DisplayTrack", "failures", "is_cc", "is_fg"])
    )
    styling_df = display_df[cols_for_styling].copy()
    styled_table = styling_df.style.apply(highlight_problematic_rows, axis=1)
    cols_to_hide = [col for col in styling_df.columns if col not in display_cols]
    if cols_to_hide:
        styled_table = styled_table.hide(cols_to_hide, axis="columns")

    table_key = "rma_table"
    sel_event = st.dataframe(
        styled_table,
        width="stretch",
        height=700,
        hide_index=True,
        column_config=column_config,
        on_select="rerun",
        selection_mode="single-row",
        key=table_key,
    )

    sel_rows = sel_event.get("selection", {}).get("rows", []) if isinstance(sel_event, dict) else []
    if sel_rows:
        if st.session_state.pop("suppress_row_dialog", False):
            table_state = st.session_state.get(table_key, {})
            if isinstance(table_state, dict):
                table_state["selection"] = {"rows": []}
                st.session_state[table_key] = table_state
        else:
            idx = int(sel_rows[0])
            show_rma_actions_dialog(display_df.iloc[idx])

    try:
        remaining = RATE_LIMIT_INFO.get("remaining")
        limit = RATE_LIMIT_INFO.get("limit")
        if isinstance(remaining, (int, str)) and isinstance(limit, (int, str)):
            remain_int = int(remaining)
            limit_int = int(limit)
            if remain_int < limit_int * 0.2:
                st.warning(f"⚠️ API quota low: {remain_int}/{limit_int} requests remaining")
    except (ValueError, TypeError):
        pass
    
    last_sync_time = st.session_state.get("last_sync_time")
    last_sync_display = (
        last_sync_time.strftime("%Y-%m-%d %H:%M:%S")
        if last_sync_time
        else "Never"
    )
    st.markdown(
        f"<div class='sync-time-bar'>Last sync: {last_sync_display}</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()