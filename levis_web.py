import streamlit as st
import streamlit.components.v1 as components
import toml
import sqlite3
import requests
import json
import pandas as pd
import os
import threading
import time
import re
import logging
from datetime import datetime, timedelta, timezone, date
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sqlalchemy import create_engine, text
import concurrent.futures
from typing import Optional, Tuple, Dict, Callable, Set, Union, Any, Mapping
from returngo_api import api_url, RMA_COMMENT_PATH

logger = logging.getLogger(__name__)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

# Suppress the "missing ScriptRunContext" warning when running in non-Streamlit environments
# This warning is benign if the script is not intended to be run directly by `streamlit run`.
# The recommended way to run Streamlit apps is `streamlit run your_script.py`.
class NoScriptRunContextWarningFilter(logging.Filter):
    def filter(self, record):
        return not record.getMessage().startswith("Thread 'MainThread': missing ScriptRunContext!")

script_run_context_logger = logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context")
script_run_context_logger.addFilter(NoScriptRunContextWarningFilter())

# ==========================================
# 1a. CONFIGURATION
# ==========================================
def get_db_url():
    secrets = toml.load(r"c:\Users\Taswell\OneDrive\Documents\GitHub\Returngo\.streamlit\secrets.toml")
    conn = secrets["connections"]["postgresql"]
    return f"{conn['dialect']}+{conn['driver']}://{conn['username']}:{conn['password']}@{conn['host']}:{conn['port']}/{conn['database']}?sslmode={conn['sslmode']}"

engine = None
try:
    engine = create_engine(get_db_url())
except Exception as e:
    st.error(f"Failed to connect to PostgreSQL: {e}")
    st.stop()

MY_API_KEY: Optional[str] = None
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
PARCEL_NINJA_TOKEN: Optional[str] = None # Initialize to None to avoid unbound variable
df_view = pd.DataFrame()
counts: Dict[str, int] = {}

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
    # ReturnGO accepts the API key in either casing; include both for comment POST parity.
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

    with RATE_LIMIT_LOCK: # type: ignore
        RATE_LIMIT_INFO["remaining"] = to_int(remaining)
        RATE_LIMIT_INFO["limit"] = to_int(limit)
        RATE_LIMIT_INFO["reset"] = to_int(reset)
        RATE_LIMIT_INFO["updated_at"] = _now_utc()


def rg_request(method: str, url: str, *, headers=None, timeout=15, json_body=None):
    session: requests.Session = get_thread_session()
    headers = headers or rg_headers()

    backoff = 1
    res = None
    for _attempt in range(1, 6):
        _sleep_for_rate_limit()
        res = session.request(method, url, headers=headers, timeout=timeout, json=json_body)
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


# ==========================================
# 3. DATABASE
# ==========================================
def init_db():
    if engine is None:
        st.error("Database engine is not initialized.")
        st.stop()
    
    create_rmas_table_query = text("""
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
    """)

    create_sync_logs_table_query = text("""
    CREATE TABLE IF NOT EXISTS sync_logs (
        scope TEXT PRIMARY KEY,
        last_sync_iso TIMESTAMP WITH TIME ZONE
    );
    """)

    try:
        with engine.connect() as connection:
            connection.execute(create_rmas_table_query)
            connection.execute(create_sync_logs_table_query)
            connection.commit()
    except Exception as e:
        st.error(f"Error creating tables: {e}")
        st.stop()





def upsert_rma(
    rma_id: str,
    status: str,
    created_at: str,
    payload: dict,
    courier_status: Optional[str] = None,
    courier_checked_iso: Optional[str] = None,
):
    if engine is None:
        return

    now_iso = _iso_utc(_now_utc())
    
    # Convert created_at to a timezone-aware datetime object
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

    with engine.connect() as connection:
        # First, select the existing values
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
            "json_data": json.dumps(payload),
            "last_fetched": now_iso,
            "courier_status": courier_status,
            "courier_last_checked": courier_checked_dt,
            "received_first_seen": received_seen
        })
        connection.commit()


def delete_rmas(rma_ids):
    if not rma_ids: # type: ignore
        return
    if isinstance(rma_ids, (str, bytes)): # type: ignore
        rma_ids = [rma_ids] # type: ignore

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
    
    if engine is None: # type: ignore
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
    if engine is None: # type: ignore
        return None

    with engine.connect() as connection:
        query = text("""
            SELECT json_data, last_fetched, courier_status, courier_last_checked, received_first_seen 
            FROM rmas WHERE rma_id=:rma_id
        """)
        row = connection.execute(query, {"rma_id": str(rma_id)}).fetchone()

    if not row:
        return None

    payload = json.loads(row[0])
    payload["_local_courier_status"] = row[2]
    payload["_local_courier_checked"] = row[3].isoformat() if row[3] else None
    payload["_local_received_first_seen"] = row[4].isoformat() if row[4] else None
    return payload, row[1].isoformat() if row[1] else None


def get_all_open_from_db():
    if engine is None: # type: ignore
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
        data = json.loads(js)
        data["_local_courier_status"] = cstat
        data["_local_courier_checked"] = cchk.isoformat() if cchk else None
        data["_local_received_first_seen"] = rcv_seen.isoformat() if rcv_seen else None
        results.append(data)
    return results


@st.cache_data(show_spinner=False, ttl=60)
def load_open_rmas(_cache_version: int):
    """Load all open RMAs from the database, cached for 60 seconds."""
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
                # The value from postgres is already a datetime object
                return row[0]
            except Exception:
                return None
        return None

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
        base_url = f"{api_url('/rmas')}?pagesize=500&status={status_param}{updated_filter}" # type: ignore
        url = f"{base_url}&cursor={cursor}" if cursor else base_url

        res = rg_request("GET", url, timeout=20)
        if res is None:
            last_error = "No response from ReturnGO"
            break
        if res.status_code != 200:
            err_text = (res.text or "").strip()
            if len(err_text) > 200:
                err_text = f"{err_text[:200]}..." # type: ignore
            last_error = f"{res.status_code}: {err_text}" if err_text else str(res.status_code)
            break

        had_success = True
        try:
            data = res.json() if res.content else {}
        except ValueError:
            last_error = "Invalid JSON response from ReturnGO" # type: ignore
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
    try:
        last_dt = datetime.fromisoformat(last_fetched_iso)
    except Exception:
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
    except Exception:
        return True


# This function would now primarily orchestrate fetching and caching of courier status
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
    
    # Use the cached check_courier_status directly
    status = check_courier_status(track_no) # This call is now cached by st.cache_data
    
    # If the status was just refreshed (i.e., cache miss for check_courier_status), update checked_iso
    # Otherwise, use the existing cached_checked
    if status != cached_status or cached_checked is None: # Simple heuristic to detect refresh
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
    # type: ignore
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
        # refresh local fields for UI
        fresh = get_rma(str(rma_id))
        if fresh:
            data["_local_received_first_seen"] = fresh[0].get("_local_received_first_seen")
        data["_local_courier_status"] = courier_status
        data["_local_courier_checked"] = courier_checked
        return data
    except Exception as e:
        logger.error("Error upserting RMA %s to DB: %s", rma_id, e)
        return cached[0] if cached else None # Return cached if upsert fails


def get_incremental_since(statuses, full: bool) -> Optional[datetime]:
    if full:
        return None
    stamps = [get_last_sync(s) for s in statuses]
    stamps = [d for d in stamps if d]
    return min(stamps) if stamps else None


def perform_sync(statuses=None, *, full=False, rerun: bool = True):
    status_msg = st.empty()
    status_msg.info("⏳ Connecting to ReturnGO...")

    if statuses is None:
        statuses = ACTIVE_STATUSES

    since_dt = get_incremental_since(statuses, full)

    list_bar = st.progress(0, text="Fetching RMA list from ReturnGO...")
    summaries, ok, err = fetch_rma_list(statuses, since_dt)
    if not ok:
        list_bar.empty()
        msg = f"ReturnGO sync failed: {err}" if err else "ReturnGO sync failed."
        status_msg.error(msg)
        return
    list_bar.progress(1.0, text=f"Fetched {len(summaries)} RMAs")
    time.sleep(0.08)
    list_bar.empty()

    api_ids = {s.get("rmaId") for s in summaries if s.get("rmaId")}

    # tidy cache only for open statuses scope
    if set(statuses) == set(ACTIVE_STATUSES) and not full and since_dt is not None:
        local_active_ids = set()
        for stt in ACTIVE_STATUSES:
            local_active_ids |= get_local_ids_for_status(stt) # type: ignore
        stale = local_active_ids - api_ids # type: ignore
        logger.info("Removing %s stale RMAs no longer in active statuses", len(stale))
        delete_rmas(stale)
    elif full:
        if engine is None:
            return
        with engine.connect() as connection:
            query = text("SELECT rma_id FROM rmas WHERE store_url=:store_url")
            result = connection.execute(query, {"store_url": STORE_URL})
            all_local_ids = {r[0] for r in result.fetchall()}
        stale_full = all_local_ids - api_ids # type: ignore
        if stale_full:
            logger.info("Full sync: removing %s RMAs not in API response", len(stale_full))
            delete_rmas(stale_full)

    # If incremental, force-refresh details for returned IDs (they changed)
    if since_dt is not None:
        to_fetch = list(api_ids)
        force = True
    else:
        to_fetch = [rid for rid in api_ids if should_refresh_detail(rid)]
        force = False

    total = len(to_fetch)
    status_msg.info(f"⏳ Syncing {total} records...")

    if total > 0:
        successful_fetches = 0
        failed_fetches = 0
        bar = st.progress(0, text="Downloading Details...") # This progress bar is for the loop, not the cached function
        bar = st.progress(0, text="Downloading Details...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex: # type: ignore
            futures = [ex.submit(fetch_rma_detail, rid, force=force) for rid in to_fetch]
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                done = i + 1
                bar.progress(done / total, text=f"Syncing: {done}/{total}")
                # Get the result (or exception)
                result = future.result()
                if result is not None: successful_fetches += 1
                else: failed_fetches += 1
        bar.empty()

    now = _now_utc()
    scope = ",".join(statuses)
    set_last_sync(scope, now)
    for s in statuses:
        set_last_sync(s, now)

    if failed_fetches == 0:
        status_msg.success(f"✅ Sync Complete! {successful_fetches} RMAs updated.")
    else:
        status_msg.warning(f"⚠️ Sync Complete with {successful_fetches} RMAs updated and {failed_fetches} failed.")
    st.session_state["show_toast"] = True
    status_msg.success("✅ Sync Complete!")
    try:
        load_open_rmas.clear()
    except Exception:
        st.error("Failed to clear open-RMA cache")
    if rerun:
        st.rerun()
 # type: ignore

def force_refresh_rma_ids(rma_ids, scope_label: str):
    ids = [str(i) for i in set(rma_ids) if i]
    if not ids: # type: ignore
        set_last_sync(scope_label, _now_utc())
        st.session_state["show_toast"] = True
        st.rerun()
        # Note: st.rerun() raises a RerunException and normally prevents execution
        # from reaching this return; it exists to satisfy type checkers / static analyzers.
        return

    msg = st.empty()
    msg.info(f"⏳ Refreshing {len(ids)} records...")

    bar = st.progress(0, text="Downloading Details...")
    total = len(ids)
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(fetch_rma_detail, rid, force=True) for rid in ids]
        done = 0
        for _ in concurrent.futures.as_completed(futures):
            done += 1
            bar.progress(done / total, text=f"Refreshing: {done}/{total}")
    bar.empty()

    set_last_sync(scope_label, _now_utc())
    st.session_state.cache_version = st.session_state.get("cache_version", 0) + 1
    st.session_state["show_toast"] = True
    msg.success("✅ Refresh Complete!")
    try:
        load_open_rmas.clear()
    except Exception:
        st.error("Failed to clear open-RMA cache")
    st.rerun()
 # type: ignore
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


def format_api_limit_display() -> Tuple[str, str]: # type: ignore
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
            sub = f"Reset: {reset}" # type: ignore
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
    s: Set[str] = set(st.session_state.active_filters)
    if name in s:
        s.remove(name)
        logger.info("Filter deselected: %s", name)
    else:
        s.add(name)
        logger.info("Filter selected: %s", name)
    st.session_state.active_filters = s  # type: ignore
    st.rerun() # type: ignore


def clear_all_filters():
    st.session_state.active_filters = set()  # type: ignore
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

    prev_map: Dict[str, Dict[str, str]] = st.session_state.data_table_prev # type: ignore
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
    st.session_state.data_table_prev = {**prev_map, **current_map} # type: ignore


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


def inject_command_center_css():
    st.markdown(
        """
        <style>
        /* ═══════════════════════════════════════════════════════════
           COMMAND CENTER CORE THEME - REFINED
           ═══════════════════════════════════════════════════════════ */

        :root {
            --cc-primary: #3b82f6;
            --cc-success: #22c55e;
            --cc-warning: #f59e0b;
            --cc-danger: #ef4444;
            --cc-surface: rgba(15, 23, 42, 0.35);
            --cc-surface-hover: rgba(31, 41, 55, 0.6);
            --cc-border-neutral: rgba(148, 163, 184, 0.15);
            --cc-border-active: #22c55e;
            --cc-glow: rgba(34, 197, 94, 0.4);
            --cc-text-primary: #f1f5f9;
            --cc-text-secondary: #94a3b8;
        }

        /* ═══════════════════════════════════════════════════════════
           GLASSMORPHIC TILE SYSTEM - NO PILLS
           ═══════════════════════════════════════════════════════════ */

        .command-tile {
            position: relative;
            background: linear-gradient(
                135deg,
                var(--cc-surface) 0%,
                rgba(30, 41, 59, 0.4) 100%
            );
            backdrop-filter: blur(20px) saturate(180%);
            -webkit-backdrop-filter: blur(20px) saturate(180%);
            border: 1px solid var(--cc-border-neutral);
            border-radius: 16px;
            padding: 24px 20px;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow:
                0 4px 6px -1px rgba(0, 0, 0, 0.1),
                0 2px 4px -1px rgba(0, 0, 0, 0.06),
                inset 0 1px 0 0 rgba(255, 255, 255, 0.05);
            overflow: hidden;
        }

        /* Selected tiles keep neutral borders (highlight only on action button) */
        .command-tile.selected {
            border-color: var(--cc-border-neutral);
            box-shadow:
                0 4px 6px -1px rgba(0, 0, 0, 0.1),
                0 2px 4px -1px rgba(0, 0, 0, 0.06),
                inset 0 1px 0 0 rgba(255, 255, 255, 0.05);
        }

        .command-tile:hover {
            transform: translateY(-2px);
            box-shadow:
                0 10px 15px -3px rgba(0, 0, 0, 0.2),
                0 4px 6px -2px rgba(0, 0, 0, 0.05);
        }

        /* Remove top accent bar */
        .command-tile::before {
            display: none;
        }

        /* Pulse animation for urgent tiles only */
        @keyframes pulse-glow {
            0%, 100% {
                box-shadow:
                    0 4px 6px -1px rgba(0, 0, 0, 0.1),
                    0 0 0 1px var(--cc-border-active),
                    0 0 20px -5px var(--cc-glow);
            }
            50% {
                box-shadow:
                    0 4px 6px -1px rgba(0, 0, 0, 0.1),
                    0 0 0 1px var(--cc-border-active),
                    0 0 30px -5px var(--cc-glow);
            }
        }

        .command-tile.urgent.selected {
            animation: none;
        }

        /* ═══════════════════════════════════════════════════════════
           HERO COUNT & TYPOGRAPHY (No color pills)
           ═══════════════════════════════════════════════════════════ */

        .tile-count {
            font-size: 3rem;
            font-weight: 800;
            line-height: 1;
            color: var(--cc-text-primary);
            margin: 0;
            font-variant-numeric: tabular-nums;
            letter-spacing: -0.02em;
            text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
        }

        .tile-label {
            font-size: 0.875rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--cc-text-secondary);
            margin: 0 0 16px 0;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .tile-label-icon {
            font-size: 1rem;
            opacity: 0.7;
        }

        .tile-divider {
            width: 100%;
            height: 1px;
            background: linear-gradient(
                90deg,
                transparent,
                rgba(148, 163, 184, 0.2) 50%,
                transparent
            );
            margin: 16px 0 12px 0;
        }

        /* ═══════════════════════════════════════════════════════════
           VIEW DETAILS BUTTON - GREEN BORDER WHEN SELECTED
           ═══════════════════════════════════════════════════════════ */

        .command-tile div[data-testid="stButton"] > button {
            all: unset;
            width: 100%;
            padding: 10px 16px;
            background: rgba(255, 255, 255, 0.03);
            border: 2px solid var(--cc-border-neutral);
            border-radius: 8px;
            color: var(--cc-text-primary);
            font-size: 0.8rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            text-align: center;
            display: block;
            box-sizing: border-box;
        }

        .command-tile div[data-testid="stButton"] > button:focus,
        .command-tile div[data-testid="stButton"] > button:focus-visible,
        .command-tile div[data-testid="stButton"] > button:active {
            outline: none;
            border-color: var(--cc-border-active);
            box-shadow: 0 0 0 3px rgba(34, 197, 94, 0.15);
        }

        /* Hover state - green border preview */
        .command-tile div[data-testid="stButton"] > button:hover {
            background: rgba(34, 197, 94, 0.08);
            border-color: var(--cc-border-active);
            transform: translateY(-1px);
            box-shadow: 0 0 0 3px rgba(34, 197, 94, 0.15);
        }

        /* Selected state - constant green border */
        .command-tile.selected div[data-testid="stButton"] > button {
            border-color: var(--cc-border-active);
            background: rgba(34, 197, 94, 0.12);
            box-shadow: 0 0 0 3px rgba(34, 197, 94, 0.15);
        }

        .command-tile.selected div[data-testid="stButton"] > button:hover {
            background: rgba(34, 197, 94, 0.18);
            box-shadow: 0 0 0 4px rgba(34, 197, 94, 0.2);
        }

        /* ═══════════════════════════════════════════════════════════
           REFRESH BUTTON - INLINE WITH COUNT
           ═══════════════════════════════════════════════════════════ */

        .tile-header-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 12px;
            margin-bottom: 4px;
        }

        .tile-count-wrapper {
            flex: 0 0 50%;
            min-width: 0;
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .tile-refresh-btn {
            flex: 0 0 50%;
            display: flex;
            justify-content: flex-end;
        }

        .tile-refresh-btn div[data-testid="stButton"] {
            margin: 0 !important;
        }

        .tile-refresh-btn button {
            all: unset;
            width: 32px;
            height: 32px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--cc-border-neutral);
            border-radius: 6px;
            color: var(--cc-text-secondary);
            cursor: pointer;
            transition: all 0.2s ease;
            font-size: 1rem;
        }

        .command-tile.selected .tile-refresh-btn div[data-testid="stButton"] > button {
            border-color: var(--cc-border-neutral);
            background: rgba(255, 255, 255, 0.03);
            box-shadow: none;
        }

        .tile-refresh-btn button:hover {
            background: rgba(255, 255, 255, 0.08);
            border-color: var(--cc-primary);
            color: var(--cc-text-primary);
            transform: rotate(90deg);
        }

        .tile-time {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 0.8rem;
            color: var(--cc-text-secondary);
            white-space: nowrap;
        }

        .tile-status-dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: var(--cc-success);
            animation: pulse-dot 2s ease-in-out infinite;
        }

        @keyframes pulse-dot {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }

        /* ═══════════════════════════════════════════════════════════
           RESPONSIVE ADJUSTMENTS
           ═══════════════════════════════════════════════════════════ */

        @media (max-width: 768px) {
            .command-tile {
                padding: 20px 16px;
            }

            .tile-count {
                font-size: 2.5rem;
            }

            .tile-header-row {
                flex-direction: column;
            }

            .tile-count-wrapper,
            .tile-refresh-btn {
                flex: 0 0 100%;
                width: 100%;
            }

            .tile-refresh-btn {
                margin-top: 8px;
                align-self: flex-end;
            }
        }

        /* ═══════════════════════════════════════════════════════════
           STREAMLIT OVERRIDES
           ═══════════════════════════════════════════════════════════ */

        .command-tile [data-testid="column"] {
            padding: 0 !important;
        }

        .command-tile .stMarkdown {
            margin: 0 !important;
        }

        .command-tile .element-container {
            margin: 0 !important;
        }

        /* ═══════════════════════════════════════════════════════════
           API ACTION BAR
           ═══════════════════════════════════════════════════════════ */

        .api-action-bar {
            background: var(--cc-surface);
            backdrop-filter: blur(20px);
            border: 1px solid var(--cc-border-neutral);
            border-radius: 12px;
            padding: 20px 24px;
            margin: 24px 0;
            box-shadow:
                0 4px 6px -1px rgba(0, 0, 0, 0.1),
                inset 0 1px 0 0 rgba(255, 255, 255, 0.05);
        }

        .api-action-bar-title {
            font-size: 0.7rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--cc-text-secondary);
            margin-bottom: 12px;
        }

        .api-action-bar.compact {
            padding: 12px 14px;
            margin: 12px 0 0 0;
        }

        .api-action-bar.compact .api-action-bar-title {
            font-size: 0.6rem;
            margin-bottom: 8px;
        }

        .api-action-bar.compact button[kind="primary"] {
            padding: 8px 14px !important;
            font-size: 0.75rem !important;
        }

        .api-action-bar [data-baseweb="select"] {
            background: rgba(15, 23, 42, 0.8) !important;
            border: 1px solid var(--cc-border-neutral) !important;
            border-radius: 8px !important;
        }

        .api-action-bar [data-baseweb="select"]:hover {
            border-color: var(--cc-primary) !important;
            box-shadow: 0 0 0 1px var(--cc-primary) !important;
        }

        .api-action-bar button[kind="primary"] {
            background: linear-gradient(135deg, var(--cc-primary), #2563eb) !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 10px 24px !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3) !important;
            transition: all 0.2s ease !important;
        }

        .api-action-bar button[kind="primary"]:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4) !important;
        }

        /* ═══════════════════════════════════════════════════════════
           UTILITIES
           ═══════════════════════════════════════════════════════════ */

        .fade-in {
            animation: fadeIn 0.6s ease-out;
        }

        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
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
    """ # type: ignore
    active: Set[str] = st.session_state.active_filters  # type: ignore
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

    Args:
        df: Filtered DataFrame (already masked by current_filter_mask)

    Returns:
        DataFrame with duplicates removed based on most recent 'last_fetched'
    """
    if df.empty:
        return df

    if "_rma_id_text" not in df.columns:
        logger.warning("Cannot deduplicate: _rma_id_text column missing")
        return df

    def get_last_fetched_ts(row: pd.Series) -> datetime:
        try:
            # Prioritize the pre-fetched timestamp from the DataFrame
            last_fetched_iso = row.get("_last_fetched_iso")
            if last_fetched_iso:
                try:
                    return datetime.fromisoformat(last_fetched_iso)
                except (ValueError, TypeError):
                    pass

            # Fallback to requested date if the primary timestamp is missing
            req_date = row.get("Requested date", "")
            if req_date and req_date != "N/A":
                try:
                    # Parse only the date part and make it timezone-aware for safe comparison
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


# ==========================================
# 13. COMMAND CENTER TILE UI
# ==========================================
def render_command_tile(
    col,
    name: str,
    refresh_scope: str,
    semantic_label: str,
    icon: str,
):
    """
    Render a glassmorphic command center tile with inline refresh button.

    Args:
        col: Streamlit column to render into
        name: Filter name (e.g., "Pending Requests")
        refresh_scope: Scope key for sync operations
        semantic_label: Human-readable action label (e.g., "PENDING REQUESTS")
        icon: Emoji icon
    """
    cfg = FILTERS.get(name)
    if not cfg:
        st.error(f"Filter '{name}' not found in FILTERS config")
        return

    count_key = cfg["count_key"]

    active: Set[str] = st.session_state.active_filters  # type: ignore
    is_selected = name in active
    is_urgent = name == "Flagged"

    count_val = counts.get(count_key, 0)
    last_sync_ts = get_last_sync(refresh_scope)
    time_ago_str = format_time_ago(last_sync_ts)

    with col:
        tile_classes = ["command-tile", "fade-in"]
        if is_selected:
            tile_classes.append("selected")
        if is_urgent and is_selected:
            tile_classes.append("urgent")

        st.markdown(f"<div class='{' '.join(tile_classes)}'>", unsafe_allow_html=True)

        st.markdown("<div class='tile-header-row'>", unsafe_allow_html=True)
        st.markdown("<div class='tile-count-wrapper'>", unsafe_allow_html=True)
        st.markdown(f"<div class='tile-count'>{count_val}</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='tile-time'>"
            f"<span class='tile-status-dot'></span>"
            f"<span>Updated {time_ago_str}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='tile-refresh-btn'>", unsafe_allow_html=True)
        if st.button("⟳", key=f"ref_{name}", help=f"Refresh {semantic_label}"):
            force_refresh_rma_ids(ids_for_filter(name), refresh_scope)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            f"<div class='tile-label'>"
            f"<span class='tile-label-icon'>{icon}</span>"
            f"<span>{semantic_label}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        button_label = "✓ VIEWING" if is_selected else "VIEW DETAILS"
        if st.button(
            button_label,
            key=f"tile_{name}",
            use_container_width=True,
        ):
            toggle_filter(name) # type: ignore

        st.markdown("<div class='tile-divider'></div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


def render_api_action_bar(*, compact: bool = False):
    """Render the centralized API operation control panel."""
    wrapper_class = "api-action-bar compact" if compact else "api-action-bar"
    st.markdown(f"<div class='{wrapper_class}'>", unsafe_allow_html=True)
    st.markdown("<div class='api-action-bar-title'>🔌 API Operations</div>", unsafe_allow_html=True)

    column_ratio = [2, 1] if compact else [3, 1]
    col1, col2 = st.columns(column_ratio)

    with col1:
        api_endpoint = st.selectbox(
            "Select Operation",
            options=list(API_OPERATIONS.keys()),
            key="api_endpoint_selector", # type: ignore
            label_visibility="collapsed",
        )

    with col2:
        operation = API_OPERATIONS.get(api_endpoint, {})
        if operation.get("requires_context"):
            st.text_input(
                "RMA ID",
                key="api_context_input", # type: ignore
                placeholder="Enter ID...",
                label_visibility="collapsed",
            )
        else:
            if st.button("⚡ Execute", key="api_execute_btn", type="primary", use_container_width=True):
                execute_api_operation(api_endpoint, "")
                return

    if operation.get("requires_context"):
        if st.button("⚡ Execute", key="api_execute_btn_ctx", type="primary", use_container_width=True): # type: ignore
            execute_api_operation(api_endpoint, st.session_state.get("api_context_input", ""))

    st.markdown("</div>", unsafe_allow_html=True)


def execute_api_operation(endpoint: str, context: str = ""):
    """Execute the selected API operation."""
    global df_view

    logger.info("Executing API operation: %s (context: %s)", endpoint, context or "none")

    operation = API_OPERATIONS.get(endpoint)
    if not operation:
        st.error(f"Unknown operation: {endpoint}")
        logger.error("Unknown API operation: %s", endpoint)
        return

    op_type = operation.get("type")
    if not op_type:
        st.error(f"Operation '{endpoint}' has no type defined in API_OPERATIONS")
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

        with st.spinner(f"Syncing {filter_name}..."):
            perform_sync(statuses=statuses, full=False, rerun=False)

            if filter_name and filter_name in FILTERS and scope:
                matching_ids = ids_for_filter(filter_name)
                if matching_ids:
                    force_refresh_rma_ids(matching_ids, scope)
                else:
                    st.info(f"No RMAs match {filter_name} filter.")
            elif filter_name and filter_name in FILTERS and not scope:
                st.error(f"Operation '{endpoint}' missing required 'scope' field")
        return

    if op_type == "rma_detail":
        if not context: # type: ignore
            st.error("Please enter an RMA ID")
            return
        with st.spinner(f"Fetching RMA {context}..."): # type: ignore
            rma_data = fetch_rma_detail(context, force=True)
        if rma_data:
            st.success(f"✅ Refreshed RMA {context}")
            st.session_state.cache_version = st.session_state.get("cache_version", 0) + 1
            st.rerun()
        else:
            st.error(f"Failed to fetch RMA {context}")
        return

    if op_type == "batch_courier":
        if df_view.empty:
            st.warning("No data loaded. Sync dashboard first.")
            return

        rmas_with_tracking = df_view[df_view["DisplayTrack"] != ""].copy()
        rma_ids = rmas_with_tracking["_rma_id_text"].tolist()

        if not rma_ids:
            st.info("No RMAs with tracking numbers found.")
            return

        force_refresh_rma_ids(rma_ids, "BATCH_COURIER_REFRESH") # type: ignore
        return

    if op_type == "export_csv":
        if df_view.empty:
            st.warning("No data to export. Sync dashboard first.")
            return

        csv = df_view.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"returngo_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
        return

    st.error(f"Unknown operation type '{op_type}' for endpoint '{endpoint}'")
    logger.error("Unknown operation type '%s' for endpoint '%s'", op_type, endpoint)


def main():
    global MY_API_KEY, PARCEL_NINJA_TOKEN, df_view, counts

    st.set_page_config(page_title="Levi's ReturnGO Ops", layout="wide", page_icon="📤")
    inject_command_center_css() # This function is defined later in the file, consider moving it up or ensuring it's called after definition.

    # --- Preserve scroll position across reruns (page scroll) ---
    components.html( # type: ignore
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

    # --- ACCESS SECRETS ---
    try:
        MY_API_KEY = st.secrets["RETURNGO_API_KEY"]
    except (FileNotFoundError, KeyError, AttributeError):
        MY_API_KEY = None

    if not MY_API_KEY:
        st.error("API Key not found! Please set 'RETURNGO_API_KEY' in your Streamlit secrets (`.streamlit/secrets.toml`).")
        st.stop()

    try:
        PARCEL_NINJA_TOKEN = st.secrets["PARCEL_NINJA_TOKEN"]
    except Exception:
        PARCEL_NINJA_TOKEN = os.environ.get("PARCEL_NINJA_TOKEN", PARCEL_NINJA_TOKEN)

    # ==========================================
    # 1b. STYLING + STICKY HEADER
    # ==========================================
    st.markdown(
        """
        <style>
          .stApp {
            background: radial-gradient(1200px 600px at 20% 0%, rgba(196,18,48,0.08), transparent 60%),
                        radial-gradient(900px 500px at 90% 10%, rgba(59,130,246,0.08), transparent 55%),
                        #0e1117;
            color: #e5e7eb;
          }

          /* Title */
          .title-wrap h1 {
            font-size: 3.1rem;
            margin-bottom: 0.15rem;
            letter-spacing: 0.2px;
            font-family: "Inter", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
            font-weight: 700;
            background: linear-gradient(90deg, rgba(255,255,255,0.92), rgba(255,255,255,0.78));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 12px 30px rgba(15, 23, 42, 0.45);
          }

          .subtitle-bar {
            display: inline-flex;
            gap: 10px;
            align-items: center;
            padding: 8px 14px;
            border-radius: 999px;
            border: 1px solid rgba(148, 163, 184, 0.22);
            background: rgba(17, 24, 39, 0.55);
            box-shadow: 0 8px 20px rgba(0,0,0,0.20);
            font-size: 1.05rem;
            color: #cbd5e1;
          }
          .subtitle-bar b { color: #e5e7eb; }
          .subtitle-dot {
            width: 7px; height: 7px;
            border-radius: 50%;
            background: rgba(196,18,48,0.9);
            display: inline-block;
          }

          .block-container {
            background: transparent !important;
          }

          /* Cards */
          .card {
            position: relative;
            background: transparent;
            border: none;
            border-radius: 14px;
            padding: 0;
            box-shadow: none;
            display: flex;
            flex-direction: column;
          }

          .card.selected {
            background: transparent;
          }

          .tile-inner {
            position: relative;
            flex: 1;
          }

          .status-tile {
            width: 70%;
            margin: 0 auto;
            display: flex;
            flex-direction: column;
            gap: 0;
          }
          /* Remove Streamlit block spacing between count, button, and refresh rows. */
          .status-tile .stMarkdown,
          .status-tile div[data-testid="stButton"] {
            margin-top: 0 !important;
            margin-bottom: 0 !important;
          }
          .status-tile > div {
            margin-top: 0 !important;
            margin-bottom: 0 !important;
          }
          .status-tile div[data-testid="stButton"] {
            padding: 0 !important;
            border: none !important;
            background: transparent !important;
            box-shadow: none !important;
          }

          .count-banner {
            width: 100%;
            height: 40px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-size: 1.35rem;
            color: #ffffff;
            background: rgba(148, 163, 184, 0.22);
            border: 1px solid rgba(148, 163, 184, 0.25);
            margin-bottom: 0;
          }

          .card.selected .count-banner {
            background: rgba(34, 197, 94, 0.18);
            border-color: rgba(34, 197, 94, 0.4);
            color: #22c55e;
          }

          .status-button {
            position: relative;
            padding-top: 0;
            margin-bottom: 0;
          }
          .status-button div.stButton {
            margin-bottom: 0 !important;
          }
          .status-button div.stButton > button {
            width: 100% !important;
            background: rgba(15, 23, 42, 0.82) !important;
            border: 1px solid rgba(148, 163, 184, 0.25) !important;
            color: #e5e7eb !important;
            box-shadow: none !important;
            border-radius: 10px !important;
            padding: 8px 12px !important;
            text-transform: uppercase !important;
            margin: 0 !important;
            white-space: normal !important;
            transition: all 0.2s ease !important;
          }

          .card.selected .status-button div.stButton > button {
            border: 2.5px solid #22c55e !important;
            background: rgba(34, 197, 94, 0.12) !important;
            box-shadow: 0 0 0 3px rgba(34, 197, 94, 0.15) !important;
          }

          .status-button div.stButton > button:hover {
            border-color: rgba(148, 163, 184, 0.45) !important;
            transform: translateY(-1px);
          }

          .card.selected .status-button div.stButton > button:hover {
            border-color: #22c55e !important;
          }

          /* Refresh row under each tile */
          .refresh-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            width: 100%;
            /* Pull the Refresh row flush under the status button. */
            margin-top: -6px;
            padding-top: 0;
          }
          .refresh-row div.stButton {
            margin: 0 !important;
          }
          .refresh-row div[data-testid="stButton"] {
            padding: 0 !important;
            border: none !important;
            background: transparent !important;
            box-shadow: none !important;
          }
          .refresh-row div.stButton > button {
            width: auto !important;
            padding: 0 !important;
            border: none !important;
            background: transparent !important;
            box-shadow: none !important;
            font-size: 13px !important;
            font-style: italic !important;
            color: #ffffff !important;
            text-decoration: underline !important;
            line-height: 1 !important;
            margin: 0 !important;
          }
          .refresh-row div.stButton > button:hover {
            color: #e2e8f0 !important;
            transform: none !important;
          }
          .updated-time {
            font-size: 12px;
            color: rgba(226, 232, 240, 0.9);
            white-space: nowrap;
          }

          /* Sticky container */
          .sticky-top {
            position: sticky !important;
            top: 0 !important;
            z-index: 999 !important;
            background: rgba(14,17,23,0.82) !important;
            backdrop-filter: blur(10px);
            border-bottom: 1px solid rgba(148,163,184,0.16);
            padding-bottom: 10px;
            margin-bottom: 8px;
          }

          /* Data editor: attempt “fit content” */
          [data-testid="stDataFrame"] {
            border-radius: 14px;
            overflow-x: auto;
            border: 1px solid rgba(148,163,184,0.18);
          }
          [data-testid="stDataFrame"] table {
            table-layout: auto !important;
          }

          /* Dialog */
          div[data-testid="stDialog"] {
            background-color: rgba(17, 24, 39, 0.95);
            border: 1px solid rgba(196,18,48,0.55);
            border-radius: 16px;
            width: min(96vw, 1200px) !important;
            max-width: 96vw !important;
          }

          /* Smaller Reset Cache button look (lighter) */
          .reset-wrap {
            position: relative;
          }
          .reset-wrap div.stButton > button {
            background: rgba(148,163,184,0.18) !important;
            border-color: rgba(148,163,184,0.25) !important;
            padding: 4px 8px !important;
            font-size: 12px !important;
            width: 50% !important;
          }
          .sync-dashboard-btn button {
            min-height: 4rem !important;
            padding: 0.9rem 1.2rem !important;
            text-transform: uppercase !important;
          }
          .sync-time-bar {
            text-align: right;
            margin: 0 0 -8px 0;
            font-size: 1.025rem;
            color: rgba(226,232,240,0.9);
          }
          .reset-wrap:hover::after {
            content: attr(data-tooltip);
            position: absolute;
            right: 0;
            top: -42px;
            background: rgba(15, 23, 42, 0.95);
            border: 1px solid rgba(148,163,184,0.35);
            color: #e2e8f0;
            padding: 6px 10px;
            border-radius: 10px;
            font-size: 0.75rem;
            white-space: nowrap;
            box-shadow: 0 10px 22px rgba(0,0,0,0.25);
            z-index: 5;
          }
          .data-table-actions {
            display: flex;
            justify-content: flex-end;
            gap: 12px;
            align-items: center;
          }
          .data-log-tab div.stButton > button {
            background: rgba(15, 23, 42, 0.72) !important;
            border-color: rgba(148, 163, 184, 0.22) !important;
            font-size: 14px !important;
            padding: 8px 12px !important;
          }
          .copy-all-btn div.stButton > button {
            background: rgba(59, 130, 246, 0.18) !important;
            border-color: rgba(59,130,246,0.35) !important;
          }
          .rows-count {
            font-size: 18px;
            font-weight: 700;
          }
          .rows-count .value {
            font-weight: 800;
          }

          /* Responsive filter tiles */
          @media (max-width: 768px) {
            [data-testid="column"] {
              min-width: 100% !important;
              flex: 0 0 100% !important;
            }
            .status-tile {
              width: 85% !important;
            }
            .title-wrap h1 {
              font-size: 2.2rem !important;
            }
            .subtitle-bar {
              font-size: 0.85rem !important;
              flex-wrap: wrap;
              gap: 6px !important;
            }
          }

          @media (min-width: 769px) and (max-width: 1024px) {
            [data-testid="column"] {
              min-width: 50% !important;
              flex: 0 0 50% !important;
            }
            .status-tile {
              width: 80% !important;
            }
          }

          @media (min-width: 1025px) {
            .status-tile {
              width: 70%;
            }
          }

          .card {
            position: relative;
            min-height: 140px;
            margin-bottom: 12px;
          }

          .status-button div.stButton > button {
            margin-top: 6px !important;
            margin-bottom: 6px !important;
          }

          .refresh-row {
            margin-top: 4px !important;
            padding: 6px 0 4px 0;
            gap: 8px;
          }

          .count-banner {
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
          }

          .card:hover .count-banner {
            transform: scale(1.05);
            box-shadow: 0 4px 12px rgba(0,0,0,0.25);
          }

          .card.selected {
            background: transparent;
          }

          [data-testid="stDataFrame"] tbody tr:nth-child(even) {
            background: rgba(255, 255, 255, 0.02) !important;
          }

          [data-testid="stDataFrame"] tbody tr:hover {
            background: rgba(59, 130, 246, 0.08) !important;
            cursor: pointer;
          }

          [data-testid="stDataFrame"] th {
            background: rgba(15, 23, 42, 0.82) !important;
            color: #f1f5f9 !important;
            font-weight: 700 !important;
            text-transform: uppercase !important;
            font-size: 13px !important;
            letter-spacing: 0.5px !important;
            border-bottom: 2px solid rgba(148, 163, 184, 0.3) !important;
          }

        </style>
        """,
        unsafe_allow_html=True,
    )

    # Sticky anchor + JS
    st.markdown('<div class="sticky-anchor"></div>', unsafe_allow_html=True) # type: ignore
    components.html(
        """
        <script>
          function applySticky() {
            const anchor = window.parent.document.querySelector('.sticky-anchor');
            if (!anchor) return;

            let el = anchor;
            for (let i = 0; i < 12; i++) {
              if (!el) break;
              if (el.getAttribute && el.getAttribute("data-testid") === "stVerticalBlock") {
                el.classList.add("sticky-top");
                return;
              }
              el = el.parentElement;
            }
            if (anchor.parentElement) anchor.parentElement.classList.add("sticky-top");
          }

          function styleSyncDashboardButton() {
            const buttons = window.parent.document.querySelectorAll('button');
            buttons.forEach((button) => {
              if (button.textContent && button.textContent.trim().includes('Sync Dashboard')) {
                const wrapper = button.closest('[data-testid="stButton"]');
                if (wrapper) wrapper.classList.add('sync-dashboard-btn');
              }
            });
          }

          applySticky();
          styleSyncDashboardButton();
          setTimeout(applySticky, 250);
          setTimeout(applySticky, 800);
          setTimeout(styleSyncDashboardButton, 250);
          setTimeout(styleSyncDashboardButton, 800);
        </script>
        """,
        height=0,
    )

    # ==========================================
    # 8. STATE
    # ==========================================
    if "active_filters" not in st.session_state: # type: ignore
        st.session_state.active_filters = set()
        logger.info("Initialized active_filters session state")
    if "search_query_input" not in st.session_state:
        st.session_state.search_query_input = ""
    if "status_multi" not in st.session_state:
        st.session_state.status_multi = []
    if "res_multi" not in st.session_state:
        st.session_state.res_multi = []
    if "actioned_multi" not in st.session_state:
        st.session_state.actioned_multi = []
    if "tracking_status_multi" not in st.session_state:
        st.session_state.tracking_status_multi = []
    if "req_dates_selected" not in st.session_state:
        st.session_state.req_dates_selected = []
    if "failure_refund" not in st.session_state:
        st.session_state.failure_refund = False
    if "failure_upload" not in st.session_state:
        st.session_state.failure_upload = False
    if "failure_shipment" not in st.session_state:
        st.session_state.failure_shipment = False
    if "cache_version" not in st.session_state:
        st.session_state.cache_version = 0

    filter_migration = {
        "Pending": "Pending Requests",
        "Approved": "Approved - Submitted",
    } # type: ignore
    active: Set[str] = st.session_state.active_filters # type: ignore
    migrated = set()
    for old_name in active:
        migrated.add(filter_migration.get(old_name, old_name))
    valid_filters = set(FILTERS.keys())
    migrated = {name for name in migrated if name in valid_filters}
    if migrated != active:
        st.session_state.active_filters = migrated  # type: ignore
        logger.info("Migrated filters: %s -> %s", active, migrated) # type: ignore

    if st.session_state.get("show_toast"):
        st.toast("✅ Updated!", icon="🔄") # type: ignore
        st.session_state["show_toast"] = False

    if RATE_LIMIT_HIT.is_set():
        st.warning(
            "ReturnGO rate limit reached (429). Sync is slowing down and retrying. "
            "If this happens often, sync less frequently or request a higher quota key."
        )
        RATE_LIMIT_HIT.clear()

    # ==========================================
    # 10. HEADER (STICKY)
    # ==========================================
    h1, h2 = st.columns([3, 1], vertical_alignment="top")
 # type: ignore
    with h1:
        st.markdown("<div class='title-wrap'>", unsafe_allow_html=True)
        st.title("Levi's ReturnGO Ops Dashboard")
        st.markdown(
            f"""
            <div class="subtitle-bar">
              <span class="subtitle-dot"></span>
              <span><b>CONNECTED TO:</b> {STORE_URL.upper()}</span>
              <span style="opacity:.45">|</span>
              <span><b>CACHE:</b> {CACHE_EXPIRY_HOURS}h</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with h2:
        last_sync = st.session_state.get("last_sync_pressed") # type: ignore
        last_sync_display = (
            last_sync.strftime("%d/%m/%Y %H:%M:%S") if isinstance(last_sync, datetime) else "--" # type: ignore
        )
        with RATE_LIMIT_LOCK:
            remaining = RATE_LIMIT_INFO.get("remaining")
            limit = RATE_LIMIT_INFO.get("limit")
        if remaining is not None and limit is not None:
            try:
                if isinstance(remaining, (int, str)) and isinstance(limit, (int, str)):
                    remain_int = int(remaining)
                    limit_int = int(limit)
                    if remain_int < limit_int * 0.2:
                        st.warning(f"⚠️ API quota low: {remain_int}/{limit_int} requests remaining")
            except (ValueError, TypeError):
                pass
        st.markdown(
            f"<div class='sync-time-bar'>Last sync: {last_sync_display}</div>", # type: ignore
            unsafe_allow_html=True,
        )
        if st.button("🔄 Sync Dashboard", key="btn_sync_all", width="stretch"):
            st.session_state.last_sync_pressed = datetime.now()
            perform_sync()
        st.markdown( # type: ignore
            "<div class='reset-wrap' data-tooltip='Clears the cached database so the next sync fetches fresh data.'>",
            unsafe_allow_html=True,
        )
        if st.button("🗑️ Reset Cache", key="btn_reset", width="content"):
            if clear_db(): # type: ignore
                st.cache_data.clear()
                st.success("Cache cleared!")
                st.rerun()
            else:
                st.warning("No database file to reset.")
        st.markdown("</div>", unsafe_allow_html=True)
        render_api_action_bar(compact=True)

    st.write("")

    # ==========================================
    # 11. LOAD + PROCESS OPEN RMAs # type: ignore
    # ========================================== # type: ignore
    raw_data = load_open_rmas(st.session_state.cache_version)
    processed_rows = []
 # type: ignore
    counts = {
        "PendingRequests": 0,
        "ApprovedSubmitted": 0, # type: ignore
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
        "refund_failure": 0, # type: ignore
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

        track_nums = [s.get("trackingNumber") for s in shipments if s.get("trackingNumber")] # type: ignore
        track_str = ", ".join(track_nums) if track_nums else ""
        shipment_id = shipments[0].get("shipmentId") if shipments else None
 # type: ignore
        local_tracking_status = rma.get("_local_courier_status") or "Unknown"
        local_tracking_status_display = format_tracking_status_with_icon(local_tracking_status)
        local_tracking_status_lower = local_tracking_status.lower()
        track_link_url = f"https://portal.thecourierguy.co.za/track?ref={track_nums[0]}" if track_nums else ""

        requested_iso = get_event_date_iso(rma, "RMA_CREATED") or (summary.get("createdAt") or "")
        approved_iso = get_event_date_iso(rma, "RMA_APPROVED")
        received_iso = get_received_date_iso(rma)
        resolution_type = get_resolution_type(rma)
        is_nt: bool = not track_nums # type: ignore
        req_dt = parse_yyyy_mm_dd(str(requested_iso)[:10] if requested_iso else "")
        is_fg = (
            status == "Pending" and isinstance(req_dt, datetime) and (today - req_dt.date()).days >= 7
        )  # type: ignore

        comment_texts = [(c.get("htmlText", "") or "") for c in comments]
        comment_texts_lower = [c.lower() for c in comment_texts]
        is_cc = "courier cancelled" in local_tracking_status_lower
        is_ad = status == "Approved" and "delivered" in local_tracking_status_lower # type: ignore
        actioned: bool = is_resolution_actioned(rma, comment_texts_lower)
        actioned_label = resolution_actioned_label(rma)
        is_ra = (actioned_label != "No") and is_ad
        is_nra = (status == "Received") and (not actioned)
        is_tracking_updated = "Yes" if has_tracking_update_comment(comment_texts) else "No"
        has_refund_failure, has_upload_failed, has_shipment_failure, failures = failure_flags(
            comment_texts_lower,
            status=status,
            approved_iso=str(approved_iso) if approved_iso else "", # type: ignore
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

    df_view = pd.DataFrame(processed_rows) # type: ignore
    update_data_table_log(processed_rows)

    for filter_name, cfg in FILTERS.items():
        count_key = cfg["count_key"]  # type: ignore
        fn: FilterFn = cfg["fn"]  # type: ignore
        if df_view.empty: # type: ignore
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
                try: # type: ignore
                    courier_status, courier_checked = maybe_refresh_courier(rma_payload)
                    if courier_status:
                        upsert_rma(
                            rma_id=str(rma_id),
                            status=status,
                            created_at=requested_iso or "",
                            payload=rma_payload,
                            courier_status=courier_status,
                            courier_checked_iso=courier_checked,
                        ) # type: ignore
                except Exception as exc:
                    logger.warning("Background courier refresh failed for %s: %s", rma_id, exc)

        refresh_thread = threading.Thread(target=refresh_courier_batch, daemon=True)
        refresh_thread.start()
        if len(rmas_needing_courier_refresh) > 5:
            st.info(
                f"🔄 Refreshing {len(rmas_needing_courier_refresh)} courier statuses in background..."
            )

    st.divider()

    if not df_view.empty and "Requested date" in df_view.columns and "Current Status" in df_view.columns:
        st.markdown("### 📅 Request Timeline")

        # Filter to open statuses (defensive: df_view may include non-active if search/filters applied)
        timeline_df = df_view[df_view["Current Status"].isin(ACTIVE_STATUSES)].copy()
        timeline_df["_req_date_parsed"] = pd.to_datetime(
            timeline_df["Requested date"], errors="coerce"
        )
        timeline_df = timeline_df.dropna(subset=["_req_date_parsed"])

        if not timeline_df.empty:
            date_counts = (
                timeline_df.groupby(timeline_df["_req_date_parsed"].dt.date) # pyright: ignore[reportAttributeAccessIssue]
                .size()
                .reset_index(name="Count")
            )
            date_counts.columns = ["Date", "Count"]
            st.line_chart(date_counts.set_index("Date"), height=300)
        else:
            st.info("No valid requested dates found for charting.")

        st.write("")

    st.markdown("### 📊 Command Center")
    st.write("")

    # Row 1: Primary statuses # type: ignore
    r1 = st.columns(5) # type: ignore
    render_command_tile(r1[0], "Pending Requests", "Pending", "PENDING REQUESTS", "⏳")
    render_command_tile(
        r1[1], "Approved - Submitted", "FILTER_ApprovedSubmitted", "SUBMITTED TO COURIER", "📬"
    )
    render_command_tile(r1[2], "Approved - In Transit", "FILTER_ApprovedInTransit", "IN TRANSIT", "🚚")
    render_command_tile(r1[3], "Received", "Received", "RECEIVED / AT WAREHOUSE", "📦")
    render_command_tile(r1[4], "No Tracking", "FILTER_NoTracking", "NO TRACKING NUMBER", "🚫")

    st.write("")

    # Row 2: Secondary statuses # type: ignore
    r2 = st.columns(5) # type: ignore
    render_command_tile(r2[0], "Flagged", "FILTER_Flagged", "FLAGGED", "🚩")
    render_command_tile(r2[1], "Courier Cancelled", "FILTER_CourierCancelled", "CANCELLED", "🛑")
    render_command_tile(
        r2[2], "Approved - Delivered", "FILTER_ApprovedDelivered", "APPROVED > DELIVERED", "📬"
    )
    render_command_tile(r2[3], "Resolution Actioned", "FILTER_ResolutionActioned", "RESOLVED", "💳")
    render_command_tile(
        r2[4],
        "No Resolution Actioned",
        "FILTER_NoResolutionActioned",
        "PENDING RESOLUTION",
        "⏸️",
    )

    st.write("")

    # ==========================================
    # 14. SEARCH + VIEW ALL + FILTER BAR
    # ==========================================
    st.write("") # type: ignore
    sc1, sc2, sc3 = st.columns([8, 1, 1], vertical_alignment="center")
    with sc1:
        st.text_input(
            "Search", # type: ignore
            placeholder="🔍 Search Order, RMA, or Tracking...",
            label_visibility="collapsed",
            key="search_query_input",
        )
    with sc2:
        st.button("🧹 Clear", width="stretch", on_click=clear_all_filters) # type: ignore
    with sc3:
        if st.button("📋 View All", width="stretch"): # type: ignore
            st.session_state.active_filters = set()  # type: ignore
            st.rerun()

    # Extra filter bar/drop (under search)
    with st.expander("Additional filters", expanded=True):
        c1, c2, c3, c4, c5, c6 = st.columns([2, 2, 2, 2, 3, 1], vertical_alignment="center") # type: ignore

        def multi_select_with_state(label: str, options: list, key: str):
            old = st.session_state.get(key, [])
            selections = [x for x in old if x in options]
            if selections != old:
                st.session_state[key] = selections
            return st.multiselect(label, options=options, key=key)

        with c1:
            multi_select_with_state("Status", ACTIVE_STATUSES, "status_multi") # type: ignore
        with c2:
            res_opts = []
            if not df_view.empty and "resolutionType" in df_view.columns:
                res_opts = sorted(
                    [x for x in df_view["resolutionType"].dropna().unique().tolist() if x and x != "N/A"]
                ) # type: ignore
            multi_select_with_state("Resolution type", res_opts, "res_multi") # type: ignore
        with c3:
            multi_select_with_state("Resolution actioned", ["Yes", "No"], "actioned_multi") # type: ignore
        with c4:
            tracking_opts = [
                format_tracking_status_with_icon(status) for status in COURIER_STATUS_OPTIONS
            ]
            multi_select_with_state("Tracking status", tracking_opts, "tracking_status_multi") # type: ignore
        with c5:
            req_dates = [] # type: ignore
            if not df_view.empty and "Requested date" in df_view.columns:
                req_dates = sorted(
                    [d for d in df_view["Requested date"].dropna().astype(str).unique().tolist() if d and d != "N/A"],
                    reverse=True, # type: ignore
                )
            multi_select_with_state("Requested date (multi-select)", req_dates, "req_dates_selected")
        with c6:
            st.button("🧼 Clear filters", width="stretch", on_click=clear_all_filters) # type: ignore

        st.markdown("**Failure filters**")
        f1, f2, f3, _ = st.columns([1.3, 1.3, 1.3, 3], vertical_alignment="center")
        with f1: # type: ignore
            st.checkbox(
                f"refund_failure ({failure_counts['refund_failure']})",
                key="failure_refund",
            )
        with f2: # type: ignore
            st.checkbox(
                f"upload_failed ({failure_counts['upload_failed']})",
                key="failure_upload",
            )
        with f3: # type: ignore
            st.checkbox(
                f"shipment_failure ({failure_counts['shipment_failure']})",
                key="failure_shipment",
            )

    st.divider()

    # ========================================== # type: ignore
    # 15. TABLE VIEW
    # ==========================================
    if df_view.empty: # type: ignore
        st.warning("Database empty. Click Sync Dashboard to start.")
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
                ("refund_failure", "failure_refund"), # type: ignore
            ("upload_failed", "failure_upload"),
            ("shipment_failure", "failure_shipment"),
        )
        if st.session_state.get(key)
    ]

    if status_multi:
        display_df = display_df[display_df["Current Status"].isin(status_multi)] # type: ignore
    if res_multi:
        display_df = display_df[display_df["resolutionType"].isin(res_multi)] # type: ignore
    if actioned_multi:
        actioned_set: Set[str] = set(actioned_multi) # type: ignore
        if actioned_set == {"Yes"}:
            display_df = display_df[display_df["Resolution actioned"] != "No"]
        elif actioned_set == {"No"}:
            display_df = display_df[display_df["Resolution actioned"] == "No"]
    if tracking_status_multi:
        display_df = display_df[display_df["Tracking Status"].isin(tracking_status_multi)]
    if req_dates_selected:
        display_df = display_df[display_df["Requested date"].isin(req_dates_selected)]
    if failure_selected:
        failure_mask = pd.Series(False, index=display_df.index) # type: ignore
        if st.session_state.get("failure_refund"):
            failure_mask |= display_df["has_refund_failure"]
        if st.session_state.get("failure_upload"):
            failure_mask |= display_df["has_upload_failed"]
        if st.session_state.get("failure_shipment"):
            failure_mask |= display_df["has_shipment_failure"]
        display_df = display_df[failure_mask] # type: ignore
 # type: ignore
    if display_df.empty:
        st.info("No matching records found.")
        st.stop()
    
    # Defensive check: Ensure 'Requested date' column exists before sorting
    if "Requested date" not in display_df.columns:
        logger.error("KeyError: 'Requested date' column missing from display_df. Columns: %s", display_df.columns.tolist())
        st.error("Internal error: 'Requested date' column is missing from the data table. Please try syncing again.")
        st.stop()

    display_df = display_df.sort_values(by="Requested date", ascending=False).reset_index(drop=True)
    display_df["No"] = (display_df.index + 1).astype(str)
 # type: ignore
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
        rma_url = row.get("RMA ID", "") # type: ignore
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
                    if not shipment_id: # type: ignore
                        st.error("No Shipment ID available for this RMA.")
                    else:
                        new_track_value = new_track.strip() if new_track else ""
                        ok, msg = push_tracking_update(rma_id_text, shipment_id, new_track_value)
                        if ok:
                            if new_track_value != track_existing:
                                comment_payload = tracking_update_comment(track_existing, new_track_value)
                                push_comment_update(rma_id_text, comment_payload)
                            st.success("Tracking updated.")
                            time.sleep(0.2) # type: ignore
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
                            ok, msg = push_comment_update(rma_id_text, comment) # type: ignore
                            if ok:
                                st.success("Comment posted.")
                                time.sleep(0.2)
                                perform_sync(["Approved", "Received"])
                            else:
                                st.error(msg)

            full: Dict[str, Any] = row.get("full_data", {}) or {} # type: ignore
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

    column_config = {
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

    _table_df = display_df[display_cols + ["_rma_id_text", "DisplayTrack", "shipment_id", "full_data"]].copy()

    total_rows = len(_table_df)
    tsv_rows = [display_cols] + _table_df[display_cols].astype(str).values.tolist()
    tsv_text = "\n".join(["\t".join(row) for row in tsv_rows])

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
                try {{
                  const payload = JSON.parse({json_payload});
                  if (navigator.clipboard?.writeText) {{
                    await navigator.clipboard.writeText(payload);
                  }} else {{
                    console.warn("Clipboard API unavailable.");
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

    def highlight_missing_tracking(row: pd.Series):
        if row.get("Current Status") == "Approved" and not row.get("DisplayTrack"):
            return ["background-color: rgba(220, 38, 38, 0.35); color: #fee2e2;"] * len(display_cols)
        return [""] * len(display_cols)

    styled_table = _table_df[display_cols].style.apply(highlight_missing_tracking, axis=1)

    table_key = "rma_table"
    sel_event = st.dataframe( # type: ignore
        styled_table,
        width="stretch",
        height=700,
        hide_index=True,
        column_config=column_config,
        on_select="rerun",
        selection_mode="single-row",
        key=table_key,
    )

    sel_rows = (sel_event.selection.rows if sel_event and sel_event.selection and hasattr(sel_event.selection, "rows") else []) or [] # type: ignore
    if sel_rows:
        if st.session_state.pop("suppress_row_dialog", False):
            table_state = st.session_state.get(table_key, {})
            if isinstance(table_state, dict):
                table_state["selection"] = {"rows": []}
                st.session_state[table_key] = table_state
        else:
            idx = int(sel_rows[0])
            show_rma_actions_dialog(display_df.iloc[idx])


if __name__ == "__main__":
    init_db()
    main() # type: ignore
