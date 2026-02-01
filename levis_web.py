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

        
# Initialize the database connection engine at the module level
# engine = init_database() - REMOVED

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

def update_performance_metrics(api_time: float):
    """Update performance metrics with new API call time."""
    metrics = load_performance_metrics()
    metrics.api_call_times.append(api_time)
    if len(metrics.api_call_times) > 100:
        metrics.api_call_times = metrics.api_call_times[-100:]
    if metrics.api_call_times:
        metrics.avg_response_time = sum(metrics.api_call_times) / len(metrics.api_call_times)
    save_performance_metrics(metrics)

# ==========================================
# 2. SAFE UTILITIES
# ==========================================
def safe_parse_date_iso(s: Optional[str]) -> Optional[datetime]:
    """Parses ISO8601 timestamp or returns None."""
    if not s or not isinstance(s, str):
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def format_date(dt: Optional[datetime], fmt: str = "%Y-%m-%d %H:%M") -> str:
    """Safely format a datetime, returning '-' if None."""
    if not dt:
        return "-"
    return dt.strftime(fmt)


def safe_get(data: Any, path: str, default=None) -> Any:
    """Safely extract nested dict value using dot notation."""
    if not isinstance(data, dict):
        return default
    keys = path.split(".")
    val = data
    for k in keys:
        if not isinstance(val, dict):
            return default
        val = val.get(k, default)
        if val is None:
            return default
    return val


def compute_days_since(from_date: Optional[datetime]) -> int:
    """Calculate how many days ago `from_date` was."""
    if not from_date:
        return 0
    now_utc = datetime.now(timezone.utc)
    from_date_utc = from_date.astimezone(timezone.utc) if from_date.tzinfo else from_date.replace(tzinfo=timezone.utc)
    delta = now_utc - from_date_utc
    return delta.days


def get_event_date(rma_data: dict, event_name: str) -> Optional[datetime]:
    """Extracts the date for a specific event from the RMA events list."""
    events = safe_get(rma_data, "rmaSummary.events", [])
    for event in events:
        if event.get("eventName") == event_name:
            return safe_parse_date_iso(event.get("eventDate"))
    return None


def get_resolution_types(rma_data: dict) -> Set[str]:
    """Gets a set of all unique resolution types from the RMA items."""
    items = safe_get(rma_data, "items", [])
    if not items:
        return set()
    return {item.get("resolutionType") for item in items if item.get("resolutionType")}

def has_tracking_update_comment(rma_data: dict) -> bool:
    """Checks if a comment indicating a tracking update exists."""
    comments = safe_get(rma_data, "comments", [])
    return any("updated shipment tracking number" in str(c.get("htmlText", "")).lower() for c in comments)

def get_requests_session() -> requests.Session:
    """Creates a session with retry logic for HTTP requests."""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
        backoff_factor=2,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def update_rate_limit_info(response_headers: Mapping[str, str]):
    """Updates the global rate limit state from response headers."""
    with RATE_LIMIT_LOCK:
        if "X-RateLimit-Remaining" in response_headers:
            RATE_LIMIT_INFO["remaining"] = response_headers.get("X-RateLimit-Remaining")
        if "X-RateLimit-Limit" in response_headers:
            RATE_LIMIT_INFO["limit"] = response_headers.get("X-RateLimit-Limit")
        if "X-RateLimit-Reset" in response_headers:
            reset_str = response_headers.get("X-RateLimit-Reset")
            if reset_str:
                try:
                    RATE_LIMIT_INFO["reset"] = datetime.fromtimestamp(int(reset_str), tz=timezone.utc)
                except (ValueError, TypeError):
                    pass
        RATE_LIMIT_INFO["updated_at"] = datetime.now(timezone.utc)

# ==========================================
# 3. RATE LIMITER
# ==========================================
class RateLimiter:
    """Enforces calls/second rate limits for API calls."""

    def __init__(self, calls_per_second: float):
        self.calls_per_second = calls_per_second
        self.interval = 1.0 / calls_per_second if calls_per_second > 0 else 0
        self.last_call = 0.0
        self.lock = threading.Lock()

    def wait_if_needed(self):
        """Blocks until we're allowed to make another call."""
        with self.lock:
            if self.interval > 0:
                elapsed = time.time() - self.last_call
                sleep_time = self.interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
            self.last_call = time.time()


returngo_limiter = RateLimiter(RG_RPS)



# ==========================================
# 5. RETURNGO API CALLS
# ==========================================
def fetch_returngo_rmas(
    api_key: str,
    store_url: str,
    status: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_pages: int = 99999,
) -> List[dict]:
    """
    Fetches RMA pages from ReturnGO API using the correct endpoint structure.
    """
    all_rmas = []
    session = get_requests_session()
    page = 1
    cursor = None

    while page <= max_pages:
        returngo_limiter.wait_if_needed()
        
        # Build params according to API spec
        params = {
            "pagesize": "100"
        }
        
        if cursor:
            params["cursor"] = cursor
        
        if status:
            params["status"] = status
            
        if start_date:
            params["rma_created_at"] = f"gte:{start_date}"
            
        if end_date:
            params["rma_updated_at"] = f"lte:{end_date}"

        try:
            start_time = time.time()
            resp = session.get(
                "https://api.returngo.ai/rmas",
                headers={
                    "x-api-key": api_key,
                    "x-shop-name": store_url,
                    "Content-Type": "application/json",
                },
                params=params,
                timeout=60,
            )
            api_time = time.time() - start_time
            update_performance_metrics(api_time)
            
            update_rate_limit_info(resp.headers)

            if resp.status_code == 429:
                logger.warning(f"Rate limit hit on page {page}. Sleeping 60s.")
                RATE_LIMIT_HIT.set()
                time.sleep(60)
                RATE_LIMIT_HIT.clear()
                continue

            resp.raise_for_status()
            data = resp.json()
            rmas = data.get("rmas", [])
            
            if not rmas:
                logger.info(f"No more RMAs on page {page}. Ending fetch.")
                break

            all_rmas.extend(rmas)
            logger.info(f"Page {page}: fetched {len(rmas)} RMAs (total so far: {len(all_rmas)})")
            
            # Check for cursor to continue pagination
            cursor = data.get("cursor")
            if not cursor:
                break
                
            page += 1

        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTP error on page {page}: {http_err}", exc_info=True)
            break
        except Exception as e:
            logger.error(f"Error fetching page {page}: {e}", exc_info=True)
            break

    logger.info(f"Fetch completed. Total RMAs retrieved: {len(all_rmas)}")
    return all_rmas


def get_rma_by_id(api_key: str, store_url: str, rma_id: str) -> Optional[dict]:
    """Fetches a single RMA by ID from ReturnGO."""
    returngo_limiter.wait_if_needed()
    session = get_requests_session()
    
    try:
        start_time = time.time()
        resp = session.get(
            f"https://api.returngo.ai/rma/{rma_id}",
            headers={
                "x-api-key": api_key,
                "x-shop-name": store_url,
                "Content-Type": "application/json",
            },
            timeout=60,
        )
        api_time = time.time() - start_time
        update_performance_metrics(api_time)
        
        update_rate_limit_info(resp.headers)
        resp.raise_for_status()
        data = resp.json()
        return data
    except Exception as e:
        logger.error(f"Error fetching RMA {rma_id}: {e}", exc_info=True)
        return None


def post_rma_comment(api_key: str, store_url: str, rma_id: str, comment_text: str) -> bool:
    """Posts a comment to a specific RMA. Returns True if successful."""
    returngo_limiter.wait_if_needed()
    session = get_requests_session()
    payload = {"comment": comment_text}
    try:
        start_time = time.time()
        resp = session.post(
            f"https://api.returngo.ai/rma/{rma_id}/comment",
            headers={
                "x-api-key": api_key,
                "x-shop-name": store_url,
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        api_time = time.time() - start_time
        update_performance_metrics(api_time)
        
        update_rate_limit_info(resp.headers)
        resp.raise_for_status()
        logger.info(f"Posted comment to RMA {rma_id}.")
        return True
    except Exception as e:
        logger.error(f"Failed to post comment to RMA {rma_id}: {e}", exc_info=True)
        return False

# ==========================================
# 6. DATA FETCHING AND CACHING
# ==========================================
@st.cache_data(ttl=43200)
def fetch_and_cache_data() -> pd.DataFrame:
    """
    Fetches all active RMAs from the ReturnGO API, enriches them with details
    and courier statuses, and caches the resulting DataFrame for 12 hours.
    This is the primary data source for the application.
    """
    if not MY_API_KEY:
        st.error("Application error: 'RETURNGO_API_KEY' is not configured in secrets.")
        return pd.DataFrame()

    logger.info("Cache miss. Fetching all active RMAs from ReturnGO API...")
    st.session_state.last_sync_time = datetime.now(timezone.utc)

    # 1. Fetch RMA lists for all active statuses
    all_rmas_list = []
    for status_val in ACTIVE_STATUSES:
        logger.info(f"Fetching RMAs with status: {status_val}")
        batch = fetch_returngo_rmas(
            api_key=MY_API_KEY,
            store_url=STORE_URL,
            status=status_val,
        )
        all_rmas_list.extend(batch)
    logger.info(f"Total RMAs fetched from lists: {len(all_rmas_list)}")

    def process_rma(rma_summary: dict) -> Optional[dict]:
        """
        Processes a single RMA summary, enriches it with courier status if applicable.
        Note: This version assumes the list endpoint provides sufficient detail.
        A full detail fetch per RMA can be added here if necessary.
        """
        rma_id = rma_summary.get("rmaId")
        if not rma_id:
            return None

        rma_detail = rma_summary

        # Fetch and add courier status for 'Approved' RMAs
        tracking_number = safe_get(rma_detail, "returnLabel.trackingNumber")
        if rma_detail.get("status", "").lower() == "approved" and tracking_number and PARCEL_NINJA_TOKEN:
            pn_data = fetch_parcel_ninja_tracking(tracking_number, PARCEL_NINJA_TOKEN)
            if pn_data:
                rma_detail['courier_status'] = pn_data.get("status")
        
        # Structure the data for the DataFrame
        return {
            "rma_id": rma_id,
            "store_url": rma_detail.get("storeUrl"),
            "status": rma_detail.get("status"),
            "created_at": safe_parse_date_iso(rma_detail.get("createdAt")),
            "json_data": rma_detail,  # The full object is the "json_data"
            "courier_status": rma_detail.get('courier_status'), # From Parcel Ninja
        }

    # 2. Concurrently process all RMAs
    processed_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_rma = {executor.submit(process_rma, rma): rma for rma in all_rmas_list}
        for future in concurrent.futures.as_completed(future_to_rma):
            result = future.result()
            if result:
                processed_results.append(result)
    
    logger.info(f"Successfully processed details for {len(processed_results)} RMAs.")

    if not processed_results:
        return pd.DataFrame()

    # 3. Convert to DataFrame and return
    df = pd.DataFrame(processed_results)
    return df

def fetch_parcel_ninja_tracking(tracking_number: str, token: str) -> Optional[dict]:
    """Fetches tracking info from Parcel Ninja."""
    if not token or not tracking_number:
        return None
    session = get_requests_session()
    try:
        url = f"https://api.shiplogic.com/v2/tracking/{tracking_number}"
        resp = session.get(
            url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error(f"Error fetching PN tracking for {tracking_number}: {e}", exc_info=True)
        return None

# ==========================================
# 8. DATA FRAME ENRICHMENT
# ==========================================
def enrich_rma_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches the raw DB dataframe with derived columns and classifications.
    """
    if df.empty:
        return df

    df = df.copy()

    # The 'json_data' column is now always a dict from the fetcher.
    # No need to parse from string anymore.

    # Extract fields using correct API field names
    df["order_name"] = df["json_data"].apply(lambda x: safe_get(x, "orderName", "-"))
    df["requested_date"] = df["json_data"].apply(lambda x: safe_parse_date_iso(safe_get(x, "createdAt")))
    df["approved_date"] = df["json_data"].apply(lambda x: safe_parse_date_iso(safe_get(x, "approvedDate")))
    df["received_date"] = df["json_data"].apply(lambda x: safe_parse_date_iso(safe_get(x, "receivedDate")))
    df["tracking_number"] = df["json_data"].apply(lambda x: safe_get(x, "returnLabel.trackingNumber", ""))
    df["resolution_type"] = df["json_data"].apply(lambda x: safe_get(x, "resolutionType", ""))
    df["resolution_actioned"] = df["json_data"].apply(lambda x: safe_get(x, "resolutionActioned", ""))
    df["order_name"] = df["json_data"].apply(lambda x: safe_get(x, "rmaSummary.order_name", "-"))
    df["requested_date"] = df["json_data"].apply(lambda x: get_event_date(x, "RMA_CREATED"))
    df["approved_date"] = df["json_data"].apply(lambda x: get_event_date(x, "RMA_APPROVED"))
    df["received_date"] = df["json_data"].apply(lambda x: get_event_date(x, "RMA_STATUS_UPDATED")) # Assuming status update to Received
    
    # Tracking number is in the first shipment
    df["tracking_number"] = df["json_data"].apply(lambda x: safe_get(x, "shipments.0.trackingNumber", ""))

    # Resolution type can be multiple, so we'll join them
    df["resolution_type"] = df["json_data"].apply(lambda x: ", ".join(get_resolution_types(x)) or "-")

    # Resolution actioned is based on successful transactions or exchange orders
    df["resolution_actioned"] = df["json_data"].apply(
        lambda x: "Yes" if safe_get(x, "transactions") or safe_get(x, "exchangeOrders") else "No"
    )

    # Days since requested
    df["days_since_requested"] = df["requested_date"].apply(compute_days_since)

    # Tracking status classification
    def classify_tracking_status(row):
        courier_status = row.get("courier_status", "")
        status = row.get("status", "")
        tracking_number = row.get("tracking_number", "")

        if not tracking_number:
            return "No tracking number"
        if courier_status:
            return courier_status
        if status.lower() == "approved":
            return "Submitted to Courier"
        return "-"

    df["tracking_status"] = df.apply(classify_tracking_status, axis=1)

    # DisplayTrack for filtering
    df["DisplayTrack"] = df.apply(
        lambda r: bool(r.get("tracking_number")) and r.get("status", "").lower() in ["approved", "received"],
        axis=1
    )

    # Check for tracking update comment
    df["Is_tracking_updated"] = df["json_data"].apply(has_tracking_update_comment)

    # Flags
    df["is_cc"] = df["tracking_status"].str.lower().str.contains("cancelled", case=False, na=False)
    df["is_fg"] = df.apply(
        lambda r: (r.get("status", "").lower() in ["approved", "received"]) and not r.get("DisplayTrack", False),
        axis=1
    )

    # Failure detection
    def detect_failures(row):
        failures = []
        status = row.get("status", "")
        tracking_status = row.get("tracking_status", "")
        resolution_actioned = row.get("resolution_actioned", "")

        if status.lower() == "approved" and not row.get("tracking_number"):
            failures.append("NO_TRACKING")
        if "cancelled" in tracking_status.lower():
            failures.append("COURIER_CANCELLED")
        if status.lower() == "received" and resolution_actioned.lower() != "yes":
            failures.append("NO_RESOLUTION")

        return ", ".join(failures) if failures else ""

    df["failures"] = df.apply(detect_failures, axis=1)

    return df


def compute_counts(df: pd.DataFrame) -> Dict[str, int]:
    """
    Computes various counts from the enriched dataframe.
    """
    if df.empty:
        return {}

    counts_dict = {}

    # Total open
    counts_dict["Total Open"] = len(df[df["status"].str.lower().isin(["pending", "approved", "received"])])

    # Status-based
    counts_dict["Pending"] = len(df[df["status"].str.lower() == "pending"])
    counts_dict["Approved"] = len(df[df["status"].str.lower() == "approved"])
    counts_dict["Received"] = len(df[df["status"].str.lower() == "received"])

    # Tracking-based
    counts_dict["In Transit"] = len(df[df["tracking_status"].str.lower() == "in transit"])
    counts_dict["Submitted"] = len(df[df["tracking_status"].str.lower() == "submitted to courier"])
    counts_dict["Delivered"] = len(df[df["tracking_status"].str.lower() == "delivered"])
    counts_dict["Courier Cancelled"] = len(df[df["tracking_status"].str.lower() == "courier cancelled"])
    counts_dict["No Tracking"] = len(df[df["tracking_status"].str.lower() == "no tracking number"])

    # Resolution
    counts_dict["Resolution Actioned"] = len(df[df["resolution_actioned"].str.lower() == "yes"])
    counts_dict["No Resolution Actioned"] = len(df[
        (df["status"].str.lower() == "received") & (df["resolution_actioned"].str.lower() != "yes")
    ])

    # Issues
    counts_dict["Issues"] = len(df[df["failures"] != ""])

    return counts_dict

# ==========================================
# 9. STREAMLIT UI - OPS LOG
# ==========================================
def append_ops_log(message: str):
    """Appends a message to the in-memory ops log for display."""
    if "ops_log" not in st.session_state:
        st.session_state.ops_log = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.ops_log.append(f"[{timestamp}] {message}")


def show_ops_log():
    """Displays the ops log in a styled container."""
    if "ops_log" not in st.session_state or not st.session_state.ops_log:
        st.info("No operations logged yet.")
        return
    
    log_text = "\n".join(st.session_state.ops_log[-50:])
    st.markdown(
        f"""
        <div style='background-color: #0a0a0a; color: #00ff00; padding: 15px; 
                    border-radius: 8px; font-family: "Courier New", monospace; 
                    font-size: 12px; max-height: 400px; overflow-y: auto; 
                    border: 1px solid #00ff00;'>
        <pre style='margin: 0; white-space: pre-wrap;'>{html.escape(log_text)}</pre>
        </div>
        """,
        unsafe_allow_html=True
    )

# ==========================================
# 10. STREAMLIT UI - DIALOGS
# ==========================================
@st.dialog("RMA Actions", width="large")
def show_rma_actions_dialog(row_data: pd.Series):
    """Shows a dialog with action buttons for a selected RMA."""
    rma_id = row_data.get("rma_id", "")
    st.subheader(f"Actions for RMA: {rma_id}")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh Data (Clear Cache)", key=f"refresh_{rma_id}"):
            st.cache_data.clear()
            append_ops_log(f"Cache cleared to refresh RMA {rma_id}. Rerunning.")
            st.success(f"Cache cleared. Data will be refreshed.")
            time.sleep(1)
            st.rerun()

    with col2:
        if st.button("💬 Add Comment", key=f"comment_{rma_id}"):
            st.session_state["show_comment_input"] = rma_id

    if st.session_state.get("show_comment_input") == rma_id:
        comment_text = st.text_area("Enter your comment:", key=f"comment_text_{rma_id}")
        if st.button("Submit Comment", key=f"submit_comment_{rma_id}"):
            if comment_text.strip():
                if not MY_API_KEY:
                    st.error("Cannot post comment: 'RETURNGO_API_KEY' is not configured.")
                    return
                success = post_rma_comment(MY_API_KEY, STORE_URL, rma_id, comment_text)
                if success:
                    st.success("Comment posted!")
                    append_ops_log(f"Posted comment to RMA {rma_id}")
                else:
                    st.error("Failed to post comment.")
                st.session_state.pop("show_comment_input", None)
                time.sleep(1)
                st.rerun()
            else:
                st.warning("Comment cannot be empty.")

    st.divider()
    st.subheader("RMA Details")
    
    details_dict = {
        "RMA ID": rma_id,
        "Order": row_data.get("order_name", "-"),
        "Status": row_data.get("status", "-"),
        "Requested": format_date(row_data.get("requested_date")),
        "Approved": format_date(row_data.get("approved_date")),
        "Received": format_date(row_data.get("received_date")),
        "Tracking Number": row_data.get("tracking_number", "-"),
        "Tracking Status": row_data.get("tracking_status", "-"),
        "Resolution Type": row_data.get("resolution_type", "-"),
        "Resolution Actioned": row_data.get("resolution_actioned", "-"),
        "Days Since Requested": row_data.get("days_since_requested", 0),
        "Failures": row_data.get("failures", "-"),
    }
    
    for key, value in details_dict.items():
        st.text(f"{key}: {value}")


@st.dialog("Data Table Log", width="large")
def show_data_table_log():
    """Shows the operations log in a dialog."""
    st.subheader("Activity Log")
    show_ops_log()
    if st.button("Clear Log"):
        st.session_state.ops_log = []
        st.rerun()

# ==========================================
# 11. STREAMLIT UI - FILTER LOGIC
# ==========================================
def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies active filters from session_state to the dataframe.
    """
    if df.empty:
        return df

    filtered = df.copy()
    active_filter = st.session_state.get("active_filter", "All")

    if active_filter == "All":
        return filtered

    # Map filter names to conditions
    filter_map = {
        "Pending Requests": lambda d: d["status"].str.lower() == "pending",
        "Received": lambda d: d["status"].str.lower() == "received",
        "Courier Cancelled": lambda d: d["tracking_status"].str.lower() == "courier cancelled",
        "Approved > Submitted": lambda d: (d["status"].str.lower() == "approved") & (d["tracking_status"].str.lower() == "submitted to courier"),
        "Approved > Delivered": lambda d: (d["status"].str.lower() == "approved") & (d["tracking_status"].str.lower() == "delivered"),
        "No Tracking": lambda d: d["tracking_status"].str.lower() == "no tracking number",
        "Resolution Actioned": lambda d: d["resolution_actioned"].str.lower() == "yes",
        "No Resolution Actioned": lambda d: (d["status"].str.lower() == "received") & (d["resolution_actioned"].str.lower() != "yes"),
        "In Transit": lambda d: d["tracking_status"].str.lower() == "in transit",
        "Issues": lambda d: d["failures"] != "",
    }

    if active_filter in filter_map:
        mask = filter_map[active_filter](filtered)
        filtered = filtered[mask]

    return filtered


def apply_additional_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply additional filters from the filter bar."""
    if df.empty:
        return df
    
    filtered = df.copy()
    
    # Status filter
    status_multi = st.session_state.get("status_multi", [])
    if status_multi:
        filtered = filtered[filtered["status"].isin(status_multi)]
    
    # Resolution type filter
    res_multi = st.session_state.get("res_multi", [])
    if res_multi:
        filtered = filtered[filtered["resolution_type"].isin(res_multi)]
    
    # Resolution actioned filter
    actioned_multi = st.session_state.get("actioned_multi", [])
    if actioned_multi:
        if set(actioned_multi) == {"Yes"}:
            filtered = filtered[filtered["resolution_actioned"].str.lower() == "yes"]
        elif set(actioned_multi) == {"No"}:
            filtered = filtered[filtered["resolution_actioned"].str.lower() != "yes"]
    
    # Tracking status filter
    tracking_multi = st.session_state.get("tracking_multi", [])
    if tracking_multi:
        filtered = filtered[filtered["tracking_status"].isin(tracking_multi)]
    
    # Search query
    search_query = st.session_state.get("search_query_input", "").strip()
    if search_query:
        search_lower = search_query.lower()
        mask = (
            filtered["rma_id"].astype(str).str.lower().str.contains(search_lower, na=False) |
            filtered["order_name"].astype(str).str.lower().str.contains(search_lower, na=False) |
            filtered["tracking_number"].astype(str).str.lower().str.contains(search_lower, na=False)
        )
        filtered = filtered[mask]
    
    return filtered


def clear_all_filters():
    """Clear all filter selections."""
    st.session_state.status_multi = []
    st.session_state.res_multi = []
    st.session_state.actioned_multi = []
    st.session_state.tracking_multi = []
    st.session_state.search_query_input = ""

# ==========================================
# 12. STREAMLIT UI - CSS STYLING
# ==========================================
def inject_custom_css():
    """Injects custom CSS for enhanced UI styling."""
    st.markdown(
        """
        <style>
        /* Top header neon gradient */
        .main-header {
            background: linear-gradient(90deg, 
                rgba(0, 255, 255, 0.3) 0%, 
                rgba(0, 255, 0, 0.3) 33%, 
                rgba(255, 0, 0, 0.3) 66%, 
                rgba(0, 255, 255, 0.3) 100%);
            padding: 20px;
            text-align: center;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 0 20px rgba(0, 255, 255, 0.5);
        }
        
        .main-header h1 {
            color: #ffffff;
            font-weight: bold;
            margin: 0;
            text-shadow: 0 0 10px rgba(0, 255, 255, 0.8);
        }
        
        /* Red neon power button */
        .power-btn {
            background-color: rgba(255, 0, 0, 0.2);
            border: 2px solid #ff0000;
            border-radius: 8px;
            padding: 10px;
            color: #ff0000;
            font-weight: bold;
            box-shadow: 0 0 15px rgba(255, 0, 0, 0.6);
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .power-btn:hover {
            background-color: rgba(255, 0, 0, 0.4);
            box-shadow: 0 0 25px rgba(255, 0, 0, 0.9);
        }
        
        /* Metric cards */
        .metric-card {
            background: linear-gradient(135deg, rgba(30, 30, 30, 0.9), rgba(50, 50, 50, 0.9));
            border: 1px solid rgba(100, 100, 100, 0.3);
            border-radius: 10px;
            padding: 15px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            transition: transform 0.2s, box-shadow 0.2s;
            cursor: pointer;
            margin-bottom: 10px;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 12px rgba(0, 0, 0, 0.5);
        }
        
        .metric-card .count {
            font-size: 32px;
            font-weight: bold;
            color: #00ff00;
            margin-bottom: 5px;
        }
        
        .metric-card .label {
            font-size: 14px;
            color: #aaaaaa;
            text-transform: uppercase;
        }
        
        .metric-card .updated {
            font-size: 11px;
            color: #666666;
            margin-top: 5px;
        }
        
        /* Performance metrics inline */
        .perf-metrics {
            display: flex;
            justify-content: space-around;
            align-items: center;
            background: rgba(20, 20, 20, 0.8);
            border: 1px solid rgba(80, 80, 80, 0.5);
            border-radius: 8px;
            padding: 10px;
            margin-top: 10px;
        }
        
        .perf-metric {
            text-align: center;
            padding: 0 15px;
        }
        
        .perf-metric .value {
            font-size: 18px;
            font-weight: bold;
            color: #00ccff;
        }
        
        .perf-metric .label {
            font-size: 11px;
            color: #888888;
            text-transform: uppercase;
        }
        
        /* Activity log terminal style */
        .activity-log {
            background-color: #0a0a0a;
            color: #00ff00;
            padding: 15px;
            border-radius: 8px;
            font-family: "Courier New", monospace;
            font-size: 12px;
            max-height: 300px;
            overflow-y: auto;
            border: 1px solid #00ff00;
            box-shadow: 0 0 10px rgba(0, 255, 0, 0.3);
        }
        
        /* Left panel navigation box */
        .nav-panel {
            background: rgba(30, 30, 30, 0.9);
            border: 1px solid rgba(100, 100, 100, 0.5);
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
        }
        
        /* Sidebar arrow box */
        .arrow-box {
            background: rgba(40, 40, 40, 0.8);
            border: 1px solid rgba(80, 80, 80, 0.6);
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 15px;
        }
        
        /* Table actions */
        .table-actions-left {
            display: flex;
            gap: 10px;
            margin-bottom: 10px;
        }
        
        .rows-count {
            font-size: 14px;
            color: #cccccc;
            margin-bottom: 10px;
        }
        
        .rows-count .value {
            font-weight: bold;
            color: #00ff00;
        }
        
        .sync-time-bar {
            background: rgba(20, 20, 20, 0.9);
            border: 1px solid rgba(60, 60, 60, 0.5);
            border-radius: 5px;
            padding: 8px;
            text-align: center;
            color: #aaaaaa;
            font-size: 12px;
            margin-top: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# ==========================================
# 13. STREAMLIT UI - MAIN FUNCTION
# ==========================================
def main():
    """Main Streamlit application entry point."""
    st.set_page_config(
        page_title="Levi's ReturnGO Ops Dashboard",
        page_icon="📦",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_custom_css()

    # Initialize session state
    if "active_filter" not in st.session_state:
        st.session_state.active_filter = "All"
    if "last_sync_time" not in st.session_state:
        st.session_state.last_sync_time = None
    if "ops_log" not in st.session_state:
        st.session_state.ops_log = []
    if "user_settings" not in st.session_state:
        st.session_state.user_settings = load_user_settings()
    if "performance_metrics" not in st.session_state:
        st.session_state.performance_metrics = load_performance_metrics()

    # Top header with neon gradient
    st.markdown(
        """
        <div class="main-header">
            <h1>LEVI'S RETURNGO OPS DASHBOARD</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Sidebar
    with st.sidebar:
        st.markdown("<div class='arrow-box'>", unsafe_allow_html=True)
        st.markdown("### Navigation")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='nav-panel'>", unsafe_allow_html=True)
        
        # Power button (red neon)
        st.markdown(
            """
            <div class='power-btn' style='text-align: center; margin-bottom: 15px;'>
                🔴 SYSTEM POWER
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Performance metrics in one line
        perf_metrics = st.session_state.performance_metrics
        avg_resp = f"{perf_metrics.avg_response_time:.2f}s" if perf_metrics.avg_response_time else "N/A"
        last_sync = f"{perf_metrics.last_sync_duration:.2f}s" if perf_metrics.last_sync_duration else "N/A"
        cache_rate = f"{perf_metrics.cache_hit_rate:.1f}%" if perf_metrics.cache_hit_rate else "N/A"
        api_calls = len(perf_metrics.api_call_times)
        
        st.markdown(
            f"""
            <div class='perf-metrics'>
                <div class='perf-metric'>
                    <div class='value'>{avg_resp}</div>
                    <div class='label'>Avg Response</div>
                </div>
                <div class='perf-metric'>
                    <div class='value'>{last_sync}</div>
                    <div class='label'>Last Sync</div>
                </div>
                <div class='perf-metric'>
                    <div class='value'>{cache_rate}</div>
                    <div class='label'>Cache Hit</div>
                </div>
                <div class='perf-metric'>
                    <div class='value'>{api_calls}</div>
                    <div class='label'>API Calls</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown("---")
        
        # Cache controls
        st.subheader("Cache Controls")
        if st.button("🔄 Clear Cache & Refresh Data", use_container_width=True):
            with st.spinner("Clearing cache and refreshing data..."):
                st.cache_data.clear()
                append_ops_log("Cache cleared. Fetching fresh data.")
            st.success("Cache cleared! Data will refresh.")
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Activity log in terminal-style box
        st.subheader("Activity Log")
        st.markdown("<div class='activity-log'>", unsafe_allow_html=True)
        if st.session_state.ops_log:
            log_text = "\n".join(st.session_state.ops_log[-20:])
            st.markdown(f"<pre style='margin: 0; white-space: pre-wrap;'>{html.escape(log_text)}</pre>", unsafe_allow_html=True)
        else:
            st.markdown("<pre style='margin: 0;'>No activity logged yet.</pre>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Main content area
    df_raw = fetch_and_cache_data()
    df_enriched = enrich_rma_dataframe(df_raw)
    counts = compute_counts(df_enriched)

    # REQUESTED TIMELINE GRAPH
    st.markdown("---")
    st.markdown("### 📅 Requested Timeline")
    
    # Filter for active statuses (Total Open RMAs)
    if not df_enriched.empty and "requested_date" in df_enriched.columns and "status" in df_enriched.columns:
        timeline_df = df_enriched[df_enriched["status"].isin(ACTIVE_STATUSES)].copy()
        timeline_df["_req_date_parsed"] = pd.to_datetime(timeline_df["requested_date"], errors="coerce")
        timeline_df = timeline_df.dropna(subset=["_req_date_parsed"])
        
        if not timeline_df.empty:
            date_counts = (
                timeline_df.groupby(timeline_df["_req_date_parsed"].dt.date)  # type: ignore
                .size()
                .reset_index(name="Count")
            )
            date_counts.columns = ["Date", "Count"]
            st.line_chart(date_counts.set_index("Date"), height=300)
        else:
            st.info("No valid requested dates found for charting.")
    else:
        st.info("No data available for Requested Timeline")

    # RMA Command Center
    st.markdown("---")
    st.markdown("### 📊 RMA Command Center")

    # RMA Details section (formerly Performance Metrics)
    st.markdown("#### RMA Details")
    
    # Create metric cards in grid layout
    metric_cols = st.columns(4)
    
    # First row: Total Open, Pending, In Transit, Issues (with hover)
    with metric_cols[0]:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='count'>{counts.get('Total Open', 0)}</div>
                <div class='label'>Total Open</div>
                <div class='updated'>Updated just now</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("View", key="btn_total_open", use_container_width=True):
            st.session_state.active_filter = "All"
            st.rerun()
    
    with metric_cols[1]:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='count'>{counts.get('Pending', 0)}</div>
                <div class='label'>Pending</div>
                <div class='updated'>Updated just now</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("View", key="btn_pending", use_container_width=True):
            st.session_state.active_filter = "Pending Requests"
            st.rerun()
    
    with metric_cols[2]:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='count'>{counts.get('In Transit', 0)}</div>
                <div class='label'>In Transit</div>
                <div class='updated'>Updated just now</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("View", key="btn_in_transit", use_container_width=True):
            st.session_state.active_filter = "In Transit"
            st.rerun()
    
    with metric_cols[3]:
        issues_count = counts.get('Issues', 0)
        st.markdown(
            f"""
            <div class='metric-card' title='Issues include: No Tracking, Courier Cancelled, No Resolution Actioned'>
                <div class='count'>{issues_count}</div>
                <div class='label'>Issues ⓘ</div>
                <div class='updated'>Updated just now</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("View", key="btn_issues", use_container_width=True):
            st.session_state.active_filter = "Issues"
            st.rerun()

    # Additional metric cards
    st.markdown("---")
    
    metric_cols2 = st.columns(4)
    
    # PENDING REQUESTS
    with metric_cols2[0]:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='count'>{counts.get('Pending', 0)}</div>
                <div class='label'>Pending Requests</div>
                <div class='updated'>Updated just now 🔄</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Filter", key="btn_pending_req", use_container_width=True):
            st.session_state.active_filter = "Pending Requests"
            st.rerun()
    
    # RECEIVED
    with metric_cols2[1]:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='count'>{counts.get('Received', 0)}</div>
                <div class='label'>Received</div>
                <div class='updated'>Updated just now 🔄</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Filter", key="btn_received", use_container_width=True):
            st.session_state.active_filter = "Received"
            st.rerun()
    
    # COURIER CANCELLED
    with metric_cols2[2]:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='count'>{counts.get('Courier Cancelled', 0)}</div>
                <div class='label'>Courier Cancelled</div>
                <div class='updated'>Updated just now 🔄</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Filter", key="btn_courier_cancelled", use_container_width=True):
            st.session_state.active_filter = "Courier Cancelled"
            st.rerun()
    
    # APPROVED > SUBMITTED
    with metric_cols2[3]:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='count'>{counts.get('Submitted', 0)}</div>
                <div class='label'>Approved > Submitted</div>
                <div class='updated'>Updated just now 🔄</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Filter", key="btn_submitted", use_container_width=True):
            st.session_state.active_filter = "Approved > Submitted"
            st.rerun()

    # Third row
    metric_cols3 = st.columns(4)
    
    # APPROVED > DELIVERED
    with metric_cols3[0]:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='count'>{counts.get('Delivered', 0)}</div>
                <div class='label'>Approved > Delivered</div>
                <div class='updated'>Updated just now 🔄</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Filter", key="btn_delivered", use_container_width=True):
            st.session_state.active_filter = "Approved > Delivered"
            st.rerun()
    
    # NO TRACKING
    with metric_cols3[1]:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='count'>{counts.get('No Tracking', 0)}</div>
                <div class='label'>No Tracking</div>
                <div class='updated'>Updated just now 🔄</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Filter", key="btn_no_tracking", use_container_width=True):
            st.session_state.active_filter = "No Tracking"
            st.rerun()
    
    # RESOLUTION ACTIONED
    with metric_cols3[2]:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='count'>{counts.get('Resolution Actioned', 0)}</div>
                <div class='label'>Resolution Actioned</div>
                <div class='updated'>Updated just now 🔄</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Filter", key="btn_res_actioned", use_container_width=True):
            st.session_state.active_filter = "Resolution Actioned"
            st.rerun()
    
    # NO RESOLUTION ACTIONED
    with metric_cols3[3]:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='count'>{counts.get('No Resolution Actioned', 0)}</div>
                <div class='label'>No Resolution Actioned</div>
                <div class='updated'>Updated just now 🔄</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Filter", key="btn_no_res_actioned", use_container_width=True):
            st.session_state.active_filter = "No Resolution Actioned"
            st.rerun()

    # Search bar and View All button
    st.markdown("---")
    sc1, sc2, sc3 = st.columns([8, 1, 1], vertical_alignment="center")
    with sc1:
        st.text_input(
            "Search",
            placeholder="🔍 Search Order, RMA, or Tracking...",
            label_visibility="collapsed",
            key="search_query_input",
        )
    with sc3:
        if st.button("📋 View All", use_container_width=True):
            st.session_state.active_filter = "All"
            clear_all_filters()
            st.rerun()

    # Additional filters expander
    with st.expander("Additional filters", expanded=False):
        c1, c2, c3, c4, c5 = st.columns([2, 2, 2, 2, 2], vertical_alignment="center")
        
        with c1:
            st.multiselect("Status", options=ACTIVE_STATUSES, key="status_multi")
        
        with c2:
            res_opts = []
            if not df_enriched.empty and "resolution_type" in df_enriched.columns:
                res_opts = sorted([
                    x for x in df_enriched["resolution_type"].dropna().unique().tolist() 
                    if x and x != "-"
                ])
            st.multiselect("Resolution type", options=res_opts, key="res_multi")
        
        with c3:
            st.multiselect("Resolution actioned", options=["Yes", "No"], key="actioned_multi")
        
        with c4:
            st.multiselect("Tracking Status", options=COURIER_STATUS_OPTIONS, key="tracking_multi")
        
        with c5:
            if st.button("🧼 Clear filters", use_container_width=True):
                clear_all_filters()
                st.rerun()

    # Data table
    st.markdown("---")
    st.subheader("RMA Data Table")
    
    # Show active filter
    if st.session_state.active_filter != "All":
        st.info(f"Active Filter: **{st.session_state.active_filter}**")
        if st.button("Clear Filter"):
            st.session_state.active_filter = "All"
            st.rerun()

    # Apply both main filter and additional filters
    df_filtered = apply_filters(df_enriched)
    df_filtered = apply_additional_filters(df_filtered)
    
    # Display columns (removed DisplayTrack, is_cc, is_fg)
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

    if not df_filtered.empty:
        display_df = df_filtered.copy()
        display_df.insert(0, "No", range(1, len(display_df) + 1))
        
        # Rename columns for display
        display_df = display_df.rename(columns={
            "rma_id": "RMA ID",
            "order_name": "Order",
            "status": "Current Status",
            "tracking_number": "Tracking Number",
            "tracking_status": "Tracking Status",
            "requested_date": "Requested date",
            "approved_date": "Approved date",
            "received_date": "Received date",
            "days_since_requested": "Days since requested",
            "resolution_type": "resolutionType",
            "resolution_actioned": "Resolution actioned",
        })
        
        # Format dates
        for col in ["Requested date", "Approved date", "Received date"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: format_date(x) if pd.notna(x) else "-")
        
        # Add placeholder columns if missing
        if "Is_tracking_updated" not in display_df.columns:
            display_df["Is_tracking_updated"] = "-"
        
        # Create RMA ID links
        display_df["RMA ID"] = display_df["RMA ID"].apply(
            lambda x: f"https://app.returngo.ai/admin/rma?rmaid={x}" if x else "-"
        )
        
        # Create tracking links
        display_df["Tracking Number"] = display_df.apply(
            lambda row: f"https://tracking.pngo.co.za/track?ref={row['Tracking Number']}" 
            if row.get("Tracking Number") and row["Tracking Number"] != "" and row["Tracking Number"] != "-"
            else "-",
            axis=1
        )
        
        render_data_table(display_df, display_cols)
    else:
        st.warning("No RMAs match the current filter.")


def render_data_table(display_df: pd.DataFrame, display_cols: List[str]):
    """Renders the main data table with all interactions."""
    
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
    
    # Rows count
    st.markdown(
        f"<div class='rows-count'>Rows in table: <span class='value'>{total_rows}</span></div>",
        unsafe_allow_html=True,
    )
    
    # CSV and Autosize buttons
    st.markdown("<div class='table-actions-left'>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 6])
    with col1:
        csv_payload = table_df.to_csv(index=False)
        st.download_button(
            label="⬇️ CSV",
            data=csv_payload,
            file_name=f"returngo_table_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key="download_table_csv",
        )
    with col2:
        if st.button("Autosize", key="autosize_btn"):
            st.session_state.table_autosize = not st.session_state.get("table_autosize", False)
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # Prepare TSV for copy
    tsv_rows = [display_cols] + table_df.astype(str).values.tolist()
    tsv_text = "\n".join(["\t".join(row) for row in tsv_rows])
    
    action_col1, action_col2 = st.columns([1, 1])
    with action_col1:
        if st.button("Data table log", key="btn_data_table_log", use_container_width=True):
            st.session_state["suppress_row_dialog"] = True
            show_data_table_log()
    
    with action_col2:
        if st.button("📋 Copy all", key="btn_copy_all", use_container_width=True):
            st.session_state["copy_all_payload"] = tsv_text
            st.session_state["suppress_row_dialog"] = True

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

        # Check failures column
        if row.get("failures"):
            highlight = True

        if highlight:
            return ["background-color: rgba(220, 38, 38, 0.35); color: #fee2e2;"] * len(row)

        return [""] * len(row)

    # Apply styling
    styled_table = table_df.style.apply(highlight_problematic_rows, axis=1)

    table_key = "rma_table"
    sel_event = st.dataframe(
        styled_table,
        use_container_width=True,
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
