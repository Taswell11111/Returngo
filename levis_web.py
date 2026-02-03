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
        level=logging.DEBUG,  # Changed to DEBUG for more visibility
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
from app_classes import UserSettings, PerformanceMetrics, Theme

# ==========================================
# 1a. CONFIGURATION
# ==========================================

# Load API secrets
try:
    MY_API_KEY = st.secrets["RETURNGO_API_KEY"]
except KeyError:
    st.error("Application error: Missing 'RETURNGO_API_KEY' in Streamlit secrets.")
    st.stop()
    MY_API_KEY = None
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
        logger.error(f"Failed to save performance metrics: {e}")

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
    # Check both top-level events and rmaSummary.events
    events = rma_data.get("events", [])
    if not events:
        events = safe_get(rma_data, "rmaSummary.events", [])
    
    for event in events:
        if event.get("eventName") == event_name:
            return safe_parse_date_iso(event.get("eventDate"))
    return None


def get_received_date_from_events_or_comments(rma_data: dict) -> Optional[datetime]:
    """
    Gets the shipment received date, prioritizing the 'SHIPMENT_RECEIVED' event,
    but falling back to the earliest system comment containing 'RECEIVED'.
    """
    # 1. Prioritize the official event
    event_date = get_event_date(rma_data, "SHIPMENT_RECEIVED")
    if event_date:
        return event_date

    # 2. Fallback to searching comments
    comments = safe_get(rma_data, "comments", [])
    if not comments:
        return None

    received_comment_dates = []
    for comment in comments:
        html_text = comment.get("htmlText", "")
        # The user's snippet is: "Shipment &lt;strong&gt;RECEIVED&lt;/strong&gt;"
        if "RECEIVED" in html_text.upper():
            comment_date = safe_parse_date_iso(comment.get("datetime"))
            if comment_date:
                received_comment_dates.append(comment_date)
    
    # Return the earliest date found in comments if any
    return min(received_comment_dates) if received_comment_dates else None


def get_resolution_types(rma_data: dict) -> Set[str]:
    """Gets a set of all unique resolution types from the RMA items."""
    items = safe_get(rma_data, "items", [])
    if not items:
        return set()
    return {item.get("resolutionType") for item in items if item.get("resolutionType")}

def has_tracking_update_comment(rma_data: dict) -> bool:
    """Checks if a comment indicating a tracking update exists."""
    comments = safe_get(rma_data, "comments", [])
    return any("was updated" in str(c.get("htmlText", "")).lower() for c in comments)

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
            cursor = data.get("next_cursor")
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

def scrape_parcel_ninja_status( tracking_url: str) -> Optional[str]:
    """
    Scrapes the Parcel Ninja website to get the latest tracking status.
    Returns the exact status line with the pipe: "Thu, 22 Jan 12:35 | Delivered"
    """
    if not tracking_url:
        return None
    
    # Use the provided tracking URL directly
    url = tracking_url
    tracking_number = url.split('=')[-1] if '=' in url else url.split('/')[-1]

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        content = response.text

        # Try multiple regex patterns to match status
        patterns = [
            r"([A-Za-z]{3}, \d{2} [A-Za-z]{3} \d{2}:\d{2}\s+\|\s+[A-Za-z ]+)",  # Original: "Thu, 22 Jan 12:35 | Delivered"
            r"(\d{2} [A-Za-z]{3} \d{2}:\d{2}\s+\|\s+[A-Za-z ]+)",  # Without day: "22 Jan 12:35 | Delivered"
            r"([A-Za-z]+, \d{2} [A-Za-z]+ \d{4} \d{2}:\d{2}\s+\|\s+[A-Za-z ]+)",  # With year
            r"(Latest Status:\s*[A-Za-z ]+)",  # Simple "Latest Status: Delivered"
            r"([A-Za-z ]+)\s*\|\s*\d{2} [A-Za-z]{3} \d{2}:\d{2}",  # Status first
            r"<strong>([A-Za-z ]+)</strong>",  # Look for strong tags
            r"Status:\s*([A-Za-z ]+)",  # Simple status prefix
        ]
        
        for i, pattern in enumerate(patterns):
            matches = re.findall(pattern, content)
            if matches:
                # Return the exact status line
                full_status = matches[0].strip()
                logger.info(f"Pattern {i} matched: Scraped status '{full_status}' for tracking {tracking_number}")
                return full_status

        # If no pattern matched, try to find any status-like text
        logger.warning(f"Could not extract status for {tracking_number} with regex patterns. Raw content snippet: {content[:500]}...")

    except requests.exceptions.RequestException as e:
        logger.error(f"Error scraping PN tracking for {tracking_number}: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error scraping PN tracking for {tracking_number}: {e}", exc_info=True)
        
    return None


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


def scrape_the_courier_guy_status(tracking_url: Optional[str]) -> Optional[str]:
    """
    Scrapes The Courier Guy website to get the latest tracking status.
    """
    if not tracking_url:
        logger.warning("scrape_the_courier_guy_status: No tracking URL provided.")
        return None

    try:
        logger.info(f"Scraping The Courier Guy status from: {tracking_url}")
        session = get_requests_session()
        response = session.get(tracking_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        response.raise_for_status()
        content = response.text

        # The Courier Guy portal uses a specific structure. The latest status is often the last item in a list.
        # Look for a pattern that matches their status descriptions.
        # Example from public sources: <div class="podb-status-description">Delivered</div>
        pattern = re.compile(r'class="podb-status-description"[^>]*>\s*([^<]+)\s*<', re.IGNORECASE)
        statuses = pattern.findall(content)

        if statuses:
            latest_status = statuses[-1].strip()
            logger.info(f"Found status on The Courier Guy: '{latest_status}'")
            return latest_status
        else:
            logger.warning(f"Could not find status pattern on The Courier Guy page for {tracking_url}")
            return "Status not found on page"

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            logger.warning(f"Tracking not found on The Courier Guy (404) for {tracking_url}")
            return "Tracking not found (404)"
        logger.error(f"HTTP error scraping The Courier Guy for {tracking_url}: {e}", exc_info=True)
        return f"Scraping HTTP error ({e.response.status_code})"
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed for The Courier Guy tracking {tracking_url}: {e}", exc_info=True)
        return "Scraping request failed"
    except Exception:
        logger.exception(f"Unexpected error scraping The Courier Guy for {tracking_url}")
        return "Scraping check failed (UNKNOWN)"


def check_courier_status_web_scraping(tracking_number: str) -> str:
    if not tracking_number:
        logger.debug("check_courier_status_web_scraping: No tracking number provided")
        return "No tracking number"

    try:
        url = f"https://optimise.parcelninja.com/shipment/track?WaybillNo={tracking_number}"
        logger.debug("Fetching courier status for %s from %s", tracking_number, url)
        session = get_requests_session() # Using existing session factory
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

        html_content = res.text

        def format_event(event: dict) -> str:
            status = (event.get("status") or event.get("description") or "Unknown")
            date_str = (event.get("date") or event.get("eventDate") or "")
            try:
                dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                formatted_date = dt.strftime("%Y-%m-%d %H:%M")
                return f"{status}\t{formatted_date}"
            except (ValueError, TypeError):
                return status

        # 1. Attempt to find embedded JSON
        var_pattern = re.compile(r"window\.__INITIAL_STATE__\s*=\s*(\{)", re.IGNORECASE)
        match = var_pattern.search(html_content)
        if match:
            json_start = match.start(1)
            json_str = _extract_json_object(html_content, json_start)
            if json_str:
                try:
                    data = json.loads(json_str)
                    events = safe_get(data, "shipment.events", [])
                    if events and isinstance(events, list):
                        return format_event(events[-1])
                except (json.JSONDecodeError, KeyError, TypeError):
                    pass # Fallback to HTML parsing

        # 2. Fallback to simple HTML parsing
        clean_html = re.sub(r"<(script|style).*?</\1>", "", html_content, flags=re.DOTALL | re.IGNORECASE)
        rows = re.findall(r"<tr[^>]*>.*?</tr>", clean_html, flags=re.IGNORECASE | re.DOTALL)
        if rows:
            # Find the first row that looks like it has event data
            for row_html in rows:
                cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row_html, flags=re.IGNORECASE | re.DOTALL)
                cleaned_cells = [re.sub(r"<[^>]+>", " ", c).strip() for c in cells]
                # A plausible row has at least 2 cells and the first one contains a digit (date/time)
                if len(cleaned_cells) >= 2 and re.search(r"\d", cleaned_cells[0]):
                    return "\t".join(cleaned_cells[:2])

        return "No tracking events found"

    except requests.exceptions.RequestException as exc:
        logger.error("Request failed for %s: %s", tracking_number, exc)
        return "Tracking request failed"
    except Exception as exc:
        logger.exception("Unexpected tracking error for %s", tracking_number)
        return "Tracking check failed (ERR_UNKNOWN)"


def get_shipment_status(rma_data: dict) -> Optional[str]:
    """
    Gets the shipment status by scraping the appropriate courier website based on tracking URL or number format.
    """
    shipments = safe_get(rma_data, "shipments", [])
    if not shipments:
        logger.debug("get_shipment_status: No shipments array in RMA data.")
        return "No tracking number"

    tracking_number = None
    tracking_url = None
    for shipment in shipments:
        if shipment and shipment.get("trackingNumber"):
            tracking_number = shipment.get("trackingNumber")
            tracking_url = shipment.get("trackingUrl")  # This might be None
            break
    
    logger.debug(f"get_shipment_status: tracking_number={tracking_number}, tracking_url={tracking_url}")

    if not tracking_number:
        logger.debug("get_shipment_status: No tracking number found in shipments.")
        return "No tracking number"

    # --- NEW LOGIC: Detect courier and scrape ---

    # 1. Check for The Courier Guy by URL or tracking number format
    is_courier_guy = False
    if tracking_url and "thecourierguy.co.za" in tracking_url:
        is_courier_guy = True
    elif tracking_number.startswith("OPT-"):
        is_courier_guy = True
        # If URL is missing, construct it.
        if not tracking_url:
            tracking_url = f"https://portal.thecourierguy.co.za/track?ref={tracking_number}"
            logger.info(f"Constructed The Courier Guy URL: {tracking_url}")
    
    if is_courier_guy:
        logger.info(f"Using The Courier Guy scraper for: {tracking_url}")
        return scrape_the_courier_guy_status(tracking_url)

    # 2. Check for Parcel Ninja via URL (if needed for other cases)
    if tracking_url and "parcelninja.com" in tracking_url:
         logger.info(f"Detected Parcel Ninja URL. Using PN scraper for: {tracking_number}")
         return check_courier_status_web_scraping(tracking_number)

    # 3. If no specific courier detected, use the default (Parcel Ninja) scraper as a fallback
    logger.info(f"No specific courier detected. Using default (Parcel Ninja) scraper for tracking number: {tracking_number}")
    scraped_status = check_courier_status_web_scraping(tracking_number)
    
    logger.debug(f"get_shipment_status: default scraper returned scraped_status='{scraped_status}'")
    if scraped_status:
        logger.info(f"Default scraper returned '{scraped_status}' for tracking {tracking_number}")
        return scraped_status
    else:
        logger.warning(f"Default scraper failed for tracking {tracking_number}")
        return f"Tracking: {tracking_number} (Status unknown)"



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

    progress_bar = st.progress(0, text="Starting data sync...")

    logger.info("Cache miss. Fetching all active RMAs from ReturnGo API...")
    sync_start = time.time()
    st.session_state.last_sync_time = datetime.now(timezone.utc)

    # 1. Fetch RMA lists for all active statuses
    all_rmas_list = []
    for i, status_val in enumerate(ACTIVE_STATUSES):
        progress_text = f"Fetching RMAs with status: {status_val}..."
        progress_bar.progress((i + 1) / (len(ACTIVE_STATUSES) + 1), text=progress_text)
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
        Processes a single RMA summary, fetches full details, and enriches with courier status.
        """
        rma_id = rma_summary.get("rmaId")
        if not rma_id:
            return None

        if not MY_API_KEY:
            logger.warning(f"Skipping RMA {rma_id} due to missing API key.")
            return None

        # Fetch full RMA details
        rma_detail = get_rma_by_id(MY_API_KEY, STORE_URL, rma_id)
        if not rma_detail:
            # Fallback to summary data if detail fetch fails
            rma_detail = rma_summary

        # Get tracking number
        tracking_number = safe_get(rma_detail, "shipments.0.trackingNumber")
        status = safe_get(rma_detail, "rmaSummary.status") or rma_detail.get("status", "")
        
        # Get courier status for Approved or Received RMAs with tracking
        courier_status = None
        if status.lower() in ["approved", "received"]:
            courier_status = get_shipment_status(rma_detail)
        
        if courier_status:
            rma_detail['courier_status'] = courier_status
        
        # Structure the data for the DataFrame
        return {
            "rma_id": rma_id,
            "store_url": safe_get(rma_detail, "rmaSummary.storeUrl") or rma_detail.get("storeUrl"),
            "status": status,
            "created_at": safe_parse_date_iso(rma_detail.get("createdAt")),
            "json_data": rma_detail,
            "courier_status": courier_status,
        }

    # 2. Concurrently process all RMAs
    processed_results = []
    total_rmas = len(all_rmas_list)
    progress_bar.progress(0.5, text=f"Processing {total_rmas} RMA details...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_rma = {executor.submit(process_rma, rma): rma for rma in all_rmas_list}
        for i, future in enumerate(concurrent.futures.as_completed(future_to_rma)):
            result = future.result()
            if result:
                processed_results.append(result)
            if total_rmas > 0:
                progress_bar.progress(0.5 + (i + 1) / total_rmas * 0.5, text=f"Processing details... ({i+1}/{total_rmas})")
    
    logger.info(f"Successfully processed details for {len(processed_results)} RMAs.")

    progress_bar.progress(1.0, text="Sync complete!")

    if not processed_results:
        return pd.DataFrame()

    # 3. Convert to DataFrame and return
    df = pd.DataFrame(processed_results)
    
    # DEBUG: Check what's in the DataFrame
    logger.debug(f"DataFrame columns: {df.columns.tolist()}")
    if "courier_status" in df.columns:
        logger.debug(f"courier_status column stats - Total: {len(df)}, Non-null: {df['courier_status'].notna().sum()}, Non-empty: {(df['courier_status'] != '').sum()}")
        sample_values = df['courier_status'].head(5).tolist()
        logger.debug(f"Sample courier_status values: {sample_values}")
    
    # Update sync duration
    sync_duration = time.time() - sync_start
    metrics = load_performance_metrics()
    metrics.last_sync_duration = sync_duration
    save_performance_metrics(metrics)
    time.sleep(1) # Keep the progress bar visible for a moment
    
    return df



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
    
    # DEBUG: Show info about courier_status column
    if "courier_status" in df.columns:
        non_null_count = df['courier_status'].notna().sum()
        non_empty_count = (df['courier_status'] != '').sum()
        st.info(f"DEBUG: DataFrame has {len(df)} rows. courier_status column: {non_null_count} non-null, {non_empty_count} non-empty")
        if non_null_count > 0:
            sample_values = df['courier_status'].head(3).tolist()
            st.info(f"DEBUG: Sample courier_status values: {sample_values}")
    else:
        st.error("DEBUG: courier_status column is missing from DataFrame!")

    def get_comment_date(comments: list, text_to_find: str) -> Optional[datetime]:
        if not comments:
            return None
        for comment in comments:
            if text_to_find in comment.get("htmlText", ""):
                return safe_parse_date_iso(comment.get("datetime"))
        return None

    # Extract order_name from multiple possible locations
    def extract_order_name(json_data: dict) -> str:
        # Try rmaSummary.order_name first
        order_name = safe_get(json_data, "rmaSummary.order_name")
        if order_name:
            return order_name
        # Try orderDetails.order_name
        order_name = safe_get(json_data, "orderDetails.order_name")
        if order_name:
            return order_name
        # Try top-level order_name (from list endpoint)
        order_name = json_data.get("order_name")
        if order_name:
            return order_name
        return "-"
    
    df["order_name"] = df["json_data"].apply(extract_order_name)
    
    def get_first_tracking(json_data: dict) -> str:
        shipments = safe_get(json_data, "shipments", [])
        if not shipments:
            return ""
        for s in shipments:
            if s and s.get("trackingNumber"):
                return s.get("trackingNumber")
        return ""
    
    df["tracking_number"] = df["json_data"].apply(get_first_tracking)
    df["requested_date"] = df["json_data"].apply(lambda x: get_event_date(x, "RMA_CREATED"))
    df["approved_date"] = df["json_data"].apply(lambda x: get_event_date(x, "RMA_APPROVED"))
    df["received_date"] = df["json_data"].apply(get_received_date_from_events_or_comments)
    
    def extract_resolution_types(json_data: dict) -> str:
        res_types = get_resolution_types(json_data)
        return ", ".join(res_types) if res_types else "-"
    
    df["resolution_type"] = df["json_data"].apply(extract_resolution_types)
    
    df["resolution_actioned"] = df["json_data"].apply(
        lambda x: "Yes" if safe_get(x, "transactions") or safe_get(x, "exchangeOrders") else "No"
    )
    df["days_since_requested"] = df["requested_date"].apply(compute_days_since)

    def classify_tracking_status(row):
        courier_status = row.get("courier_status", "")
        status = row.get("status", "")
        tracking_number = row.get("tracking_number", "")
        
        logger.debug(f"classify_tracking_status: courier_status='{courier_status}', status='{status}', tracking_number='{tracking_number}'")

        if not tracking_number: 
            logger.debug(f"No tracking number, returning 'No tracking number'")
            return "No tracking number"

        # Define substrings of statuses that are not "real" tracking updates
        non_update_substrings = [
            "not found",
            "blocked",
            "unauthorised",
            "rate limited",
            "service error",
            "tracking error",
            "no tracking events",
            "request failed",
            "check failed",
            "status unknown", # From the fallback in get_shipment_status
        ]

        is_real_update = courier_status and not any(sub in courier_status.lower() for sub in non_update_substrings)

        # If we have a scraped courier status, and it's a real update, use it.
        if is_real_update:
            logger.debug(f"Using informative courier_status: {courier_status}")
            return courier_status
        
        # If the status is "approved" and we have a tracking number, but no informative courier status yet,
        # it's most likely just been submitted.
        if status.lower() == "approved":
            logger.debug(f"No informative courier_status, but status is approved. Returning 'Submitted to Courier'")
            return "Submitted to Courier"
        
        # For other RMA statuses (like 'Received'), if we have a non-update status, it might be relevant to show it.
        if courier_status:
            logger.debug(f"Returning the original non-informative courier_status: {courier_status}")
            return courier_status
        
        logger.debug(f"Final fallback, returning '-'")
        return "-"

    df["tracking_status"] = df.apply(classify_tracking_status, axis=1)

    df["DisplayTrack"] = df.apply(
        lambda r: bool(r.get("tracking_number")) and r.get("status", "").lower() in ["approved", "received"],
        axis=1
    )

    df["Is_tracking_updated"] = df["json_data"].apply(has_tracking_update_comment)

    df["is_cc"] = df["tracking_status"].str.lower().str.contains("cancelled", case=False, na=False)
    df["is_fg"] = df.apply(
        lambda r: (r.get("status", "").lower() in ["approved", "received"]) and not r.get("DisplayTrack", False),
        axis=1
    )

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
    
    # DEBUG: Show final tracking_status values
    st.info(f"DEBUG: Final tracking_status values sample: {df['tracking_status'].head(5).tolist()}")
    st.info(f"DEBUG: tracking_status value counts: {df['tracking_status'].value_counts().to_dict()}")

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
        /* Top header with Night-Blue/Sapphire-Red gradient and more sparkles */
        .main-header {
            background: linear-gradient(90deg, 
                #0A1931 0%, /* Night Blue */
                #C70039 100%); /* Sapphire Red */
            padding: 20px;
            text-align: center;
            border-radius: 10px;
            margin-bottom: 10px;
            box-shadow: 0 0 20px rgba(10, 25, 49, 0.7), 0 0 20px rgba(199, 0, 57, 0.7);
            position: relative;
            overflow: hidden;
        }
        
        .main-header::before {
            content: '✨';
            position: absolute; font-size: 12px;
            animation: sparkle 2.5s infinite linear;
            left: 15%; top: 30%;
        }
        
        .main-header::after {
            content: '✨';
            position: absolute; font-size: 14px;
            animation: sparkle 2s infinite linear 0.5s;
            right: 20%; bottom: 25%;
        }
        
        @keyframes sparkle {
            0%, 100% { opacity: 0; }
            50% { opacity: 1; }
        }
        
        .main-header h1 {
            color: #ffffff;
            font-weight: bold; margin: 0;
            text-shadow: 0 0 10px rgba(255, 255, 255, 0.8);
        }
        
        /* Connection status bar */
        .connection-status {
            background: rgba(20, 20, 20, 0.9);
            border: 1px solid rgba(100, 100, 100, 0.3);
            border-radius: 8px;
            padding: 10px;
            text-align: center;
            margin-bottom: 20px;
            color: #00ff00;
            font-size: 14px;
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
        
        /* API Details panel */
        .api-panel {
            background: rgba(30, 30, 30, 0.9);
            border: 1px solid rgba(100, 100, 100, 0.5);
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
        }
        
        .api-panel h3 {
            color: #00ccff;
            margin-bottom: 10px;
            font-size: 16px;
        }
        
        .api-setting {
            margin-bottom: 10px;
            padding: 8px;
            background: rgba(40, 40, 40, 0.6);
            border-radius: 5px;
        }
        
        .api-setting-label {
            color: #aaaaaa;
            font-size: 12px;
            margin-bottom: 3px;
        }
        
        .api-setting-value {
            color: #00ff00;
            font-weight: bold;
            font-size: 14px;
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

    # Top header with blue-red gradient and sparkles
    st.markdown(
        """
        <div class="main-header">
            <h1>LEVI'S RETURNGO OPS DASHBOARD</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Connection status bar
    last_sync_time = st.session_state.get("last_sync_time")
    sync_time_display = (
        last_sync_time.strftime("%Y-%m-%d %H:%M:%S UTC")
        if last_sync_time
        else "N/A"
    )
    st.markdown(
        f"""
        <div class="connection-status">
            🟢 Connected to store: <strong>{STORE_URL}</strong> | Last sync: <strong>{sync_time_display}</strong>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Sidebar
    with st.sidebar:
        st.markdown("<div class='api-panel'>", unsafe_allow_html=True)
        st.markdown("### 🔌 API Details")
        
        # API Settings
        st.markdown("<div class='api-setting'>", unsafe_allow_html=True)
        st.markdown("<div class='api-setting-label'>Store URL</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='api-setting-value'>{STORE_URL}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='api-setting'>", unsafe_allow_html=True)
        st.markdown("<div class='api-setting-label'>API Key Status</div>", unsafe_allow_html=True)
        api_status = "🟢 Active" if MY_API_KEY else "🔴 Missing"
        st.markdown(f"<div class='api-setting-value'>{api_status}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='api-setting'>", unsafe_allow_html=True)
        st.markdown("<div class='api-setting-label'>Rate Limit (RPS)</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='api-setting-value'>{RG_RPS}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='api-setting'>", unsafe_allow_html=True)
        st.markdown("<div class='api-setting-label'>Cache TTL</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='api-setting-value'>{CACHE_EXPIRY_HOURS}h</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='api-setting'>", unsafe_allow_html=True)
        st.markdown("<div class='api-setting-label'>Max Workers</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='api-setting-value'>{MAX_WORKERS}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
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
        if st.button("🔄 Clear Cache & Refresh Data", width="stretch"):
            with st.spinner("Clearing cache and refreshing data..."):
                st.cache_data.clear()
                append_ops_log("Cache cleared. Fetching fresh data.")
            st.success("Cache cleared! Data will refresh.")
            st.rerun()

        st.markdown("---")
        
        # Activity log in terminal-style box
        st.subheader("📋 Activity Log")
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

    # RMA Details section
    st.markdown("#### RMA Details")
    
    # Create metric cards in grid layout
    metric_cols = st.columns(4)
    
    # First row: Total Open, Pending, In Transit, Issues
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
        if st.button("View", key="btn_total_open", width="stretch"):
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
        if st.button("View", key="btn_pending", width="stretch"):
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
        if st.button("View", key="btn_in_transit", width="stretch"):
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
        if st.button("View", key="btn_issues", width="stretch"):
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
        if st.button("Filter", key="btn_pending_req", width="stretch"):
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
        if st.button("Filter", key="btn_received", width="stretch"):
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
        if st.button("Filter", key="btn_courier_cancelled", width="stretch"):
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
        if st.button("Filter", key="btn_submitted", width="stretch"):
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
        if st.button("Filter", key="btn_delivered", width="stretch"):
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
        if st.button("Filter", key="btn_no_tracking", width="stretch"):
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
        if st.button("Filter", key="btn_res_actioned", width="stretch"):
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
        if st.button("Filter", key="btn_no_res_actioned", width="stretch"):
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
        if st.button("📋 View All", width="stretch"):
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
            if st.button("🧼 Clear filters", width="stretch"):
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
    
    # Display columns
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
            lambda x: f"https://app.returngo.ai/dashboard/returns?filter_status=open&rmaid={x}" if x else "-"
        )
        
        # Create tracking links
        display_df["Tracking Number"] = display_df.apply(
            lambda row: f"https://portal.thecourierguy.co.za/track?ref={row['Tracking Number']}" 
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
            display_text=r"rmaid=([^&]*)",
        ),
        "Order": st.column_config.TextColumn("Order"),
        "Current Status": st.column_config.TextColumn("Current Status"),
        "Tracking Number": st.column_config.LinkColumn("Tracking Number", display_text=r"ref=(.*)", help="Click to track on The Courier Guy portal"),
        "Tracking Status": st.column_config.TextColumn("Tracking Status"),
        "Requested date": st.column_config.TextColumn("Requested date"),
        "Approved date": st.column_config.TextColumn("Approved date"),
        "Received date": st.column_config.TextColumn("Received date"),
        "Days since requested": st.column_config.NumberColumn("Days since requested"),
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
        "RMA ID": {"display_text": r"rmaid=([^&]*)"},
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
            elif key == "Days since requested":
                column_config[key] = st.column_config.NumberColumn(
                    key,
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
        if st.button("Data table log", key="btn_data_table_log", width="stretch"):
            st.session_state["suppress_row_dialog"] = True
            show_data_table_log()
    
    with action_col2:
        if st.button("📋 Copy all", key="btn_copy_all", width="stretch"):
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

    sel_event = st.dataframe(
        styled_table,
        use_container_width=False,
        height=700,
        hide_index=True,
        column_config=column_config,
        on_select="rerun", # type: ignore
        selection_mode="single-row",
        key="rma_table",
    )

    sel_rows = sel_event.get("selection", {}).get("rows", []) if isinstance(sel_event, dict) else []
    if sel_rows:
        if not st.session_state.pop("suppress_row_dialog", False):
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
