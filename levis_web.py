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
from typing import Optional, Tuple, Dict, Callable, Set

# ==========================================
# 1. CONFIGURATION
# ==========================================
st.set_page_config(page_title="Levi's ReturnGO Ops", layout="wide", page_icon="📤")

# --- Preserve scroll position across reruns (page scroll) ---
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

# --- ACCESS SECRETS ---
try:
    MY_API_KEY = st.secrets["RETURNGO_API_KEY"]
except (FileNotFoundError, KeyError):
    MY_API_KEY = os.environ.get("RETURNGO_API_KEY")

if not MY_API_KEY:
    st.error("API Key not found! Please set 'RETURNGO_API_KEY' in secrets or env vars.")
    st.stop()

try:
    PARCEL_NINJA_TOKEN = st.secrets["PARCEL_NINJA_TOKEN"]
except Exception:
    PARCEL_NINJA_TOKEN = os.environ.get("PARCEL_NINJA_TOKEN", "")

STORE_URL = "levis-sa.myshopify.com"
DB_FILE = "levis_cache.db"
DB_LOCK = threading.Lock()

# Efficiency controls
CACHE_EXPIRY_HOURS = 24
COURIER_REFRESH_HOURS = 12
MAX_WORKERS = 3
RG_RPS = 2
SYNC_OVERLAP_MINUTES = 5

ACTIVE_STATUSES = ["Pending", "Approved", "Received"]

RATE_LIMIT_HIT = threading.Event()
RATE_LIMIT_INFO = {"remaining": None, "limit": None, "reset": None, "updated_at": None}
RATE_LIMIT_LOCK = threading.Lock()

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
        background: linear-gradient(90deg, rgba(255,255,255,0.92), rgba(255,255,255,0.78));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
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

      /* Cards */
      .card {
        position: relative;
        background: rgba(17, 24, 39, 0.70);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 14px;
        padding: 26px 12px 10px 12px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.25);
      }

      .card.selected {
        border: 1px solid rgba(34,197,94,0.55);
        box-shadow: 0 10px 24px rgba(0,0,0,0.34);
      }

      .card::before {
        content: "";
        position: absolute;
        left: 10px;
        right: 10px;
        top: 10px;
        height: 10px;
        border-radius: 999px;
        background: rgba(148,163,184,0.10);
        border: 1px solid rgba(148,163,184,0.12);
      }

      .card.selected::before {
        background: rgba(34,197,94,0.7);
        border-color: rgba(34,197,94,0.95);
        box-shadow: 0 0 12px rgba(34,197,94,0.7);
      }

      .tile-inner {
        position: relative;
      }

      /* Updated pill */
      .updated-pill {
        position: absolute;
        left: 14px;
        top: 2px;
        font-size: 0.80rem;
        color: rgba(148,163,184,0.95);
        padding: 4px 10px;
        border-radius: 999px;
        background: rgba(15, 23, 42, 0.72);
        border: 1px solid rgba(148, 163, 184, 0.18);
        z-index: 2;
      }

      .card.selected .updated-pill {
        background: rgba(34,197,94,0.18);
        border-color: rgba(34,197,94,0.95);
        color: #bbf7d0;
      }

      .api-box {
        display: flex;
        flex-direction: column;
        gap: 2px;
        align-items: flex-start;
        padding: 8px 10px;
        border-radius: 12px;
        background: rgba(15, 23, 42, 0.72);
        border: 1px solid rgba(148,163,184,0.22);
        color: rgba(226,232,240,0.95);
        font-size: 0.85rem;
        font-weight: 700;
        box-shadow: inset 0 0 0 1px rgba(255,255,255,0.02);
      }
      .api-sub {
        font-size: 0.72rem;
        font-weight: 600;
        color: rgba(148,163,184,0.9);
      }

      .header-right-card {
        background: rgba(17,24,39,0.72);
        border: 1px solid rgba(148,163,184,0.18);
        border-radius: 16px;
        padding: 12px;
        box-shadow: 0 10px 26px rgba(0,0,0,0.25);
      }
      .header-right-card .stButton > button{
        background: rgba(51,65,85,0.60) !important;
      }

      /* Buttons */
      div.stButton > button {
        width: 100%;
        border: 1px solid rgba(148, 163, 184, 0.25) !important;
        background: rgba(31, 41, 55, 0.92) !important;
        color: #e5e7eb !important;
        border-radius: 10px !important;
        padding: 12px 14px !important;
        font-size: 15px !important;
        font-weight: 850 !important;
        transition: 0.15s ease-in-out;
      }
      div.stButton > button:hover {
        border-color: rgba(196,18,48,0.70) !important;
        color: #fff !important;
        transform: translateY(-1px);
      }

      /* Refresh box under each tile: lighter + touches main button */
      .refresh-box {
        margin-top: -2px;           /* touch main button border */
        padding-top: 8px;
        padding-bottom: 6px;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(148,163,184,0.14);
        border-top: 0;
        border-radius: 0 0 12px 12px;
        display: flex;
        justify-content: center;
      }

      .refresh-box div.stButton > button {
        width: auto !important;     /* shrink-to-fit */
        padding: 6px 12px !important;
        font-size: 13px !important;
        border-radius: 10px !important;
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
      }

      /* Smaller Reset Cache button look (lighter) */
      .reset-wrap div.stButton > button {
        background: rgba(148,163,184,0.18) !important;
        border-color: rgba(148,163,184,0.25) !important;
        padding: 9px 12px !important;
        font-size: 14px !important;
      }

    </style>
    """,
    unsafe_allow_html=True,
)

# Sticky anchor + JS
st.markdown('<div class="sticky-anchor"></div>', unsafe_allow_html=True)
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

      applySticky();
      setTimeout(applySticky, 250);
      setTimeout(applySticky, 800);
    </script>
    """,
    height=0,
)

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


def rg_headers() -> dict:
    return {"x-api-key": MY_API_KEY, "x-shop-name": STORE_URL}


def update_rate_limit_info(headers: dict):
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
    session = get_thread_session()
    headers = headers or rg_headers()

    backoff = 1
    res = None
    for _attempt in range(1, 6):
        _sleep_for_rate_limit()
        res = session.request(method, url, headers=headers, timeout=timeout, json=json_body)
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

    return res


# ==========================================
# 3. DATABASE
# ==========================================
def init_db():
    with DB_LOCK:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()

        c.execute(
            """
            CREATE TABLE IF NOT EXISTS rmas (
                rma_id TEXT PRIMARY KEY,
                store_url TEXT,
                status TEXT,
                created_at TEXT,
                json_data TEXT,
                last_fetched TEXT,
                courier_status TEXT
            )
            """
        )

        existing_cols = {row[1] for row in c.execute("PRAGMA table_info(rmas)").fetchall()}

        def add_col(col_name: str, col_type: str):
            if col_name not in existing_cols:
                try:
                    c.execute(f"ALTER TABLE rmas ADD COLUMN {col_name} {col_type}")
                except Exception:
                    pass

        add_col("courier_last_checked", "TEXT")
        add_col("received_first_seen", "TEXT")

        c.execute(
            """
            CREATE TABLE IF NOT EXISTS sync_logs (
                scope TEXT PRIMARY KEY,
                last_sync_iso TEXT
            )
            """
        )

        conn.commit()
        conn.close()


def clear_db():
    with DB_LOCK:
        try:
            if os.path.exists(DB_FILE):
                os.remove(DB_FILE)
            init_db()
            return True
        except Exception:
            return False


def upsert_rma(
    rma_id: str,
    status: str,
    created_at: str,
    payload: dict,
    courier_status: Optional[str] = None,
    courier_checked_iso: Optional[str] = None,
):
    with DB_LOCK:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        now_iso = _iso_utc(_now_utc())

        c.execute(
            "SELECT courier_status, courier_last_checked, received_first_seen FROM rmas WHERE rma_id=?",
            (str(rma_id),),
        )
        row = c.fetchone()
        existing_cstat = row[0] if row else None
        existing_cchk = row[1] if row else None
        existing_received = row[2] if row else None

        if courier_status is None:
            courier_status = existing_cstat
        if courier_checked_iso is None:
            courier_checked_iso = existing_cchk

        received_seen = existing_received
        if status == "Received" and not existing_received:
            received_seen = now_iso

        c.execute(
            """
            INSERT OR REPLACE INTO rmas
            (rma_id, store_url, status, created_at, json_data, last_fetched,
             courier_status, courier_last_checked, received_first_seen)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(rma_id),
                STORE_URL,
                status,
                created_at,
                json.dumps(payload),
                now_iso,
                courier_status,
                courier_checked_iso,
                received_seen,
            ),
        )
        conn.commit()
        conn.close()


def delete_rmas(rma_ids):
    if not rma_ids:
        return
    with DB_LOCK:
           with sqlite3.connect(DB_FILE) as conn:
            conn.executemany(
                "DELETE FROM rmas WHERE rma_id=? AND store_url=?",
                [(str(i), STORE_URL) for i in rma_ids],
            )


def get_rma(rma_id: str):
    with DB_LOCK:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute(
            "SELECT json_data, last_fetched, courier_status, courier_last_checked, received_first_seen "
            "FROM rmas WHERE rma_id=?",
            (str(rma_id),),
        )
        row = c.fetchone()
        conn.close()

    if not row:
        return None

    payload = json.loads(row[0])
    payload["_local_courier_status"] = row[2]
    payload["_local_courier_checked"] = row[3]
    payload["_local_received_first_seen"] = row[4]
    return payload, row[1]


def get_all_open_from_db():
    with DB_LOCK:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        placeholders = ",".join("?" for _ in ACTIVE_STATUSES)
        c.execute(
            f"""
            SELECT json_data, courier_status, courier_last_checked, received_first_seen
            FROM rmas
            WHERE store_url=? AND status IN ({placeholders})
            """,
            (STORE_URL, *ACTIVE_STATUSES),
        )
        rows = c.fetchall()
        conn.close()

    results = []
    for js, cstat, cchk, rcv_seen in rows:
        data = json.loads(js)
        data["_local_courier_status"] = cstat
        data["_local_courier_checked"] = cchk
        data["_local_received_first_seen"] = rcv_seen
        results.append(data)
    return results


def db_mtime() -> float:
    try:
        return os.path.getmtime(DB_FILE)
    except OSError:
        return 0.0


@st.cache_data(show_spinner=False)
def load_open_rmas(_db_mtime: float):
    return get_all_open_from_db()


def get_local_ids_for_status(status: str):
    with DB_LOCK:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT rma_id FROM rmas WHERE store_url=? AND status=?", (STORE_URL, status))
        rows = c.fetchall()
        conn.close()
    return {r[0] for r in rows}


def set_last_sync(scope: str, dt: datetime):
    with DB_LOCK:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute(
            "INSERT OR REPLACE INTO sync_logs (scope, last_sync_iso) VALUES (?, ?)",
            (scope, _iso_utc(dt)),
        )
        conn.commit()
        conn.close()


def get_last_sync(scope: str) -> Optional[datetime]:
    with DB_LOCK:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        try:
            c.execute("SELECT last_sync_iso FROM sync_logs WHERE scope=?", (scope,))
            row = c.fetchone()
            if row and row[0]:
                try:
                    return datetime.fromisoformat(row[0])
                except Exception:
                    return None
            return None
        finally:
            conn.close()


init_db()

# ==========================================
# 4. COURIER STATUS
# ==========================================
def check_courier_status(tracking_number: str) -> str:
    if not tracking_number or not PARCEL_NINJA_TOKEN:
        return "Unknown"
    try:
        url = f"https://optimise.parcelninja.com/shipment/track/{tracking_number}"
        headers = {"User-Agent": "Mozilla/5.0", "Authorization": f"Bearer {PARCEL_NINJA_TOKEN}"}
        session = get_parcel_session()
        res = session.get(url, headers=headers, timeout=10)

        if res.status_code == 200:
            try:
                data = res.json()
                if isinstance(data, dict) and data.get("history"):
                    s0 = data["history"][0]
                    return (s0.get("status") or s0.get("description") or "Unknown") or "Unknown"
                return (data.get("status") or data.get("currentStatus") or "Unknown") or "Unknown"
            except Exception:
                pass

            clean_html = re.sub(r"<(script|style).*?</\1>", "", res.text, flags=re.DOTALL | re.IGNORECASE)
            for kw in ["Courier Cancelled", "Booked Incorrectly", "Delivered", "Out For Delivery"]:
                if re.search(re.escape(kw), clean_html, re.IGNORECASE):
                    return kw
            return "Unknown"

        if res.status_code == 404:
            return "Tracking Not Found"

        return f"Error {res.status_code}"

    except Exception:
        return "Check Failed"


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
        base_url = f"https://api.returngo.ai/rmas?pagesize=500&status={status_param}{updated_filter}"
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
    try:
        last_dt = datetime.fromisoformat(last_fetched_iso)
    except Exception:
        return True
    if last_dt.tzinfo is None:
        last_dt = last_dt.replace(tzinfo=timezone.utc)
    return (_now_utc() - last_dt) > timedelta(hours=CACHE_EXPIRY_HOURS)


def maybe_refresh_courier(rma_payload: dict) -> Tuple[Optional[str], Optional[str]]:
    shipments = rma_payload.get("shipments", []) or []
    track_no = None
    for s in shipments:
        if s.get("trackingNumber"):
            track_no = s.get("trackingNumber")
            break
    if not track_no:
        return None, None

    cached_status = rma_payload.get("_local_courier_status")
    cached_checked = rma_payload.get("_local_courier_checked")

    if cached_checked:
        try:
            last_chk = datetime.fromisoformat(cached_checked)
            if last_chk.tzinfo is None:
                last_chk = last_chk.replace(tzinfo=timezone.utc)
            if (_now_utc() - last_chk) <= timedelta(hours=COURIER_REFRESH_HOURS):
                return cached_status, cached_checked
        except Exception:
            pass

    status = check_courier_status(track_no)
    checked_iso = _iso_utc(_now_utc())
    return status, checked_iso


def fetch_rma_detail(rma_id: str, *, force: bool = False):
    cached = get_rma(rma_id)
    if (not force) and cached and not should_refresh_detail(rma_id):
        return cached[0]

    url = f"https://api.returngo.ai/rma/{rma_id}"
    res = rg_request("GET", url, timeout=20)
    if res.status_code != 200:
        return cached[0] if cached else None

    data = res.json() if res.content else {}
    summary = data.get("rmaSummary", {}) or {}

    if cached:
        data["_local_courier_status"] = cached[0].get("_local_courier_status")
        data["_local_courier_checked"] = cached[0].get("_local_courier_checked")
        data["_local_received_first_seen"] = cached[0].get("_local_received_first_seen")

    courier_status, courier_checked = maybe_refresh_courier(data)

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


def get_incremental_since(statuses, full: bool) -> Optional[datetime]:
    if full:
        return None
    stamps = [get_last_sync(s) for s in statuses]
    stamps = [d for d in stamps if d]
    return min(stamps) if stamps else None


def perform_sync(statuses=None, *, full=False):
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
            local_active_ids |= get_local_ids_for_status(stt)
        stale = local_active_ids - api_ids
        delete_rmas(stale)

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
        bar = st.progress(0, text="Downloading Details...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = [ex.submit(fetch_rma_detail, rid, force=force) for rid in to_fetch]
            done = 0
            for _ in concurrent.futures.as_completed(futures):
                done += 1
                bar.progress(done / total, text=f"Syncing: {done}/{total}")
        bar.empty()

    now = _now_utc()
    scope = ",".join(statuses)
    set_last_sync(scope, now)
    for s in statuses:
        set_last_sync(s, now)

    st.session_state["show_toast"] = True
    status_msg.success("✅ Sync Complete!")
    st.rerun()


def force_refresh_rma_ids(rma_ids, scope_label: str):
    ids = [str(i) for i in set(rma_ids) if i]
    if not ids:
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
    st.session_state["show_toast"] = True
    msg.success("✅ Refresh Complete!")
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
        "trackingURL": f"https://optimise.parcelninja.com/shipment/track/{tracking_number}",
        "labelURL": "https://sellerportal.dpworld.com/api/file-download?link=null",
    }

    try:
        res = rg_request("PUT", f"https://api.returngo.ai/shipment/{shipment_id}", headers=headers, timeout=15, json_body=payload)
        if res.status_code == 200:
            fetch_rma_detail(rma_id, force=True)
            return True, "Success"
        return False, f"API Error {res.status_code}: {res.text}"
    except Exception as e:
        return False, str(e)


def push_comment_update(rma_id, comment_text):
    headers = {**rg_headers(), "Content-Type": "application/json"}
    payload = {"text": comment_text, "isPublic": False}

    try:
        res = rg_request("POST", f"https://api.returngo.ai/rma/{rma_id}/comment", headers=headers, timeout=15, json_body=payload)
        if res.status_code in (200, 201):
            fetch_rma_detail(rma_id, force=True)
            return True, "Success"
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


def pretty_resolution(rt: str) -> str:
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


def parse_yyyy_mm_dd(s: str) -> Optional[datetime]:
    try:
        if not s or s == "N/A":
            return None
        return datetime.fromisoformat(s[:10])
    except Exception:
        return None


def days_since(dt_str: str, today: Optional[datetime.date] = None) -> str:
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
    elif updated_at:
        sub = f"Updated: {updated_at.astimezone().strftime('%H:%M')}"

    return main, sub


# ==========================================
# 8. STATE
# ==========================================
if "active_filters" not in st.session_state:
    st.session_state.active_filters = set()  # type: ignore
if "search_query_input" not in st.session_state:
    st.session_state.search_query_input = ""
if "status_multi" not in st.session_state:
    st.session_state.status_multi = []
if "res_multi" not in st.session_state:
    st.session_state.res_multi = []
if "actioned_multi" not in st.session_state:
    st.session_state.actioned_multi = []
if "req_dates_selected" not in st.session_state:
    st.session_state.req_dates_selected = []

if st.session_state.get("show_toast"):
    st.toast("✅ Updated!", icon="🔄")
    st.session_state["show_toast"] = False

if RATE_LIMIT_HIT.is_set():
    st.warning(
        "ReturnGO rate limit reached (429). Sync is slowing down and retrying. "
        "If this happens often, sync less frequently or request a higher quota key."
    )
    RATE_LIMIT_HIT.clear()


def toggle_filter(name: str):
    s: Set[str] = st.session_state.active_filters  # type: ignore
    if name in s:
        s.remove(name)
    else:
        s.add(name)
    st.session_state.active_filters = s  # type: ignore
    st.rerun()


def clear_all_filters():
    st.session_state.active_filters = set()  # type: ignore
    st.session_state.search_query_input = ""
    st.session_state.status_multi = []
    st.session_state.res_multi = []
    st.session_state.actioned_multi = []
    st.session_state.req_dates_selected = []
    st.rerun()


def updated_pill(scope: str) -> str:
    ts = get_last_sync(scope)
    if not ts:
        return "<span class='updated-pill'>UPDATED: -</span>"
    return f"<span class='updated-pill'>UPDATED: {ts.strftime('%H:%M')}</span>"


# ==========================================
# 10. HEADER (STICKY)
# ==========================================
h1, h2 = st.columns([3, 1], vertical_alignment="top")

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
    st.markdown("<div class='header-right-card'>", unsafe_allow_html=True)
    api_col, sync_col = st.columns([1, 1.4], vertical_alignment="center")
    with api_col:
        api_main, api_sub = format_api_limit_display()
        st.markdown(
            f"""
            <div class="api-box">
              <div>{api_main}</div>
              <div class="api-sub">{api_sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with sync_col:
        if st.button("🔄 Sync Dashboard", key="btn_sync_all", use_container_width=True):
            perform_sync()
    st.markdown("<div class='reset-wrap'>", unsafe_allow_html=True)
    if st.button("🗑️ Reset Cache", key="btn_reset", use_container_width=True):
        if clear_db():
            st.success("Cache cleared!")
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.write("")

# ==========================================
# 11. LOAD + PROCESS OPEN RMAs
# ==========================================
raw_data = load_open_rmas(db_mtime())
processed_rows = []

counts = {
    "Pending": 0,
    "Approved": 0,
    "Received": 0,
    "NoTracking": 0,
    "Flagged": 0,
    "CourierCancelled": 0,
    "ApprovedDelivered": 0,
    "ResolutionActioned": 0,
    "NoResolutionActioned": 0,
}

search_query = st.session_state.get("search_query_input", "")
search_query_lower = search_query.lower().strip()
search_active = bool(search_query_lower)
today = datetime.now().date()

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

    local_tracking_status = rma.get("_local_courier_status", "") or ""
    local_tracking_status_lower = local_tracking_status.lower()
    track_link_url = f"https://portal.thecourierguy.co.za/track?ref={track_nums[0]}" if track_nums else ""

    requested_iso = get_event_date_iso(rma, "RMA_CREATED") or (summary.get("createdAt") or "")
    approved_iso = get_event_date_iso(rma, "RMA_APPROVED")
    received_iso = get_received_date_iso(rma)
    resolution_type = get_resolution_type(rma)

    comment_texts = [(c.get("htmlText", "") or "").lower() for c in comments]
    is_cc = "courier cancelled" in local_tracking_status_lower
    is_ad = status == "Approved" and "delivered" in local_tracking_status_lower
    is_cc = bool(local_tracking_status) and ("courier cancelled" in local_tracking_status_lower)
    is_ad = (status == "Approved") and (bool(local_tracking_status) and ("delivered" in local_tracking_status_lower))
    actioned = is_resolution_actioned(rma, comment_texts)
    actioned_label = resolution_actioned_label(rma)
    is_ra = actioned
    is_nra = (status == "Received") and (not actioned)

    if status in ("Pending", "Approved", "Received"):
        counts[status] += 1
    if is_nt:
        counts["NoTracking"] += 1
    if is_fg:
        counts["Flagged"] += 1
    if is_cc:
        counts["CourierCancelled"] += 1
    if is_ad:
        counts["ApprovedDelivered"] += 1
    if is_ra:
        counts["ResolutionActioned"] += 1
    if is_nra:
        counts["NoResolutionActioned"] += 1

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
                "Tracking Status": local_tracking_status,
                "Requested date": str(requested_iso)[:10] if requested_iso else "N/A",
                "Approved date": str(approved_iso)[:10] if approved_iso else "N/A",
                "Received date": str(received_iso)[:10] if received_iso else "N/A",
                "Days since requested": days_since(str(requested_iso)[:10] if requested_iso else "N/A", today=today),
                "resolutionType": resolution_type if resolution_type else "N/A",
                "Resolution actioned": actioned_label,
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

# ==========================================
# 12. FILTER DEFINITIONS (multi-select)
# ==========================================
FilterFn = Callable[[pd.DataFrame], pd.Series]

FILTERS: Dict[str, Dict[str, object]] = {
    "Pending": {"icon": "⏳", "count_key": "Pending", "scope": "Pending", "fn": lambda d: d["Current Status"] == "Pending"},
    "Approved": {"icon": "✅", "count_key": "Approved", "scope": "Approved", "fn": lambda d: d["Current Status"] == "Approved"},
    "Received": {"icon": "📦", "count_key": "Received", "scope": "Received", "fn": lambda d: d["Current Status"] == "Received"},
    "No Tracking": {"icon": "🚫", "count_key": "NoTracking", "scope": "FILTER_NoTracking", "fn": lambda d: d["is_nt"] == True},
    "Flagged": {"icon": "🚩", "count_key": "Flagged", "scope": "FILTER_Flagged", "fn": lambda d: d["is_fg"] == True},
    "Courier Cancelled": {"icon": "🛑", "count_key": "CourierCancelled", "scope": "FILTER_CourierCancelled", "fn": lambda d: d["is_cc"] == True},
    "Approved > Delivered": {"icon": "📬", "count_key": "ApprovedDelivered", "scope": "FILTER_ApprovedDelivered", "fn": lambda d: d["is_ad"] == True},
    "Resolution Actioned": {"icon": "💳", "count_key": "ResolutionActioned", "scope": "FILTER_ResolutionActioned", "fn": lambda d: d["is_ra"] == True},
    "No Resolution Actioned": {"icon": "⏸️", "count_key": "NoResolutionActioned", "scope": "FILTER_NoResolutionActioned", "fn": lambda d: d["is_nra"] == True},
}

def current_filter_mask(df: pd.DataFrame) -> pd.Series:
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
        fn: FilterFn = cfg["fn"]  # type: ignore
        masks.append(fn(df))

    if not masks:
        return pd.Series([True] * len(df), index=df.index)

    out = masks[0].copy()
    for m in masks[1:]:
        out = out | m
    return out


def ids_for_filter(name: str) -> list:
    if df_view.empty:
        return []
    cfg = FILTERS.get(name)
    if not cfg:
        return []
    fn: FilterFn = cfg["fn"]  # type: ignore
    mask = fn(df_view)
    if mask is None or mask.empty:
        return []
    return df_view.loc[mask, "_rma_id_text"].astype(str).tolist()


# ==========================================
# 13. FILTER TILE UI (selected border highlight)
# ==========================================
def render_filter_tile(col, name: str, refresh_scope: str):
    cfg = FILTERS[name]
    icon = cfg["icon"]  # type: ignore
    count_key = cfg["count_key"]  # type: ignore
    scope = cfg["scope"]  # type: ignore

    active: Set[str] = st.session_state.active_filters  # type: ignore
    selected = (name in active)

    count_val = counts.get(count_key, 0)

    # Selection indicator: card border highlight + top bar
    label = f"{icon} {name.upper()} [**{count_val}**]"

    with col:
        card_class = "card selected" if selected else "card"
        st.markdown(f"<div class='{card_class}'><div class='tile-inner'>", unsafe_allow_html=True)
        st.markdown(updated_pill(str(scope)), unsafe_allow_html=True)

        if st.button(label, key=f"flt_{name}", use_container_width=True):
            toggle_filter(name)

        st.markdown("<div class='refresh-box'>", unsafe_allow_html=True)
        if st.button("🔄 Refresh", key=f"ref_{name}", use_container_width=False):
            force_refresh_rma_ids(ids_for_filter(name), refresh_scope)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div></div>", unsafe_allow_html=True)


# Row 1
r1 = st.columns(5)
render_filter_tile(r1[0], "Pending", "Pending")
render_filter_tile(r1[1], "Approved", "Approved")
render_filter_tile(r1[2], "Received", "Received")
render_filter_tile(r1[3], "No Tracking", "FILTER_NoTracking")
render_filter_tile(r1[4], "Flagged", "FILTER_Flagged")

st.write("")

# Row 2
r2 = st.columns(4)
render_filter_tile(r2[0], "Courier Cancelled", "FILTER_CourierCancelled")
render_filter_tile(r2[1], "Approved > Delivered", "FILTER_ApprovedDelivered")
render_filter_tile(r2[2], "Resolution Actioned", "FILTER_ResolutionActioned")
render_filter_tile(r2[3], "No Resolution Actioned", "FILTER_NoResolutionActioned")

st.write("")

# ==========================================
# 14. SEARCH + VIEW ALL + FILTER BAR
# ==========================================
st.write("")
sc1, sc2, sc3 = st.columns([8, 1, 1], vertical_alignment="center")
with sc1:
    st.text_input(
        "Search",
        placeholder="🔍 Search Order, RMA, or Tracking...",
        label_visibility="collapsed",
        key="search_query_input",
    )
with sc2:
    if st.button("🧹 Clear", use_container_width=True):
        clear_all_filters()
with sc3:
    if st.button("📋 View All", use_container_width=True):
        st.session_state.active_filters = set()  # type: ignore
        st.rerun()

# Extra filter bar/drop (under search)
with st.expander("Additional filters", expanded=True):
    c1, c2, c3, c4, c5 = st.columns([2, 2, 2, 3, 1], vertical_alignment="center")

    def multi_select_with_state(label: str, options: list, key: str):
        old = st.session_state.get(key, [])
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
        req_dates = []
        if not df_view.empty and "Requested date" in df_view.columns:
            req_dates = sorted(
                [d for d in df_view["Requested date"].dropna().astype(str).unique().tolist() if d and d != "N/A"],
                reverse=True,
            )
        multi_select_with_state("Requested date (multi-select)", req_dates, "req_dates_selected")
    with c5:
        if st.button("🧼 Clear filters", use_container_width=True):
            clear_all_filters()

st.divider()

# ==========================================
# 15. TABLE VIEW
# ==========================================
if df_view.empty:
    st.warning("Database empty. Click Sync Dashboard to start.")
    st.stop()

display_df = df_view[current_filter_mask(df_view)].copy()

# Apply extra filters (AND logic)
status_multi = st.session_state.get("status_multi", [])
res_multi = st.session_state.get("res_multi", [])
actioned_multi = st.session_state.get("actioned_multi", [])
req_dates_selected = st.session_state.get("req_dates_selected", [])

if status_multi:
    display_df = display_df[display_df["Current Status"].isin(status_multi)]
if res_multi:
    display_df = display_df[display_df["resolutionType"].isin(res_multi)]
if actioned_multi:
    actioned_set = set(actioned_multi)
    if actioned_set == {"Yes"}:
        display_df = display_df[display_df["Resolution actioned"] != "No"]
    elif actioned_set == {"No"}:
        display_df = display_df[display_df["Resolution actioned"] == "No"]
if req_dates_selected:
    display_df = display_df[display_df["Requested date"].isin(req_dates_selected)]

if display_df.empty:
    st.info("No matching records found.")
    st.stop()

display_df = display_df.sort_values(by="Requested date", ascending=False).reset_index(drop=True)
display_df["No"] = (display_df.index + 1).astype(str)

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
                    ok, msg = push_tracking_update(rma_id_text, shipment_id, new_track.strip())
                    if ok:
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

        full = row.get("full_data", {}) or {}
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
]

column_config = {
    "No": st.column_config.TextColumn("No", width="small"),
    "RMA ID": st.column_config.LinkColumn(
        "RMA ID",
        display_text=r"rmaid=([^&]+)",
        width="small",
    ),
    "Order": st.column_config.TextColumn("Order", width="medium"),
    "Current Status": st.column_config.TextColumn("Current Status", width="small"),
    "Tracking Number": st.column_config.LinkColumn("Tracking Number", display_text=r"ref=(.*)", width="medium"),
    "Tracking Status": st.column_config.TextColumn("Tracking Status", width="large"),
    "Requested date": st.column_config.TextColumn("Requested date", width="small"),
    "Approved date": st.column_config.TextColumn("Approved date", width="small"),
    "Received date": st.column_config.TextColumn("Received date", width="small"),
    "Days since requested": st.column_config.TextColumn("Days since requested", width="small"),
    "resolutionType": st.column_config.TextColumn("resolutionType", width="medium"),
    "Resolution actioned": st.column_config.TextColumn("Resolution actioned", width="medium"),
}

_table_df = display_df[display_cols + ["_rma_id_text", "DisplayTrack", "shipment_id", "full_data"]].copy()

sel_event = st.dataframe(
    _table_df[display_cols],
    use_container_width=True,
    height=700,
    hide_index=True,
    column_config=column_config,
    on_select="rerun",
    selection_mode="single-row",
)

sel_rows = (sel_event.selection.rows if sel_event and hasattr(sel_event, "selection") else []) or []
if sel_rows:
    idx = int(sel_rows[0])
    show_rma_actions_dialog(display_df.iloc[idx])
