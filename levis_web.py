
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
from typing import Optional, Tuple

# ==========================================
# 1. CONFIGURATION
# ==========================================
st.set_page_config(page_title="Levi's ReturnGO Ops", layout="wide", page_icon="📤")

# --- Preserve scroll position across reruns (prevents page jump after dialogs close) ---
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

# IMPORTANT: move this token into secrets/env if you can.
try:
    PARCEL_NINJA_TOKEN = st.secrets["PARCEL_NINJA_TOKEN"]
except Exception:
    PARCEL_NINJA_TOKEN = os.environ.get("PARCEL_NINJA_TOKEN", "")

STORE_URL = "levis-sa.myshopify.com"
DB_FILE = "levis_cache.db"
DB_LOCK = threading.Lock()

# Efficiency controls
CACHE_EXPIRY_HOURS = 4           # don't refetch ReturnGO detail if cached within this window
COURIER_REFRESH_HOURS = 12       # don't re-check courier tracking too often
MAX_WORKERS = 3                  # ReturnGO docs recommend minimizing concurrency
RG_RPS = 2                       # soft rate-limit (requests per second) to avoid 429
SYNC_OVERLAP_MINUTES = 5         # pull a small overlap on incremental sync

ACTIVE_STATUSES = ["Pending", "Approved", "Received"]

# Thread-safe UI signal for 429s (never call st.* from worker threads)
RATE_LIMIT_HIT = threading.Event()

# ==========================================
# 1b. STYLING (modernised)
# ==========================================
st.markdown(
    """
    <style>
      /* App background */
      .stApp {
        background: radial-gradient(1200px 600px at 20% 0%, rgba(196,18,48,0.08), transparent 60%),
                    radial-gradient(900px 500px at 90% 10%, rgba(59,130,246,0.08), transparent 55%),
                    #0e1117;
        color: #e5e7eb;
      }

      /* Header spacing */
      .app-subtitle {
        color: #cbd5e1;
        font-size: 0.95rem;
        margin-top: -0.35rem;
      }

      /* Card containers */
      .card {
        background: rgba(17, 24, 39, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 14px;
        padding: 14px 14px 10px 14px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.25);
      }

      .pill-row {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-bottom: 10px;
      }

      /* Streamlit buttons */
      div.stButton > button {
        width: 100%;
        border: 1px solid rgba(148, 163, 184, 0.25) !important;
        background: rgba(31, 41, 55, 0.9) !important;
        color: #e5e7eb !important;
        border-radius: 10px !important;
        padding: 12px 14px !important;
        font-size: 15px !important;
        font-weight: 700 !important;
        transition: 0.15s ease-in-out;
      }
      div.stButton > button:hover {
        border-color: rgba(196,18,48,0.7) !important;
        color: #fff !important;
        transform: translateY(-1px);
      }
      /* Smaller secondary buttons (sync) */
      .small-btn div.stButton > button {
        padding: 9px 12px !important;
        font-size: 13px !important;
        font-weight: 700 !important;
        border-radius: 10px !important;
      }

      .sync-time {
        font-size: 0.8em;
        color: rgba(148,163,184,0.9);
        text-align: center;
        margin-top: 8px;
      }

      /* Data editor tweaks */
      [data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid rgba(148,163,184,0.18);
      }

      /* Dialog box styling */
      div[data-testid="stDialog"] {
        background-color: rgba(17, 24, 39, 0.95);
        border: 1px solid rgba(196,18,48,0.55);
        border-radius: 16px;
      }
    </style>
    """,
    unsafe_allow_html=True,
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
    """Soft limiter to spread requests and reduce 429s."""
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
    """One session per thread (requests.Session is not guaranteed thread-safe)."""
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


def rg_headers() -> dict:
    return {"x-api-key": MY_API_KEY, "x-shop-name": STORE_URL}


def rg_request(method: str, url: str, *, headers=None, timeout=15, json_body=None):
    """ReturnGO request wrapper with pacing and 429-aware backoff."""
    session = get_thread_session()
    headers = headers or rg_headers()

    backoff = 1
    for attempt in range(1, 6):
        _sleep_for_rate_limit()
        res = session.request(method, url, headers=headers, timeout=timeout, json=json_body)

        if res.status_code != 429:
            return res

        # 429 encountered: signal UI (thread-safe)
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

        # 1) Ensure rmas exists (base columns)
        c.execute("""
            CREATE TABLE IF NOT EXISTS rmas (
                rma_id TEXT PRIMARY KEY,
                store_url TEXT,
                status TEXT,
                created_at TEXT,
                json_data TEXT,
                last_fetched TEXT,
                courier_status TEXT
            )
        """)

        # 2) Migrate rmas: add missing columns (safe to run repeatedly)
        existing_cols = {row[1] for row in c.execute("PRAGMA table_info(rmas)").fetchall()}

        def add_col(col_name: str, col_type: str):
            if col_name not in existing_cols:
                try:
                    c.execute(f"ALTER TABLE rmas ADD COLUMN {col_name} {col_type}")
                except Exception:
                    pass

        add_col("courier_last_checked", "TEXT")
        add_col("received_first_seen", "TEXT")

        # 3) Ensure sync_logs exists with new schema
        c.execute("""
            CREATE TABLE IF NOT EXISTS sync_logs (
                scope TEXT PRIMARY KEY,
                last_sync_iso TEXT
            )
        """)

        # 4) Migrate old sync_logs schema if it exists
        sync_cols = {row[1] for row in c.execute("PRAGMA table_info(sync_logs)").fetchall()}

        # If it somehow exists in old format (status/last_sync), rebuild as new
        if ("status" in sync_cols) or ("last_sync" in sync_cols):
            try:
                c.execute("""
                    CREATE TABLE IF NOT EXISTS sync_logs_new (
                        scope TEXT PRIMARY KEY,
                        last_sync_iso TEXT
                    )
                """)
                c.execute("""
                    INSERT OR REPLACE INTO sync_logs_new (scope, last_sync_iso)
                    SELECT status, last_sync FROM sync_logs
                """)
                c.execute("DROP TABLE sync_logs")
                c.execute("ALTER TABLE sync_logs_new RENAME TO sync_logs")
            except Exception:
                pass

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

        # Preserve courier fields + received_first_seen unless overridden / set
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
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.executemany(
            "DELETE FROM rmas WHERE rma_id=? AND store_url=?",
            [(str(i), STORE_URL) for i in rma_ids],
        )
        conn.commit()
        conn.close()


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
    """Open = Pending/Approved/Received only."""
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
        except sqlite3.OperationalError:
            # Fallback for older schema if it somehow persists
            try:
                c.execute("SELECT last_sync FROM sync_logs WHERE status=?", (scope,))
                row = c.fetchone()
                if row and row[0]:
                    try:
                        return datetime.fromisoformat(row[0])
                    except Exception:
                        return None
                return None
            except Exception:
                return None
        finally:
            conn.close()


init_db()

# ==========================================
# 4. PARCEL NINJA COURIER STATUS (cached)
# ==========================================

def check_courier_status(tracking_number: str) -> str:
    if not tracking_number or not PARCEL_NINJA_TOKEN:
        return "Unknown"

    try:
        url = f"https://optimise.parcelninja.com/shipment/track/{tracking_number}"
        headers = {"User-Agent": "Mozilla/5.0", "Authorization": f"Bearer {PARCEL_NINJA_TOKEN}"}
        res = requests.get(url, headers=headers, timeout=10)
        final_status = "Unknown"

        if res.status_code == 200:
            # JSON first
            try:
                data = res.json()
                if isinstance(data, dict) and data.get("history"):
                    final_status = data["history"][0].get("status") or data["history"][0].get("description")
                else:
                    final_status = data.get("status") or data.get("currentStatus") or "Unknown"
                return final_status or "Unknown"
            except Exception:
                pass

            # HTML fallback
            content = res.text
            clean_html = re.sub(r"<(script|style).*?</\1>", "", content, flags=re.DOTALL | re.IGNORECASE)
            history_section = re.search(r"<table[^>]*?tracking-history.*?>(.*?)</table>", clean_html, re.DOTALL | re.IGNORECASE)
            content_to_parse = history_section.group(1) if history_section else clean_html
            rows = re.findall(r"<tr[^>]*>(.*?)</tr>", content_to_parse, re.DOTALL | re.IGNORECASE)

            for r_html in rows:
                if "<th" in r_html.lower():
                    continue
                cells = re.findall(r"<td[^>]*>(.*?)</td>", r_html, re.DOTALL | re.IGNORECASE)
                if cells:
                    cleaned = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
                    cleaned = [c for c in cleaned if c]
                    if cleaned:
                        return re.sub(r"\s+", " ", " - ".join(cleaned)).strip()

            for kw in ["Courier Cancelled", "Booked Incorrectly", "Delivered", "Out For Delivery"]:
                if re.search(re.escape(kw), clean_html, re.IGNORECASE):
                    return kw

        if res.status_code == 404:
            return "Tracking Not Found"

        return f"Error {res.status_code}"

    except Exception:
        return "Check Failed"


# ==========================================
# 5. RETURNGO FETCHING (incremental + cached)
# ==========================================

def fetch_rma_list(statuses, since_dt: Optional[datetime]):
    """Fetch RMA summaries for one or many statuses using cursor pagination."""
    all_summaries = []
    cursor = None
    status_param = ",".join(statuses)

    updated_filter = ""
    if since_dt is not None:
        since_dt = since_dt - timedelta(minutes=SYNC_OVERLAP_MINUTES)
        updated_filter = f"&rma_updated_at=gte:{_iso_utc(since_dt)}"

    while True:
        base_url = f"https://api.returngo.ai/rmas?pagesize=500&status={status_param}{updated_filter}"
        url = f"{base_url}&cursor={cursor}" if cursor else base_url

        res = rg_request("GET", url, timeout=20)
        if res.status_code != 200:
            break

        data = res.json() if res.content else {}
        rmas = data.get("rmas", []) or []
        if not rmas:
            break

        all_summaries.extend(rmas)
        cursor = data.get("next_cursor")
        if not cursor:
            break

    return all_summaries


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


def fetch_rma_detail(rma_id: str):
    cached = get_rma(rma_id)
    if cached and not should_refresh_detail(rma_id):
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

    # refresh local fields for UI
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
    summaries = fetch_rma_list(statuses, since_dt)
    list_bar.progress(1.0, text=f"Fetched {len(summaries)} RMAs")
    time.sleep(0.15)
    list_bar.empty()

    api_ids = {s.get("rmaId") for s in summaries if s.get("rmaId")}

    # tidy cache for open RMAs
    if set(statuses) == set(ACTIVE_STATUSES) and not full:
        local_active_ids = set()
        for stt in ACTIVE_STATUSES:
            local_active_ids |= get_local_ids_for_status(stt)
        stale = local_active_ids - api_ids
        delete_rmas(stale)

    to_fetch = [rid for rid in api_ids if should_refresh_detail(rid)]
    total = len(to_fetch)
    status_msg.info(f"⏳ Syncing {total} records...")

    if total > 0:
        bar = st.progress(0, text="Downloading Details...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = [ex.submit(fetch_rma_detail, rid) for rid in to_fetch]
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

    # Mark "FULL" only when full rebuild is done
    if full and set(statuses) == set(ACTIVE_STATUSES):
        set_last_sync("FULL", now)

    st.session_state["last_sync"] = now.strftime("%Y-%m-%d %H:%M")
    st.session_state["show_toast"] = True

    status_msg.success("✅ Sync Complete!")
    st.rerun()


# ==========================================
# 6. MUTATIONS (Tracking + Comments)
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
            fresh = fetch_rma_detail(rma_id)
            return True, "Success" if fresh else "Updated"
        return False, f"API Error {res.status_code}: {res.text}"
    except Exception as e:
        return False, str(e)


def push_comment_update(rma_id, comment_text):
    headers = {**rg_headers(), "Content-Type": "application/json"}
    payload = {"text": comment_text, "isPublic": False}

    try:
        res = rg_request("POST", f"https://api.returngo.ai/rma/{rma_id}/comment", headers=headers, timeout=15, json_body=payload)
        if res.status_code in (200, 201):
            fresh = fetch_rma_detail(rma_id)
            return True, "Success" if fresh else "Posted"
        return False, f"API Error {res.status_code}: {res.text}"
    except Exception as e:
        return False, str(e)


# ==========================================
# 7. Helpers for requested/approved/received dates + resolutionType
# ==========================================

def get_event_date_iso(rma_payload: dict, event_name: str) -> str:
    summary = rma_payload.get("rmaSummary", {}) or {}
    for e in (summary.get("events") or []):
        if e.get("eventName") == event_name and e.get("eventDate"):
            return str(e["eventDate"])
    return ""


def get_received_date_iso(rma_payload: dict) -> str:
    # Try common event names if present, otherwise fall back to our first-seen timestamp
    for name in ("SHIPMENT_RECEIVED", "RMA_RECEIVED", "RMA_STATUS_RECEIVED"):
        dt = get_event_date_iso(rma_payload, name)
        if dt:
            return dt
    # fallback to locally stored first-seen received timestamp
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


# ==========================================
# 8. FRONTEND UI
# ==========================================

# Modal state
if "modal_rma" not in st.session_state:
    st.session_state.modal_rma = None
if "modal_action" not in st.session_state:
    st.session_state.modal_action = None
if "table_key" not in st.session_state:
    st.session_state.table_key = 0

# Default view = all open RMAs
if "filter_status" not in st.session_state:
    st.session_state.filter_status = "AllOpen"

# Global toast
if st.session_state.get("show_toast"):
    st.toast("✅ API Sync Complete!", icon="🔄")
    st.session_state["show_toast"] = False

# Show 429 warning (thread-safe)
if RATE_LIMIT_HIT.is_set():
    st.warning(
        "ReturnGO rate limit reached (429). Sync is slowing down and retrying. "
        "If this happens often, sync less frequently or request a higher quota key."
    )
    RATE_LIMIT_HIT.clear()


def set_filter(f):
    st.session_state.filter_status = f
    st.rerun()


@st.dialog("Update Tracking")
def show_update_tracking_dialog(record):
    st.markdown(f"### Update Tracking for `{record['RMA ID']}`")
    with st.form("upd_track"):
        new_track = st.text_input("New Tracking Number", value=record["DisplayTrack"])
        if st.form_submit_button("Save Changes"):
            if not record["shipment_id"]:
                st.error("No Shipment ID.")
            else:
                ok, msg = push_tracking_update(record["RMA ID"], record["shipment_id"], new_track)
                if ok:
                    st.success("Updated!")
                    time.sleep(0.35)
                    st.rerun()
                else:
                    st.error(msg)


@st.dialog("View Timeline")
def show_timeline_dialog(record):
    st.markdown(f"### Timeline for `{record['RMA ID']}`")

    with st.expander("➕ Add Comment", expanded=False):
        with st.form("add_comm"):
            comment_text = st.text_area("New Note")
            if st.form_submit_button("Post Comment"):
                ok, msg = push_comment_update(record["RMA ID"], comment_text)
                if ok:
                    st.success("Posted!")
                    st.rerun()
                else:
                    st.error(msg)

    full = record["full_data"]
    timeline = full.get("comments", []) or []
    if not timeline:
        st.info("No timeline events found.")
    else:
        for t in timeline:
            d_str = (t.get("datetime", "") or "")[:16].replace("T", " ")
            st.markdown(f"**{d_str}** | `{t.get('triggeredBy', 'System')}`\n> {t.get('htmlText', '')}")
            st.divider()


# Handle modal trigger after rerun
if st.session_state.modal_rma is not None:
    current_rma = st.session_state.modal_rma
    current_act = st.session_state.modal_action
    st.session_state.modal_rma = None
    st.session_state.modal_action = None
    if current_act == "edit":
        show_update_tracking_dialog(current_rma)
    elif current_act == "view":
        show_timeline_dialog(current_rma)

# --------------------------
# Header
# --------------------------
col1, col2 = st.columns([3, 1], vertical_alignment="top")

with col1:
    st.title("Levi's ReturnGO Ops Dashboard")
    last_full = get_last_sync("FULL")
    last_full_str = last_full.strftime("%Y-%m-%d %H:%M") if last_full else "N/A"
    st.markdown(
        f"<div class='app-subtitle'><b>CONNECTED TO:</b> {STORE_URL.upper()} | "
        f"<b>LAST FULL SYNC:</b> <span style='color:#34d399'>{last_full_str}</span></div>",
        unsafe_allow_html=True,
    )

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    if st.button("🔄 Sync All Data", key="btn_sync_all", use_container_width=True):
        perform_sync()
    if st.button("🧾 Full Rebuild (Slow)", key="btn_full_rebuild", use_container_width=True,
                 help="Ignores incremental sync and rebuilds cache (more API calls)"):
        perform_sync(full=True)
    if st.button("🗑️ Reset Cache", key="btn_reset", use_container_width=True):
        if clear_db():
            st.success("Cache cleared!")
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

st.write("")

# --------------------------
# Load + process cached open RMAs
# --------------------------
raw_data = get_all_open_from_db()
processed_rows = []

counts = {"Pending": 0, "Approved": 0, "Received": 0, "NoTrack": 0, "Flagged": 0}

search_query = st.session_state.get("search_query_input", "")

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
    track_link_url = f"https://portal.thecourierguy.co.za/track?ref={track_nums[0]}" if track_nums else ""

    requested_iso = get_event_date_iso(rma, "RMA_CREATED") or (summary.get("createdAt") or "")
    approved_iso = get_event_date_iso(rma, "RMA_APPROVED")
    received_iso = get_received_date_iso(rma)
    resolution = get_resolution_type(rma)

    # Counts
    if status in counts:
        counts[status] += 1
    is_nt = (status == "Approved" and not track_str)
    is_fg = any("flagged" in (c.get("htmlText", "").lower()) for c in comments)
    if is_nt:
        counts["NoTrack"] += 1
    if is_fg:
        counts["Flagged"] += 1

    # Search filter
    if not search_query or (
        search_query.lower() in str(rma_id).lower()
        or search_query.lower() in str(order_name).lower()
        or search_query.lower() in str(track_str).lower()
    ):
        processed_rows.append(
            {
                "No": "",
                "RMA ID": rma_id,
                "Order": order_name,
                "Current Status": status,
                "Tracking Number": track_link_url,
                "Tracking Status": local_tracking_status,
                "Requested date": str(requested_iso)[:10] if requested_iso else "N/A",
                "Approved date": str(approved_iso)[:10] if approved_iso else "N/A",
                "Received date": str(received_iso)[:10] if received_iso else "N/A",
                "resolutionType": resolution if resolution else "N/A",
                "Update Tracking Number": False,
                "View Timeline": False,
                "DisplayTrack": track_str,
                "shipment_id": shipment_id,
                "full_data": rma,
                "is_nt": is_nt,
                "is_fg": is_fg,
            }
        )

df_view = pd.DataFrame(processed_rows)

# --------------------------
# Metrics row (modern cards + sync button underneath)
# --------------------------
b1, b2, b3, b4, b5 = st.columns(5)

def get_status_time_html(s: str) -> str:
    try:
        ts = get_last_sync(s)
        if not ts:
            return "<div class='sync-time'>UPDATED: -</div>"
        return f"<div class='sync-time'>UPDATED: {ts.strftime('%H:%M')}</div>"
    except Exception:
        return "<div class='sync-time'>UPDATED: -</div>"

with b1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    if st.button(f"PENDING\n{counts['Pending']}", key="btn_p", use_container_width=True):
        set_filter("Pending")
    st.markdown("<div class='small-btn'>", unsafe_allow_html=True)
    if st.button("🔄 Sync Pending", key="sync_p", use_container_width=True):
        perform_sync(["Pending"])
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown(get_status_time_html("Pending"), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with b2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    if st.button(f"APPROVED\n{counts['Approved']}", key="btn_a", use_container_width=True):
        set_filter("Approved")
    st.markdown("<div class='small-btn'>", unsafe_allow_html=True)
    if st.button("🔄 Sync Approved", key="sync_a", use_container_width=True):
        perform_sync(["Approved"])
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown(get_status_time_html("Approved"), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with b3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    if st.button(f"RECEIVED\n{counts['Received']}", key="btn_r", use_container_width=True):
        set_filter("Received")
    st.markdown("<div class='small-btn'>", unsafe_allow_html=True)
    if st.button("🔄 Sync Received", key="sync_r", use_container_width=True):
        perform_sync(["Received"])
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown(get_status_time_html("Received"), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with b4:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    if st.button(f"NO TRACKING\n{counts['NoTrack']}", key="btn_nt", use_container_width=True):
        set_filter("NoTrack")
    st.markdown("<div class='sync-time'>Filter</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with b5:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    if st.button(f"🚩 FLAGGED\n{counts['Flagged']}", key="btn_fg", use_container_width=True):
        set_filter("Flagged")
    st.markdown("<div class='sync-time'>Filter</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.write("")

# --------------------------
# Filters + List All Open RMAs + Search
# --------------------------
fc1, fc2, fc3, fc4 = st.columns([1.25, 1.55, 1.3, 5.9], vertical_alignment="center")

with fc1:
    if st.button("📋 List All Open RMAs", key="btn_all_open", use_container_width=True):
        st.session_state.filter_status = "AllOpen"
        st.rerun()

with fc2:
    if st.button("Courier Cancelled", key="btn_filter_cc", use_container_width=True):
        st.session_state.filter_status = "CourierCancelled"
        st.rerun()

with fc3:
    if st.button("Approved > Delivered", key="btn_filter_ad", use_container_width=True):
        st.session_state.filter_status = "ApprovedDelivered"
        st.rerun()

with fc4:
    # Search + clear
    sc1, sc2 = st.columns([10, 1], vertical_alignment="center")
    with sc1:
        st.text_input(
            "Search",
            placeholder="🔍 Search Order, RMA, or Tracking...",
            label_visibility="collapsed",
            key="search_query_input",
        )
    with sc2:
        def clear_search_cb():
            st.session_state.search_query_input = ""
        st.button("❌", help="Clear Search", key="clear_search_btn", on_click=clear_search_cb, use_container_width=True)

st.divider()

# --------------------------
# Table view
# --------------------------
if df_view.empty:
    st.warning("Database empty. Click Sync All Data to start.")
    st.stop()

f_stat = st.session_state.filter_status

display_df = df_view

if f_stat == "Pending":
    display_df = df_view[df_view["Current Status"] == "Pending"]
elif f_stat == "Approved":
    display_df = df_view[df_view["Current Status"] == "Approved"]
elif f_stat == "Received":
    display_df = df_view[df_view["Current Status"] == "Received"]
elif f_stat == "NoTrack":
    display_df = df_view[df_view["is_nt"] == True]
elif f_stat == "Flagged":
    display_df = df_view[df_view["is_fg"] == True]
elif f_stat == "CourierCancelled":
    display_df = df_view[df_view["Tracking Status"].str.contains("Courier Cancelled", case=False, na=False)]
elif f_stat == "ApprovedDelivered":
    display_df = df_view[
        (df_view["Current Status"] == "Approved")
        & (df_view["Tracking Status"].str.contains("Delivered", case=False, na=False))
    ]
elif f_stat == "AllOpen":
    display_df = df_view

if display_df.empty:
    st.info("No matching records found in cache.")
    st.stop()

display_df = display_df.sort_values(by="Requested date", ascending=False).reset_index(drop=True)
display_df["No"] = (display_df.index + 1).astype(str)

edited = st.data_editor(
    display_df[
        [
            "No",
            "RMA ID",
            "Order",
            "Current Status",
            "Tracking Number",
            "Tracking Status",
            "Requested date",
            "Approved date",
            "Received date",
            "resolutionType",
            "Update Tracking Number",
            "View Timeline",
        ]
    ],
    use_container_width=True,
    height=700,
    hide_index=True,
    key=f"main_table_{st.session_state.table_key}",
    column_config={
        "No": st.column_config.TextColumn("No", width="small"),
        "RMA ID": st.column_config.TextColumn("RMA ID", width="small"),
        "Order": st.column_config.TextColumn("Order", width="small"),
        "Current Status": st.column_config.TextColumn("Current Status", width="small"),
        "Tracking Number": st.column_config.LinkColumn("Tracking Number", display_text=r"ref=(.*)", width="medium"),
        "Tracking Status": st.column_config.TextColumn("Tracking Status", width="medium"),
        "Requested date": st.column_config.TextColumn("Requested date", width="small"),
        "Approved date": st.column_config.TextColumn("Approved date", width="small"),
        "Received date": st.column_config.TextColumn("Received date", width="small"),
        "resolutionType": st.column_config.TextColumn("resolutionType", width="small"),
        "Update Tracking Number": st.column_config.CheckboxColumn("Update Tracking Number", width="small"),
        "View Timeline": st.column_config.CheckboxColumn("View Timeline", width="small"),
    },
    disabled=[
        "No",
        "RMA ID",
        "Order",
        "Current Status",
        "Tracking Number",
        "Tracking Status",
        "Requested date",
        "Approved date",
        "Received date",
        "resolutionType",
    ],
)

# ONE-CLICK & AUTO-CLEAR LOGIC:
editor_key = f"main_table_{st.session_state.table_key}"
if editor_key in st.session_state:
    edits = st.session_state[editor_key].get("edited_rows", {})
    for row_idx, changes in edits.items():
        idx = int(row_idx)
        if "Update Tracking Number" in changes and changes["Update Tracking Number"]:
            st.session_state.modal_rma = display_df.iloc[idx]
            st.session_state.modal_action = "edit"
            st.session_state.table_key += 1
            st.rerun()
        elif "View Timeline" in changes and changes["View Timeline"]:
            st.session_state.modal_rma = display_df.iloc[idx]
            st.session_state.modal_action = "view"
            st.session_state.table_key += 1
            st.rerun()
