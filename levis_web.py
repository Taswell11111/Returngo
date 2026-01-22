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
from io import BytesIO
from datetime import datetime, timedelta, timezone
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import concurrent.futures
from typing import Optional, Tuple, Dict, Callable, Set

# ==========================================
# 1. CONFIGURATION
# ==========================================
st.set_page_config(page_title="Levi's ReturnGO Ops", layout="wide", page_icon="📤")

# --- Preserve scroll position across reruns ---
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
CACHE_EXPIRY_HOURS = 24          # cache detail freshness window
COURIER_REFRESH_HOURS = 12       # don't re-check courier tracking too often
MAX_WORKERS = 3                  # ReturnGO docs recommend minimising concurrency
RG_RPS = 2                       # soft rate-limit to reduce 429
SYNC_OVERLAP_MINUTES = 5         # overlap window for incremental sync

ACTIVE_STATUSES = ["Pending", "Approved", "Received"]

# Thread-safe UI signal for 429s (never call st.* from worker threads)
RATE_LIMIT_HIT = threading.Event()

# ==========================================
# 1b. STYLING + STICKY HEADER
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

      /* Header typography */
      .title-wrap h1 {
        font-size: 3.0rem;
        margin-bottom: 0.2rem;
        letter-spacing: 0.2px;
      }

      .subtitle-bar {
        display: inline-flex;
        gap: 10px;
        align-items: center;
        padding: 8px 12px;
        border-radius: 999px;
        border: 1px solid rgba(148, 163, 184, 0.22);
        background: rgba(17, 24, 39, 0.55);
        box-shadow: 0 8px 20px rgba(0,0,0,0.20);
        font-size: 1.02rem;
        color: #cbd5e1;
      }
      .subtitle-bar b { color: #e5e7eb; }
      .subtitle-dot {
        width: 6px; height: 6px;
        border-radius: 50%;
        background: rgba(196,18,48,0.9);
        display: inline-block;
      }

      /* Card containers */
      .card {
        background: rgba(17, 24, 39, 0.70);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 14px;
        padding: 12px 12px 10px 12px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.25);
      }

      /* "Updated" pill inside card */
      .updated-pill {
        display: inline-block;
        font-size: 0.80rem;
        color: rgba(148,163,184,0.95);
        padding: 4px 10px;
        border-radius: 999px;
        background: rgba(15, 23, 42, 0.55);
        border: 1px solid rgba(148, 163, 184, 0.18);
        margin-bottom: 10px;
      }

      /* Streamlit buttons */
      div.stButton > button {
        width: 100%;
        border: 1px solid rgba(148, 163, 184, 0.25) !important;
        background: rgba(31, 41, 55, 0.92) !important;
        color: #e5e7eb !important;
        border-radius: 10px !important;
        padding: 12px 14px !important;
        font-size: 15px !important;
        font-weight: 800 !important;
        transition: 0.15s ease-in-out;
      }
      div.stButton > button:hover {
        border-color: rgba(196,18,48,0.70) !important;
        color: #fff !important;
        transform: translateY(-1px);
      }

      /* Smaller mini refresh buttons */
      .mini-refresh div.stButton > button {
        padding: 7px 10px !important;
        font-size: 13px !important;
        font-weight: 800 !important;
        border-radius: 10px !important;
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

      /* Sticky container class applied by JS */
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

      /* Reduce default vertical gaps a bit */
      [data-testid="stVerticalBlock"] > div:has(.card) {
        margin-top: 6px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# Anchor + JS to “stick” the whole header block
st.markdown('<div class="sticky-anchor"></div>', unsafe_allow_html=True)
components.html(
    """
    <script>
      function applySticky() {
        const anchor = window.parent.document.querySelector('.sticky-anchor');
        if (!anchor) return;

        // climb parents until we find a vertical block (Streamlit container)
        let el = anchor;
        for (let i = 0; i < 12; i++) {
          if (!el) break;
          if (el.getAttribute && el.getAttribute("data-testid") === "stVerticalBlock") {
            el.classList.add("sticky-top");
            return;
          }
          el = el.parentElement;
        }
        // fallback: just stick parent
        if (anchor.parentElement) anchor.parentElement.classList.add("sticky-top");
      }

      // run now + after short delay (Streamlit DOM can hydrate)
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


def rg_headers() -> dict:
    return {"x-api-key": MY_API_KEY, "x-shop-name": STORE_URL}


def rg_request(method: str, url: str, *, headers=None, timeout=15, json_body=None):
    session = get_thread_session()
    headers = headers or rg_headers()

    backoff = 1
    res = None
    for _attempt in range(1, 6):
        _sleep_for_rate_limit()
        res = session.request(method, url, headers=headers, timeout=timeout, json=json_body)

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

        if res.status_code == 200:
            try:
                data = res.json()
                if isinstance(data, dict) and data.get("history"):
                    s0 = data["history"][0]
                    return (s0.get("status") or s0.get("description") or "Unknown") or "Unknown"
                return (data.get("status") or data.get("currentStatus") or "Unknown") or "Unknown"
            except Exception:
                pass

            content = res.text
            clean_html = re.sub(r"<(script|style).*?</\1>", "", content, flags=re.DOTALL | re.IGNORECASE)

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
# 5. RETURNGO FETCHING (incremental + cached)
# ==========================================
def fetch_rma_list(statuses, since_dt: Optional[datetime]):
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
    summaries = fetch_rma_list(statuses, since_dt)
    list_bar.progress(1.0, text=f"Fetched {len(summaries)} RMAs")
    time.sleep(0.10)
    list_bar.empty()

    api_ids = {s.get("rmaId") for s in summaries if s.get("rmaId")}

    # tidy cache for open RMAs (only for the “all open” scope)
    if set(statuses) == set(ACTIVE_STATUSES) and not full:
        local_active_ids = set()
        for stt in ACTIVE_STATUSES:
            local_active_ids |= get_local_ids_for_status(stt)
        stale = local_active_ids - api_ids
        delete_rmas(stale)

    # KEY BEHAVIOUR:
    # - If we’re doing incremental (since_dt is not None), then EVERY rmaId returned by that list changed recently,
    #   so refresh details regardless of cache freshness.
    # - If full (since_dt is None), then respect cache expiry for efficiency.
    if since_dt is not None:
        to_fetch = list(api_ids)
    else:
        to_fetch = [rid for rid in api_ids if should_refresh_detail(rid)]

    total = len(to_fetch)
    status_msg.info(f"⏳ Syncing {total} records...")

    if total > 0:
        bar = st.progress(0, text="Downloading Details...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = [ex.submit(fetch_rma_detail, rid, force=(since_dt is not None)) for rid in to_fetch]
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
    """Force-refresh details for specific RMAs (ignores cache window)."""
    ids = [str(i) for i in set(rma_ids) if i]
    if not ids:
        set_last_sync(scope_label, _now_utc())
        st.session_state["show_toast"] = True
        st.rerun()

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
# 7. Helpers (dates, resolution, actioned)
# ==========================================
def get_event_date_iso(rma_payload: dict, event_name: str) -> str:
    summary = rma_payload.get("rmaSummary", {}) or {}
    for e in (summary.get("events") or []):
        if e.get("eventName") == event_name and e.get("eventDate"):
            return str(e["eventDate"])
    return ""


def get_received_date_iso(rma_payload: dict) -> str:
    # Try common event names + shipment received hints
    for name in ("SHIPMENT_RECEIVED", "RMA_RECEIVED", "RMA_STATUS_RECEIVED"):
        dt = get_event_date_iso(rma_payload, name)
        if dt:
            return dt

    # Try comments for "Shipment RECEIVED"
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


def is_resolution_actioned(rma_payload: dict) -> bool:
    # 1) Transactions with Success
    for t in (rma_payload.get("transactions") or []):
        if (t.get("status") or "").lower() == "success":
            return True

    # 2) Exchange released (exchangeOrders present)
    ex_orders = rma_payload.get("exchangeOrders") or []
    if isinstance(ex_orders, list) and len(ex_orders) > 0:
        return True

    # 3) Fallback: comments containing credit/refund/exchange release hints
    for c in (rma_payload.get("comments") or []):
        txt = (c.get("htmlText", "") or "").lower()
        if "transaction id" in txt and ("refund" in txt or "credited" in txt):
            return True
        if "exchange order" in txt and "released" in txt:
            return True
        if "total refund amount" in txt:
            return True

    return False


def resolution_actioned_label(rma_payload: dict) -> str:
    # give a human label where possible
    for t in (rma_payload.get("transactions") or []):
        if (t.get("status") or "").lower() == "success":
            typ = (t.get("type") or "").strip()
            if typ:
                return pretty_resolution(typ)
            return "Yes"
    ex_orders = rma_payload.get("exchangeOrders") or []
    if isinstance(ex_orders, list) and len(ex_orders) > 0:
        return "Exchange released"
    if is_resolution_actioned(rma_payload):
        return "Yes"
    return "No"


# ==========================================
# 8. FRONTEND STATE
# ==========================================
if "modal_rma" not in st.session_state:
    st.session_state.modal_rma = None
if "modal_action" not in st.session_state:
    st.session_state.modal_action = None
if "table_key" not in st.session_state:
    st.session_state.table_key = 0

# Multi-select filters
if "active_filters" not in st.session_state:
    st.session_state.active_filters = set()  # type: ignore

# Search state
if "search_query_input" not in st.session_state:
    st.session_state.search_query_input = ""

# Global toast
if st.session_state.get("show_toast"):
    st.toast("✅ Updated!", icon="🔄")
    st.session_state["show_toast"] = False

# Show 429 warning
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


def clear_filters():
    st.session_state.active_filters = set()  # type: ignore
    st.rerun()


def updated_pill(scope: str) -> str:
    ts = get_last_sync(scope)
    if not ts:
        return "<span class='updated-pill'>UPDATED: -</span>"
    return f"<span class='updated-pill'>UPDATED: {ts.strftime('%H:%M')}</span>"


# ==========================================
# 9. DIALOGS
# ==========================================
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
                    time.sleep(0.25)
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


# modal trigger after rerun
if st.session_state.modal_rma is not None:
    current_rma = st.session_state.modal_rma
    current_act = st.session_state.modal_action
    st.session_state.modal_rma = None
    st.session_state.modal_action = None
    if current_act == "edit":
        show_update_tracking_dialog(current_rma)
    elif current_act == "view":
        show_timeline_dialog(current_rma)


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
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    if st.button("🔄 Sync Dashboard", key="btn_sync_all", use_container_width=True):
        perform_sync()
    if st.button("🗑️ Reset Cache", key="btn_reset", use_container_width=True):
        if clear_db():
            st.success("Cache cleared!")
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

st.write("")

# ==========================================
# 11. LOAD + PROCESS OPEN RMAs
# ==========================================
raw_data = get_all_open_from_db()
processed_rows = []

# Base counts (statuses)
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
    resolution_type = get_resolution_type(rma)

    actioned = is_resolution_actioned(rma)
    actioned_label = resolution_actioned_label(rma)

    # Flags
    is_nt = (status == "Approved" and not track_str)
    is_fg = any("flagged" in (c.get("htmlText", "").lower()) for c in comments)
    is_cc = bool(local_tracking_status) and ("courier cancelled" in local_tracking_status.lower())
    is_ad = (status == "Approved") and (bool(local_tracking_status) and ("delivered" in local_tracking_status.lower()))
    is_ra = actioned
    is_nra = (status == "Received") and (not actioned)

    # Counts
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

    # Search filter at row-build stage (faster)
    if not search_query or (
        search_query.lower() in str(rma_id).lower()
        or search_query.lower() in str(order_name).lower()
        or search_query.lower() in str(track_str).lower()
    ):
        processed_rows.append(
            {
                "No": "",
                "RMA ID": rma_id,
                "RMA Link": f"https://app.returngo.ai/dashboard/returns?filter_status=open&rmaid={rma_id}",
                "Order": order_name,
                "Current Status": status,
                "Tracking Number": track_link_url,
                "Tracking Status": local_tracking_status,
                "Requested date": str(requested_iso)[:10] if requested_iso else "N/A",
                "Approved date": str(approved_iso)[:10] if approved_iso else "N/A",
                "Received date": str(received_iso)[:10] if received_iso else "N/A",
                "resolutionType": resolution_type if resolution_type else "N/A",
                "Resolution actioned": actioned_label,
                "Update Tracking Number": False,
                "View Timeline": False,
                "DisplayTrack": track_str,
                "shipment_id": shipment_id,
                "full_data": rma,
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
# 12. FILTER DEFINITIONS (for multi-select)
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


# ==========================================
# 13. FILTER TILE UI (with mini refresh)
# ==========================================
def render_filter_tile(col, name: str, *, refresh_ids_provider: Callable[[], list], refresh_scope: str):
    cfg = FILTERS[name]
    icon = cfg["icon"]  # type: ignore
    count_key = cfg["count_key"]  # type: ignore
    scope = cfg["scope"]  # type: ignore

    active: Set[str] = st.session_state.active_filters  # type: ignore
    selected = (name in active)

    count_val = counts.get(count_key, 0)
    sel_mark = "✅ " if selected else ""
    label = f"{sel_mark}{icon} {name.upper()} [{count_val}]"

    with col:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(updated_pill(str(scope)), unsafe_allow_html=True)

        if st.button(label, key=f"flt_{name}", use_container_width=True):
            toggle_filter(name)

        st.markdown("<div class='mini-refresh'>", unsafe_allow_html=True)
        if st.button("🔄 Refresh", key=f"ref_{name}", use_container_width=True):
            ids = refresh_ids_provider()
            force_refresh_rma_ids(ids, refresh_scope)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


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
    return df_view.loc[mask, "RMA ID"].astype(str).tolist()


# Row 1 (5 tiles)
r1 = st.columns(5)
render_filter_tile(r1[0], "Pending", refresh_ids_provider=lambda: ids_for_filter("Pending"), refresh_scope="Pending")
render_filter_tile(r1[1], "Approved", refresh_ids_provider=lambda: ids_for_filter("Approved"), refresh_scope="Approved")
render_filter_tile(r1[2], "Received", refresh_ids_provider=lambda: ids_for_filter("Received"), refresh_scope="Received")
render_filter_tile(r1[3], "No Tracking", refresh_ids_provider=lambda: ids_for_filter("No Tracking"), refresh_scope="FILTER_NoTracking")
render_filter_tile(r1[4], "Flagged", refresh_ids_provider=lambda: ids_for_filter("Flagged"), refresh_scope="FILTER_Flagged")

st.write("")

# Row 2 (4 tiles)
r2 = st.columns(4)
render_filter_tile(r2[0], "Courier Cancelled", refresh_ids_provider=lambda: ids_for_filter("Courier Cancelled"), refresh_scope="FILTER_CourierCancelled")
render_filter_tile(r2[1], "Approved > Delivered", refresh_ids_provider=lambda: ids_for_filter("Approved > Delivered"), refresh_scope="FILTER_ApprovedDelivered")
render_filter_tile(r2[2], "Resolution Actioned", refresh_ids_provider=lambda: ids_for_filter("Resolution Actioned"), refresh_scope="FILTER_ResolutionActioned")
render_filter_tile(r2[3], "No Resolution Actioned", refresh_ids_provider=lambda: ids_for_filter("No Resolution Actioned"), refresh_scope="FILTER_NoResolutionActioned")

st.write("")

# ==========================================
# 14. SEARCH + VIEW ALL (still inside sticky header)
# ==========================================
fc1, fc2 = st.columns([1.4, 8.6], vertical_alignment="center")

with fc1:
    if st.button("📋 View All Open RMAs", key="btn_view_all_open", use_container_width=True):
        clear_filters()

with fc2:
    sc1, sc2 = st.columns([10, 1], vertical_alignment="center")
    with sc1:
        st.text_input(
            "Search",
            placeholder="🔎 Search Order, RMA, or Tracking...",
            label_visibility="collapsed",
            key="search_query_input",
        )
    with sc2:
        def clear_search_cb():
            st.session_state.search_query_input = ""
        st.button("❌", help="Clear Search", key="clear_search_btn", on_click=clear_search_cb, use_container_width=True)

st.divider()

# ==========================================
# 15. TABLE VIEW
# ==========================================
if df_view.empty:
    st.warning("Database empty. Click Sync Dashboard to start.")
    st.stop()

display_df = df_view[current_filter_mask(df_view)].copy()

if display_df.empty:
    st.info("No matching records found.")
    st.stop()

display_df = display_df.sort_values(by="Requested date", ascending=False).reset_index(drop=True)
display_df["No"] = (display_df.index + 1).astype(str)

edited = st.data_editor(
    display_df[
        [
            "No",
            "RMA Link",
            "RMA ID",
            "Order",
            "Current Status",
            "Tracking Number",
            "Tracking Status",
            "Requested date",
            "Approved date",
            "Received date",
            "resolutionType",
            "Resolution actioned",
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
        "RMA Link": st.column_config.LinkColumn("RMA", display_text="Open", width="small"),
        "RMA ID": st.column_config.TextColumn("RMA ID", width="small"),
        "Order": st.column_config.TextColumn("Order", width="small"),
        "Current Status": st.column_config.TextColumn("Current Status", width="small"),
        "Tracking Number": st.column_config.LinkColumn("Tracking Number", display_text=r"ref=(.*)", width="medium"),
        "Tracking Status": st.column_config.TextColumn("Tracking Status", width="medium"),
        "Requested date": st.column_config.TextColumn("Requested date", width="small"),
        "Approved date": st.column_config.TextColumn("Approved date", width="small"),
        "Received date": st.column_config.TextColumn("Received date", width="small"),
        "resolutionType": st.column_config.TextColumn("resolutionType", width="small"),
        "Resolution actioned": st.column_config.TextColumn("Resolution actioned", width="small"),
        "Update Tracking Number": st.column_config.CheckboxColumn("Update Tracking Number", width="small"),
        "View Timeline": st.column_config.CheckboxColumn("View Timeline", width="small"),
    },
    disabled=[
        "No",
        "RMA Link",
        "RMA ID",
        "Order",
        "Current Status",
        "Tracking Number",
        "Tracking Status",
        "Requested date",
        "Approved date",
        "Received date",
        "resolutionType",
        "Resolution actioned",
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
