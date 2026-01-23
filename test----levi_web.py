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
from typing import Optional, Tuple, List, Dict, Any

# ==========================================
# 1. CONFIGURATION
# ==========================================
st.set_page_config(page_title="Levi's ReturnGO Ops", layout="wide", page_icon="📤")

# --- Preserve scroll position across reruns (page + dataframe scroll) ---
components.html(
    """
    <script>
    (function() {
      const KEY_PAGE = "rg_scrollY_page";
      const KEY_DF   = "rg_scrollY_df";

      function savePageScroll() {
        try { parent.localStorage.setItem(KEY_PAGE, String(parent.window.scrollY || 0)); } catch(e){}
      }

      function restorePageScroll() {
        try {
          const y = parent.localStorage.getItem(KEY_PAGE);
          if (y) parent.window.scrollTo(0, parseInt(y));
        } catch(e){}
      }

      function findDfScroller() {
        const roots = parent.document.querySelectorAll('[data-testid="stDataFrame"], [data-testid="stDataEditor"]');
        for (const r of roots) {
          const candidates = r.querySelectorAll('div');
          for (const c of candidates) {
            const style = parent.getComputedStyle(c);
            if (style && (style.overflowY === "auto" || style.overflowY === "scroll") && c.scrollHeight > c.clientHeight) {
              return c;
            }
          }
        }
        return null;
      }

      function attachDfScroll() {
        const scroller = findDfScroller();
        if (!scroller) return false;

        try {
          const y = parent.localStorage.getItem(KEY_DF);
          if (y) scroller.scrollTop = parseInt(y);
        } catch(e){}

        scroller.addEventListener("scroll", () => {
          try { parent.localStorage.setItem(KEY_DF, String(scroller.scrollTop || 0)); } catch(e){}
        }, { passive: true });

        return true;
      }

      restorePageScroll();
      parent.window.addEventListener("scroll", savePageScroll, { passive: true });

      let tries = 0;
      const timer = setInterval(() => {
        tries += 1;
        const ok = attachDfScroll();
        if (ok || tries > 30) clearInterval(timer);
      }, 250);
    })();
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
# 1b. STYLING (sticky header + modern UI)
# ==========================================
st.markdown(
    """
    <style>
      :root{
        --bg0:#0b0f14;
        --card: rgba(17, 24, 39, 0.68);
        --border: rgba(148, 163, 184, 0.22);
        --text: #e5e7eb;
        --muted: rgba(148,163,184,0.95);
        --greenbg: rgba(34,197,94,0.14);
        --redglow: rgba(196,18,48,0.13);
        --blueglow: rgba(59,130,246,0.10);
      }

      .stApp {
        background:
          radial-gradient(1100px 520px at 15% 0%, var(--redglow), transparent 60%),
          radial-gradient(850px 520px at 88% 8%, var(--blueglow), transparent 55%),
          linear-gradient(180deg, rgba(255,255,255,0.02), transparent 40%),
          var(--bg0);
        color: var(--text);
      }

      /* Sticky header wrapper */
      .rg-sticky {
        position: sticky;
        top: 0;
        z-index: 999;
        padding-top: 6px;
        background: linear-gradient(180deg, rgba(11,15,20,0.95), rgba(11,15,20,0.75), rgba(11,15,20,0.05));
        backdrop-filter: blur(10px);
        border-bottom: 1px solid rgba(148,163,184,0.10);
      }

      /* Title area */
      .rg-title {
        font-size: 2.6rem;
        font-weight: 900;
        letter-spacing: -0.02em;
        margin-top: -22px;
        margin-bottom: 8px;
        text-shadow: 0 8px 22px rgba(0,0,0,0.35);
      }
      .rg-sub-pill {
        display: inline-flex;
        gap: 10px;
        align-items: center;
        padding: 10px 14px;
        border-radius: 999px;
        background: rgba(17, 24, 39, 0.78);
        border: 1px solid rgba(148, 163, 184, 0.20);
        box-shadow: 0 10px 24px rgba(0,0,0,0.25);
        font-size: 0.98rem;
        color: rgba(226,232,240,0.95);
      }
      .rg-dot {
        width: 8px; height: 8px; border-radius: 999px;
        background: rgba(196,18,48,0.95);
        box-shadow: 0 0 0 6px rgba(196,18,48,0.14);
      }

      /* Button cards */
      .rg-tile {
        position: relative;
        background: rgba(17, 24, 39, 0.68);
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 14px;
        padding: 12px 12px 10px 12px;
        box-shadow: 0 10px 26px rgba(0,0,0,0.25);
        overflow: hidden;
      }
      .rg-tile.selected {
        border-color: rgba(34,197,94,0.55);
        background: linear-gradient(180deg, var(--greenbg), rgba(17,24,39,0.72));
      }
      .rg-tile::before {
        content: "";
        position: absolute;
        left: 10px; right: 10px; top: 10px;
        height: 10px;
        border-radius: 999px;
        background: rgba(148,163,184,0.10);
        border: 1px solid rgba(148,163,184,0.12);
      }
      .rg-tile.selected::before {
        background: rgba(34,197,94,0.7);
        border-color: rgba(34,197,94,0.95);
        box-shadow: 0 0 12px rgba(34,197,94,0.7);
      }
      .rg-updated-pill {
        position: absolute;
        left: 14px;
        top: 2px;
        padding: 4px 9px;
        border-radius: 999px;
        background: rgba(15, 23, 42, 0.72);
        border: 1px solid rgba(148,163,184,0.18);
        color: rgba(148,163,184,0.95);
        font-size: 0.78rem;
      }
      .rg-api-box {
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
      .rg-api-sub {
        font-size: 0.72rem;
        font-weight: 600;
        color: rgba(148,163,184,0.9);
      }

      /* Streamlit button */
      div.stButton > button {
        width: 100%;
        border: 1px solid rgba(148, 163, 184, 0.22) !important;
        background: rgba(31, 41, 55, 0.88) !important;
        color: #e5e7eb !important;
        border-radius: 12px !important;
        padding: 12px 14px !important;
        font-size: 15px !important;
        font-weight: 800 !important;
        transition: 0.15s ease-in-out;
      }
      div.stButton > button:hover {
        border-color: rgba(196,18,48,0.65) !important;
        color: #fff !important;
        transform: translateY(-1px);
      }

      /* Mini refresh button */
      .rg-mini div.stButton > button {
        width: auto !important;
        margin: 0 auto !important;
        padding: 7px 12px !important;
        font-size: 13px !important;
        font-weight: 800 !important;
        border-radius: 999px !important;
        background: rgba(51, 65, 85, 0.55) !important;
        border: 1px solid rgba(148,163,184,0.18) !important;
      }
      .rg-tile div.stButton {
        margin-bottom: 0 !important;
      }
      .rg-mini {
        margin-top: -2px;
        display: flex;
        justify-content: center;
      }

      /* Right control card */
      .rg-right-card {
        background: rgba(17,24,39,0.72);
        border: 1px solid rgba(148,163,184,0.18);
        border-radius: 16px;
        padding: 12px;
        box-shadow: 0 10px 26px rgba(0,0,0,0.25);
      }
      .rg-right-card .stButton > button{
        background: rgba(51,65,85,0.60) !important;
      }

      /* Keep dataframe text on one line */
      [data-testid="stDataFrame"] * { white-space: nowrap !important; }
      [data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid rgba(148,163,184,0.18);
      }

      /* Dialog */
      div[data-testid="stDialog"] {
        background-color: rgba(17, 24, 39, 0.96);
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

        sync_cols = {row[1] for row in c.execute("PRAGMA table_info(sync_logs)").fetchall()}
        if ("status" in sync_cols) or ("last_sync" in sync_cols):
            try:
                c.execute(
                    """
                    CREATE TABLE IF NOT EXISTS sync_logs_new (
                        scope TEXT PRIMARY KEY,
                        last_sync_iso TEXT
                    )
                    """
                )
                c.execute(
                    """
                    INSERT OR REPLACE INTO sync_logs_new (scope, last_sync_iso)
                    SELECT status, last_sync FROM sync_logs
                    """
                )
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

def check_parcel_ninja_status(tracking_number: str):
    if not PARCEL_NINJA_TOKEN or not tracking_number:
        return None

    url = f"https://optimise.parcelninja.com/api/v1/shipments?search={tracking_number}"

    try:
        res = requests.get(url, headers={"Authorization": f"Bearer {PARCEL_NINJA_TOKEN}"}, timeout=12)
        if res.status_code != 200:
            return None
        data = res.json()
        shipments = data.get("shipments") or []
        if not shipments:
            return None
        return shipments[0].get("status")
    except Exception:
        return None


# ==========================================
# 5. SYNC LOGIC (ReturnGO)
# ==========================================

def fetch_rma_detail(rma_id: str) -> bool:
    url = f"https://api.returngo.ai/rma/{rma_id}"
    res = rg_request("GET", url, timeout=18)
    if res.status_code != 200:
        return False
    data = res.json()
    summary = data.get("rmaSummary", {}) or {}
    status = summary.get("status", "Unknown")
    created = summary.get("createdAt", "")
    upsert_rma(rma_id, status, created, data)
    return True


def fetch_status_rmas(status: str) -> List[dict]:
    status_param = status.lower()
    since_iso = ""
    last = get_last_sync(status)
    if last:
        since = last - timedelta(minutes=SYNC_OVERLAP_MINUTES)
        since_iso = _iso_utc(since)
    updated_filter = f"&updatedAfter={since_iso}" if since_iso else ""
    base_url = f"https://api.returngo.ai/rmas?pagesize=500&status={status_param}{updated_filter}"

    all_rmas = []
    next_url = base_url
    while next_url:
        res = rg_request("GET", next_url, timeout=18)
        if res.status_code != 200:
            break
        payload = res.json()
        rmas = payload.get("rmas") or []
        all_rmas.extend(rmas)
        next_url = payload.get("nextPage")

    return all_rmas


def perform_sync(statuses: Optional[List[str]] = None):
    if statuses is None:
        statuses = ["Pending", "Approved", "Received"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_status = {executor.submit(fetch_status_rmas, st): st for st in statuses}
        collected = {}
        for future in concurrent.futures.as_completed(future_to_status):
            st_name = future_to_status[future]
            try:
                collected[st_name] = future.result()
            except Exception:
                collected[st_name] = []

    for status in statuses:
        remote_rmas = collected.get(status, [])
        remote_ids = {str(r.get("rmaId")) for r in remote_rmas if r.get("rmaId")}

        local_ids = get_local_ids_for_status(status)
        missing = local_ids - remote_ids
        delete_rmas(missing)

        for r in remote_rmas:
            rid = str(r.get("rmaId"))
            created_at = r.get("createdAt", "")
            upsert_rma(rid, status, created_at, r)

        set_last_sync(status, _now_utc())

    st.session_state["show_toast"] = True
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
        res = rg_request(
            "PUT",
            f"https://api.returngo.ai/shipment/{shipment_id}",
            headers=headers,
            timeout=15,
            json_body=payload,
        )
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
        res = rg_request(
            "POST",
            f"https://api.returngo.ai/rma/{rma_id}/comment",
            headers=headers,
            timeout=15,
            json_body=payload,
        )
        if res.status_code in (200, 201):
            fresh = fetch_rma_detail(rma_id)
            return True, "Success" if fresh else "Posted"
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

    shipments = rma_payload.get("shipments", []) or []
    for s in shipments:
        if str(s.get("status", "")).lower() == "received":
            lu = rma_payload.get("lastUpdated")
            if lu:
                return str(lu)

    return str(rma_payload.get("_local_received_first_seen") or "")


def pretty_resolution(rt: str) -> str:
    if not rt:
        return ""
    out = re.sub(r"([a-z])([A-Z])", r"\1 \2", rt).strip()
    return out.replace("To ", "to ")


def get_resolution_type(rma_payload: dict) -> str:
    items = rma_payload.get("items") or (rma_payload.get("rmaSummary", {}) or {}).get("items") or []
    types = {i.get("resolutionType") for i in items if isinstance(i, dict) and i.get("resolutionType")}
    if len(types) > 1:
        return "Mix"
    if len(types) == 1:
        return pretty_resolution(next(iter(types)))
    return ""


def resolution_actioned_label(rma_payload: dict) -> str:
    txns = rma_payload.get("transactions", []) or []
    for t in txns:
        if str(t.get("status", "")).lower() == "success":
            return f"Yes ({t.get('type', 'Transaction')})"

    exchange_orders = rma_payload.get("exchangeOrders", []) or []
    if exchange_orders:
        return "Yes (Exchange released)"

    comments = rma_payload.get("comments", []) or []
    hay = " ".join((c.get("htmlText", "") or "") for c in comments).lower()
    if any(k in hay for k in ["transaction id", "credited", "total refund amount", "exchange order", "released"]):
        return "Yes"

    return "No"


def days_since(date_str: str) -> str:
    try:
        if not date_str or date_str == "N/A":
            return "N/A"
        d = datetime.fromisoformat(date_str)
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return str((_now_utc().date() - d.date()).days)
    except Exception:
        try:
            d = datetime.strptime(date_str, "%Y-%m-%d")
            return str((_now_utc().date() - d.date()).days)
        except Exception:
            return "N/A"


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
        except Exception:
            sub = f"Reset: {reset}"
    elif reset:
        sub = f"Reset: {reset}"
    elif updated_at:
        sub = f"Updated: {updated_at.astimezone().strftime('%H:%M')}"

    return main, sub


# ==========================================
# 8. UI STATE
# ==========================================
if "selected_filters" not in st.session_state:
    st.session_state.selected_filters = set()
if "search_query_input" not in st.session_state:
    st.session_state.search_query_input = ""

if st.session_state.get("show_toast"):
    st.toast("✅ API Sync Complete!", icon="🔄")
    st.session_state["show_toast"] = False

if RATE_LIMIT_HIT.is_set():
    st.warning(
        "ReturnGO rate limit reached (429). Sync is slowing down and retrying. "
        "If this happens often, sync less frequently or request a higher quota key."
    )
    RATE_LIMIT_HIT.clear()

# ==========================================
# 9. STICKY HEADER AREA
# ==========================================
st.markdown("<div class='rg-sticky'>", unsafe_allow_html=True)

top_left, top_right = st.columns([3.3, 1], vertical_alignment="top")

with top_left:
    st.markdown("<div class='rg-title'>Levi's ReturnGO Ops Dashboard</div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="rg-sub-pill">
            <span class="rg-dot"></span>
            <span><b>CONNECTED TO:</b> {STORE_URL.upper()}</span>
            <span style="opacity:.6">|</span>
            <span><b>CACHE:</b> {int(CACHE_EXPIRY_HOURS)}h</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with top_right:
    st.markdown("<div class='rg-right-card'>", unsafe_allow_html=True)
    api_col, sync_col = st.columns([1, 1.4], vertical_alignment="center")
    with api_col:
        api_main, api_sub = format_api_limit_display()
        st.markdown(
            f"""
            <div class="rg-api-box">
              <div>{api_main}</div>
              <div class="rg-api-sub">{api_sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with sync_col:
        if st.button("🔄 Sync Dashboard", key="btn_sync_all", use_container_width=True):
            perform_sync()
    if st.button("🗑️ Reset Cache", key="btn_reset", use_container_width=True):
        if clear_db():
            st.success("Cache cleared!")
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# 10. LOAD + PROCESS CACHED OPEN RMAs
# ==========================================
raw_data = get_all_open_from_db()
processed_rows: List[Dict[str, Any]] = []

counts = {
    "Pending": 0,
    "Approved": 0,
    "Received": 0,
    "NoTrack": 0,
    "Flagged": 0,
    "CourierCancelled": 0,
    "ApprovedDelivered": 0,
    "ResolutionActioned": 0,
    "NoResolutionActioned": 0,
}

search_query = (st.session_state.get("search_query_input") or "").strip()
search_query_norm = search_query.lower()

for rma in raw_data:
    summary = rma.get("rmaSummary", {}) or {}
    shipments = rma.get("shipments", []) or []
    comments = rma.get("comments", []) or []

    status = summary.get("status", "Unknown")
    order_name = summary.get("order_name", summary.get("orderName", "N/A"))
    rma_id_text = str(summary.get("rmaId", "N/A"))
    order_text = str(order_name)

    track_nums = [s.get("trackingNumber") for s in shipments if s.get("trackingNumber")]
    track_str = ", ".join(track_nums) if track_nums else ""
    shipment_id = shipments[0].get("shipmentId") if shipments else None

    local_tracking_status = rma.get("_local_courier_status", "") or ""
    track_link_url = f"https://portal.thecourierguy.co.za/track?ref={track_nums[0]}" if track_nums else ""

    requested_iso = get_event_date_iso(rma, "RMA_CREATED") or (summary.get("createdAt") or rma.get("createdAt") or "")
    approved_iso = get_event_date_iso(rma, "RMA_APPROVED")
    received_iso = get_received_date_iso(rma)
    resolution_type = get_resolution_type(rma)
    actioned_label = resolution_actioned_label(rma)

    is_nt = (status == "Approved" and not track_str)
    is_fg = any("flagged" in (c.get("htmlText", "").lower()) for c in comments)
    is_cc = bool(re.search(r"courier cancelled", str(local_tracking_status), re.IGNORECASE))
    is_ad = bool(status == "Approved" and re.search(r"delivered", str(local_tracking_status), re.IGNORECASE))
    is_ra = (actioned_label or "").lower().startswith("yes")
    is_nra = (not is_ra) and (status == "Received")

    if status in ("Pending", "Approved", "Received"):
        counts[status] += 1
    if is_nt:
        counts["NoTrack"] += 1
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

    if search_query and not (
        search_query_norm in rma_id_text.lower()
        or search_query_norm in order_text.lower()
        or search_query_norm in track_str.lower()
    ):
        continue

    rma_url = f"https://app.returngo.ai/dashboard/returns?filter_status=open&rmaid={rma_id_text}"

    processed_rows.append(
        {
            "No": "",
            # URL is stored here; we will display the ID using LinkColumn display_text
            "RMA ID": rma_url,
            "_rma_id_text": rma_id_text,

            "Order": order_text,
            "Current Status": status,
            "Tracking Number": track_link_url,
            "Tracking Status": local_tracking_status,
            "Requested date": str(requested_iso)[:10] if requested_iso else "N/A",
            "Approved date": str(approved_iso)[:10] if approved_iso else "N/A",
            "Received date": str(received_iso)[:10] if received_iso else "N/A",
            "Days since requested": days_since(str(requested_iso)[:10] if requested_iso else "N/A"),
            "resolutionType": resolution_type if resolution_type else "N/A",
            "Resolution actioned": actioned_label or "No",

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


def get_status_time_html(s: str) -> str:
    try:
        ts = get_last_sync(s)
        if not ts:
            return "UPDATED: -"
        return f"UPDATED: {ts.strftime('%H:%M')}"
    except Exception:
        return "UPDATED: -"


def toggle_filter(key: str):
    s = st.session_state.selected_filters
    if key in s:
        s.remove(key)
    else:
        s.add(key)
    st.session_state.selected_filters = s
    st.rerun()


def clear_all_filters():
    st.session_state.selected_filters = set()
    st.session_state.search_query_input = ""
    st.session_state.req_dates_selected = []
    st.session_state.status_multi = []
    st.session_state.res_multi = []
    st.session_state.actioned_multi = []
    st.rerun()


def refresh_for_filter(filter_key: str):
    if filter_key in ("Pending",):
        perform_sync(["Pending"])
    elif filter_key in ("Approved", "NoTrack", "ApprovedDelivered"):
        perform_sync(["Approved"])
    elif filter_key in ("Received", "CourierCancelled", "ResolutionActioned", "NoResolutionActioned", "Flagged"):
        perform_sync(["Received"])
    else:
        perform_sync()


def tile(label: str, key: str, count_val: int, icon: str, scope_for_updated: str):
    selected = key in st.session_state.selected_filters
    cls = "rg-tile selected" if selected else "rg-tile"

    st.markdown(f"<div class='{cls}'>", unsafe_allow_html=True)
    st.markdown(f"<div class='rg-updated-pill'>{get_status_time_html(scope_for_updated)}</div>", unsafe_allow_html=True)

    if st.button(f"{icon}  {label.upper()} [**{count_val}**]", key=f"tile_{key}", use_container_width=True):
        toggle_filter(key)

    st.markdown("<div class='rg-mini'>", unsafe_allow_html=True)
    if st.button("🔄 Refresh", key=f"ref_{key}"):
        refresh_for_filter(key)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# Tiles layout
r1 = st.columns(5, vertical_alignment="top")
with r1[0]:
    tile("Pending", "Pending", counts["Pending"], "⏳", "Pending")
with r1[1]:
    tile("Approved", "Approved", counts["Approved"], "✅", "Approved")
with r1[2]:
    tile("Received", "Received", counts["Received"], "📦", "Received")
with r1[3]:
    tile("No Tracking", "NoTrack", counts["NoTrack"], "🚫", "Approved")
with r1[4]:
    tile("Flagged", "Flagged", counts["Flagged"], "🚩", "Received")

r2 = st.columns(4, vertical_alignment="top")
with r2[0]:
    tile("Courier Cancelled", "CourierCancelled", counts["CourierCancelled"], "🧾", "Received")
with r2[1]:
    tile("Approved > Delivered", "ApprovedDelivered", counts["ApprovedDelivered"], "📬", "Approved")
with r2[2]:
    tile("Resolution Actioned", "ResolutionActioned", counts["ResolutionActioned"], "💸", "Received")
with r2[3]:
    tile("No Resolution Actioned", "NoResolutionActioned", counts["NoResolutionActioned"], "🕒", "Received")

# Search row
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
        st.session_state.selected_filters = set()
        st.rerun()

# Filter bar default open
with st.expander("Additional filters", expanded=True):
    c1, c2, c3, c4, c5 = st.columns([2, 2, 2, 3, 1], vertical_alignment="center")

    with c1:
        st.multiselect(
            "Status",
            options=["Pending", "Approved", "Received"],
            default=st.session_state.get("status_multi", []),
            key="status_multi",
        )

    with c2:
        res_opts = []
        if not df_view.empty and "resolutionType" in df_view.columns:
            res_opts = sorted([x for x in df_view["resolutionType"].dropna().unique().tolist() if x and x != "N/A"])
        st.multiselect(
            "Resolution type",
            options=res_opts,
            default=st.session_state.get("res_multi", []),
            key="res_multi",
        )

    with c3:
        st.multiselect(
            "Resolution actioned",
            options=["Yes", "No"],
            default=st.session_state.get("actioned_multi", []),
            key="actioned_multi",
        )

    with c4:
        req_dates = []
        if not df_view.empty and "Requested date" in df_view.columns:
            req_dates = sorted(
                [d for d in df_view["Requested date"].dropna().astype(str).unique().tolist() if d and d != "N/A"],
                reverse=True,
            )
        st.multiselect(
            "Requested date (multi-select)",
            options=req_dates,
            default=st.session_state.get("req_dates_selected", []),
            key="req_dates_selected",
        )

    with c5:
        if st.button("🧼 Clear filters", use_container_width=True):
            clear_all_filters()

st.markdown("</div>", unsafe_allow_html=True)  # close sticky

st.divider()

# ==========================================
# 12. BUILD DISPLAY DF
# ==========================================
if df_view.empty:
    st.warning("Database empty. Click Sync Dashboard to start.")
    st.stop()

display_df = df_view.copy()

sel = st.session_state.selected_filters
if sel:
    masks = []
    for k in sel:
        if k == "Pending":
            masks.append(display_df["Current Status"] == "Pending")
        elif k == "Approved":
            masks.append(display_df["Current Status"] == "Approved")
        elif k == "Received":
            masks.append(display_df["Current Status"] == "Received")
        elif k == "NoTrack":
            masks.append(display_df["is_nt"] == True)
        elif k == "Flagged":
            masks.append(display_df["is_fg"] == True)
        elif k == "CourierCancelled":
            masks.append(display_df["is_cc"] == True)
        elif k == "ApprovedDelivered":
            masks.append(display_df["is_ad"] == True)
        elif k == "ResolutionActioned":
            masks.append(display_df["is_ra"] == True)
        elif k == "NoResolutionActioned":
            masks.append(display_df["is_nra"] == True)

    if masks:
        m = masks[0]
        for mm in masks[1:]:
            m = m | mm
        display_df = display_df[m]

# secondary filters AND
if st.session_state.get("status_multi"):
    display_df = display_df[display_df["Current Status"].isin(st.session_state["status_multi"])]

if st.session_state.get("res_multi"):
    display_df = display_df[display_df["resolutionType"].isin(st.session_state["res_multi"])]

if st.session_state.get("actioned_multi"):
    want = set(st.session_state["actioned_multi"])
    norm = display_df["Resolution actioned"].astype(str).str.lower().apply(lambda x: "Yes" if x.startswith("yes") else "No")
    display_df = display_df[norm.isin(want)]

if st.session_state.get("req_dates_selected"):
    display_df = display_df[display_df["Requested date"].isin(st.session_state["req_dates_selected"])]

if display_df.empty:
    st.info("No matching records found in cache.")
    st.stop()

display_df = display_df.sort_values(by="Requested date", ascending=False).reset_index(drop=True)
display_df["No"] = (display_df.index + 1).astype(str)

# ==========================================
# 13. OPTION A: ROW CLICK OPENS DIALOG (native table)
# ==========================================
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
                    ok, msg = push_comment_update(rma_id_text, comment_text)
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

# Create table data; keep helper fields available for dialog selection
_table_df = display_df[display_cols + ["_rma_id_text", "DisplayTrack", "shipment_id", "full_data"]].copy()

sel_event = st.dataframe(
    _table_df[display_cols],
    use_container_width=True,
    hide_index=True,
    column_config=column_config,
    on_select="rerun",
    selection_mode="single-row",
)

sel_rows = (sel_event.selection.rows if sel_event and hasattr(sel_event, "selection") else []) or []
if sel_rows:
    idx = int(sel_rows[0])
    show_rma_actions_dialog(display_df.iloc[idx])
