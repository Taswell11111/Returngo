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

# =========================================================
# Levi's ReturnGO Ops Dashboard (Levi's SA)
# - Streamlit Cloud friendly
# - NO AG Grid
# - NO XLSX export / NO openpyxl
# - Cache: 24h
# - Multi-select tiles with refresh buttons
# - Row click opens a dialog (tabs: Update Tracking + View Timeline)
# =========================================================

st.set_page_config(page_title="Levi's ReturnGO Ops", layout="wide", page_icon="📤")

STORE_URL = "levis-sa.myshopify.com"
DB_FILE = "levis_cache.db"
DB_LOCK = threading.Lock()

CACHE_EXPIRY_HOURS = 24
COURIER_REFRESH_HOURS = 12
MAX_WORKERS = 3
RG_RPS = 2
SYNC_OVERLAP_MINUTES = 5

ACTIVE_STATUSES = ["Pending", "Approved", "Received"]
RATE_LIMIT_HIT = threading.Event()

# --- Secrets ---
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

# =========================================================
# Preserve scroll position (page + table) across reruns
# =========================================================
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

# =========================================================
# Styling
# =========================================================
st.markdown(
    """
    <style>
      :root{
        --bg0:#0b0f14;
        --card: rgba(17, 24, 39, 0.70);
        --border: rgba(148, 163, 184, 0.22);
        --text: #e5e7eb;
        --muted: rgba(148,163,184,0.95);
        --green: rgba(0,255,102,1); /* brighter */
        --greenSoft: rgba(0,255,102,0.20);
        --redglow: rgba(196,18,48,0.12);
        --blueglow: rgba(59,130,246,0.10);
      }

      .stApp {
        background:
          radial-gradient(1200px 560px at 18% 0%, var(--redglow), transparent 62%),
          radial-gradient(900px 560px at 86% 10%, var(--blueglow), transparent 58%),
          linear-gradient(180deg, rgba(255,255,255,0.02), transparent 42%),
          var(--bg0);
        color: var(--text);
      }

      /* sticky top zone */
      .rg-sticky {
        position: sticky;
        top: 0;
        z-index: 999;
        padding: 10px 0 10px 0;
        background: linear-gradient(180deg, rgba(11,15,20,0.96), rgba(11,15,20,0.80), rgba(11,15,20,0.06));
        backdrop-filter: blur(10px);
        border-bottom: 1px solid rgba(148,163,184,0.10);
      }

      .rg-title {
        font-size: 2.85rem;
        font-weight: 900;
        letter-spacing: -0.02em;
        margin-top: -24px;
        margin-bottom: 10px;
        text-shadow: 0 10px 24px rgba(0,0,0,0.35);
      }

      .rg-subpill {
        display: inline-flex;
        gap: 10px;
        align-items: center;
        padding: 10px 14px;
        border-radius: 16px;
        background: rgba(17,24,39,0.78);
        border: 1px solid rgba(148,163,184,0.20);
        box-shadow: 0 10px 24px rgba(0,0,0,0.25);
        font-size: 1.02rem;
        color: rgba(226,232,240,0.95);
      }

      .rg-right-card {
        background: rgba(17,24,39,0.70);
        border: 1px solid rgba(148,163,184,0.18);
        border-radius: 16px;
        padding: 10px 10px 8px 10px;
        box-shadow: 0 12px 28px rgba(0,0,0,0.25);
      }

      .rg-right-card .small-reset div[data-testid="stButton"] > button{
        padding: 8px 10px !important;
        font-size: 13px !important;
        font-weight: 800 !important;
        background: rgba(51,65,85,0.55) !important;
      }

      .rg-tile {
        position: relative;
        background: rgba(17,24,39,0.68);
        border: 1px solid rgba(148,163,184,0.22);
        border-radius: 14px;
        padding: 34px 12px 8px 12px;
        box-shadow: 0 10px 26px rgba(0,0,0,0.25);
        overflow: hidden;
      }

      .rg-updated-pill {
        position: absolute;
        left: 12px;
        top: 10px;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 0.75rem;
        letter-spacing: 0.03em;
        color: rgba(148,163,184,0.95);
        border: 1px solid rgba(148,163,184,0.14);
        background: rgba(2,6,23,0.25);
      }

      .rg-tile::before {
        content: "";
        position: absolute;
        left: 10px; right: 10px;
        top: 34px;
        height: 10px;
        border-radius: 999px;
        background: rgba(148,163,184,0.10);
        border: 1px solid rgba(148,163,184,0.12);
      }

      .rg-tile.selected {
        border-color: rgba(0,255,102,0.95);
        background: linear-gradient(180deg, rgba(0,255,102,0.22), rgba(17,24,39,0.72));
        box-shadow: 0 0 0 1px rgba(0,255,102,0.18), 0 14px 30px rgba(0,0,0,0.30);
      }

      .rg-tile.selected::before {
        background: rgba(0,255,102,1);
        border-color: rgba(0,255,102,1);
        box-shadow: 0 0 18px rgba(0,255,102,0.55);
      }

      /* buttons */
      div[data-testid="stButton"] > button {
        width: 100%;
        border: 1px solid rgba(148,163,184,0.22) !important;
        background: rgba(31,41,55,0.90) !important;
        color: #e5e7eb !important;
        border-radius: 12px !important;
        padding: 12px 14px !important;
        font-size: 15px !important;
        font-weight: 850 !important;
        transition: 0.15s ease-in-out;
      }

      div[data-testid="stButton"] > button:hover {
        border-color: rgba(196,18,48,0.65) !important;
        color: #fff !important;
        transform: translateY(-1px);
      }

      /* mini refresh container - make it touch main button */
      .rg-mini {
        display: flex;
        justify-content: center;
        margin-top: -14px; /* pulls up to touch */
      }

      .rg-mini div[data-testid="stButton"] > button {
        width: auto !important;
        padding: 7px 12px !important;
        font-size: 12px !important;
        font-weight: 850 !important;
        border-radius: 999px !important;
        background: rgba(51,65,85,0.62) !important;
        border: 1px solid rgba(148,163,184,0.22) !important;
      }

      .rg-search-wrap {
        background: rgba(17,24,39,0.60);
        border: 1px solid rgba(148,163,184,0.18);
        border-radius: 14px;
        padding: 10px;
        box-shadow: 0 10px 26px rgba(0,0,0,0.18);
      }

      [data-testid="stDataFrame"], [data-testid="stDataEditor"] {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid rgba(148,163,184,0.18);
      }

      /* Force nowrap to avoid wrapping weirdness; true autosize isn't supported by Streamlit */
      [data-testid="stDataFrame"] td, [data-testid="stDataFrame"] th,
      [data-testid="stDataEditor"] td, [data-testid="stDataEditor"] th {
        white-space: nowrap !important;
      }

      /* Make links look clearer */
      a {
        text-decoration: none !important;
        font-weight: 800;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# HTTP session + soft pacing
# =========================================================
_thread_local = threading.local()
_rate_lock = threading.Lock()
_last_req_ts = 0.0


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def iso_utc(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def sleep_for_rate_limit():
    global _last_req_ts
    if RG_RPS <= 0:
        return
    min_interval = 1.0 / float(RG_RPS)
    with _rate_lock:
        t = time.time()
        wait = (_last_req_ts + min_interval) - t
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


def rg_request(method: str, url: str, *, headers=None, timeout=20, json_body=None):
    session = get_thread_session()
    headers = headers or rg_headers()

    backoff = 1
    for _attempt in range(1, 6):
        sleep_for_rate_limit()
        res = session.request(method, url, headers=headers, timeout=timeout, json=json_body)

        if res.status_code != 429:
            return res

        RATE_LIMIT_HIT.set()
        ra = res.headers.get("Retry-After")
        if ra and str(ra).isdigit():
            sleep_s = int(ra)
        else:
            sleep_s = backoff
            backoff = min(backoff * 2, 30)
        time.sleep(sleep_s)

    return res

# =========================================================
# Database
# =========================================================

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
                courier_status TEXT,
                courier_last_checked TEXT,
                received_first_seen TEXT
            )
            """
        )

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


def clear_db() -> bool:
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
        now_iso = iso_utc(now_utc())

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


def delete_rmas(rma_ids: List[str]):
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
            "SELECT json_data, last_fetched, courier_status, courier_last_checked, received_first_seen FROM rmas WHERE rma_id=?",
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


def get_all_open_from_db() -> List[dict]:
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


def get_local_ids_for_status(status: str) -> set:
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
            (scope, iso_utc(dt)),
        )
        conn.commit()
        conn.close()


def get_last_sync(scope: str) -> Optional[datetime]:
    with DB_LOCK:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT last_sync_iso FROM sync_logs WHERE scope=?", (scope,))
        row = c.fetchone()
        conn.close()

    if row and row[0]:
        try:
            return datetime.fromisoformat(row[0])
        except Exception:
            return None
    return None


init_db()

# =========================================================
# Courier status (Parcel Ninja)
# =========================================================

def collapse_spaces(s: str) -> str:
    return " ".join((s or "").split())


def check_courier_status(tracking_number: str) -> str:
    if not tracking_number or not PARCEL_NINJA_TOKEN:
        return "Unknown"

    try:
        url = f"https://optimise.parcelninja.com/shipment/track/{tracking_number}"
        headers = {"User-Agent": "Mozilla/5.0", "Authorization": f"Bearer {PARCEL_NINJA_TOKEN}"}
        res = requests.get(url, headers=headers, timeout=10)

        if res.status_code == 200:
            # Try JSON
            try:
                data = res.json()
                if isinstance(data, dict) and data.get("history"):
                    h0 = data["history"][0] or {}
                    return h0.get("status") or h0.get("description") or "Unknown"
                if isinstance(data, dict):
                    return data.get("status") or data.get("currentStatus") or "Unknown"
            except Exception:
                pass

            # Fallback: parse HTML
            content = res.text
            clean_html = re.sub(r"<(script|style).*?>.*?</(script|style)>", "", content, flags=re.DOTALL | re.IGNORECASE)

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
                        return collapse_spaces(" - ".join(cleaned))

            for kw in ["Courier Cancelled", "Booked Incorrectly", "Delivered", "Out For Delivery"]:
                if kw.lower() in clean_html.lower():
                    return kw

        if res.status_code == 404:
            return "Tracking Not Found"

        return f"Error {res.status_code}"

    except Exception:
        return "Check Failed"

# =========================================================
# ReturnGO fetching (incremental + cached)
# =========================================================

def fetch_rma_list(statuses: List[str], since_dt: Optional[datetime], list_full: bool = False) -> List[dict]:
    all_summaries: List[dict] = []
    cursor = None
    status_param = ",".join(statuses)

    updated_filter = ""
    if (since_dt is not None) and (not list_full):
        since_dt = since_dt - timedelta(minutes=SYNC_OVERLAP_MINUTES)
        updated_filter = f"&rma_updated_at=gte:{iso_utc(since_dt)}"

    while True:
        base_url = f"https://api.returngo.ai/rmas?pagesize=500&status={status_param}{updated_filter}"
        url = f"{base_url}&cursor={cursor}" if cursor else base_url

        res = rg_request("GET", url, timeout=25)
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

    return (now_utc() - last_dt) > timedelta(hours=CACHE_EXPIRY_HOURS)


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
            if (now_utc() - last_chk) <= timedelta(hours=COURIER_REFRESH_HOURS):
                return cached_status, cached_checked
        except Exception:
            pass

    status = check_courier_status(track_no)
    checked_iso = iso_utc(now_utc())
    return status, checked_iso


def fetch_rma_detail(rma_id: str, *, force: bool = False):
    cached = get_rma(rma_id)
    if cached and (not force) and (not should_refresh_detail(rma_id)):
        return cached[0]

    url = f"https://api.returngo.ai/rma/{rma_id}"
    res = rg_request("GET", url, timeout=25)
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


def get_incremental_since(statuses: List[str], full: bool) -> Optional[datetime]:
    if full:
        return None
    stamps = [get_last_sync(s) for s in statuses]
    stamps = [d for d in stamps if d]
    return min(stamps) if stamps else None


def perform_sync(
    statuses: Optional[List[str]] = None,
    *,
    full: bool = False,
    force: bool = False,
    list_full: bool = False,
):
    status_msg = st.empty()
    status_msg.info("Connecting to ReturnGO...")

    if statuses is None:
        statuses = ACTIVE_STATUSES

    since_dt = get_incremental_since(statuses, full)

    list_bar = st.progress(0, text="Fetching RMA list...")
    summaries = fetch_rma_list(statuses, since_dt, list_full=list_full)
    list_bar.progress(1.0, text=f"Fetched {len(summaries)} RMAs")
    time.sleep(0.10)
    list_bar.empty()

    api_ids = {s.get("rmaId") for s in summaries if s.get("rmaId")}

    # tidy cache only when syncing the main open set
    if set(statuses) == set(ACTIVE_STATUSES):
        local_active_ids = set()
        for stt in ACTIVE_STATUSES:
            local_active_ids |= get_local_ids_for_status(stt)
        stale = local_active_ids - api_ids
        delete_rmas(list(stale))

    if force:
        to_fetch = list(api_ids)
    else:
        to_fetch = [rid for rid in api_ids if should_refresh_detail(rid)]

    total = len(to_fetch)
    status_msg.info(f"Syncing {total} record(s)...")

    if total > 0:
        bar = st.progress(0, text="Downloading details...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = [ex.submit(fetch_rma_detail, rid, force=force) for rid in to_fetch]
            done = 0
            for _ in concurrent.futures.as_completed(futures):
                done += 1
                bar.progress(done / total, text=f"Syncing: {done}/{total}")
        bar.empty()

    now = now_utc()
    scope = ",".join(statuses)
    set_last_sync(scope, now)
    for s in statuses:
        set_last_sync(s, now)

    if full and set(statuses) == set(ACTIVE_STATUSES):
        set_last_sync("FULL", now)

    st.session_state["show_toast"] = True
    status_msg.success("Sync complete")
    st.rerun()

# =========================================================
# Mutations (tracking + comments)
# =========================================================

def push_tracking_update(rma_id: str, shipment_id: str, tracking_number: str):
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
            timeout=18,
            json_body=payload,
        )
        if res.status_code == 200:
            fetch_rma_detail(rma_id, force=True)
            return True, "Success"
        return False, f"API Error {res.status_code}: {res.text}"
    except Exception as e:
        return False, str(e)


def push_comment_update(rma_id: str, comment_text: str):
    headers = {**rg_headers(), "Content-Type": "application/json"}
    payload = {"text": comment_text, "isPublic": False}

    try:
        res = rg_request(
            "POST",
            f"https://api.returngo.ai/rma/{rma_id}/comment",
            headers=headers,
            timeout=18,
            json_body=payload,
        )
        if res.status_code in (200, 201):
            fetch_rma_detail(rma_id, force=True)
            return True, "Success"
        return False, f"API Error {res.status_code}: {res.text}"
    except Exception as e:
        return False, str(e)

# =========================================================
# Helpers
# =========================================================

def get_event_date_iso(rma_payload: dict, event_name: str) -> str:
    summary = rma_payload.get("rmaSummary", {}) or {}
    for e in (summary.get("events") or []):
        if e.get("eventName") == event_name and e.get("eventDate"):
            return str(e["eventDate"])
    return ""


def get_received_date_iso(rma_payload: dict) -> str:
    # Try common event names first
    for name in ("SHIPMENT_RECEIVED", "RMA_RECEIVED", "RMA_STATUS_RECEIVED"):
        dt = get_event_date_iso(rma_payload, name)
        if dt:
            return dt

    # Then look for a timeline entry mentioning received
    comments = rma_payload.get("comments", []) or []
    for c in comments:
        txt = (c.get("htmlText") or "").lower()
        if "received" in txt and c.get("datetime"):
            return str(c.get("datetime"))

    # Fallback to local first-seen received timestamp
    return str(rma_payload.get("_local_received_first_seen") or "")


def pretty_resolution(rt: str) -> str:
    if not rt:
        return ""

    # Split CamelCase without backref replacement strings
    def repl(m):
        return m.group(1) + " " + m.group(2)

    out = re.sub(r"([a-z])([A-Z])", repl, rt).strip()
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
    txs = rma_payload.get("transactions") or []
    for t in txs:
        if (t.get("status") or "").lower() == "success":
            ttype = (t.get("type") or "").lower()
            if "storecredit" in ttype:
                return "Store credit"
            if "payment" in ttype:
                return "Refund"
            return "Processed"

    ex_orders = rma_payload.get("exchangeOrders") or []
    if ex_orders:
        return "Exchange released"

    comments = rma_payload.get("comments") or []
    for c in comments:
        txt = (c.get("htmlText") or "").lower()
        if "credited" in txt:
            return "Store credit"
        if ("refund" in txt) and ("transaction" in txt or "total refund" in txt):
            return "Refund"
        if ("exchange order" in txt) and ("released" in txt):
            return "Exchange released"

    return "No"


def days_since(date_yyyy_mm_dd: str) -> str:
    try:
        d = datetime.strptime(date_yyyy_mm_dd, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return str((now_utc() - d).days)
    except Exception:
        return ""

# =========================================================
# UI state
# =========================================================
if "selected_filters" not in st.session_state:
    st.session_state.selected_filters = set(["AllOpen"])

if "search_query_input" not in st.session_state:
    st.session_state.search_query_input = ""

if "filter_requested_dates" not in st.session_state:
    st.session_state.filter_requested_dates = []

if "filter_resolution" not in st.session_state:
    st.session_state.filter_resolution = "All"

if st.session_state.get("show_toast"):
    st.toast("Sync complete", icon="🔄")
    st.session_state["show_toast"] = False

if RATE_LIMIT_HIT.is_set():
    st.warning(
        "ReturnGO rate limit reached (429). Sync is slowing down and retrying. "
        "If this happens often, sync less frequently or request a higher quota key."
    )
    RATE_LIMIT_HIT.clear()

# =========================================================
# Header (sticky)
# =========================================================
st.markdown("<div class='rg-sticky'>", unsafe_allow_html=True)

left, right = st.columns([3.4, 1.1], vertical_alignment="top")
with left:
    st.markdown("<div class='rg-title'>Levi's ReturnGO Ops Dashboard</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='rg-subpill'><b>CONNECTED TO:</b> {STORE_URL.upper()} <span style='opacity:.55'>|</span> <b>CACHE:</b> {CACHE_EXPIRY_HOURS}h</div>",
        unsafe_allow_html=True,
    )

with right:
    st.markdown("<div class='rg-right-card'>", unsafe_allow_html=True)
    if st.button("🔄 Sync Dashboard", key="btn_sync_all", use_container_width=True):
        perform_sync()

    st.markdown("<div class='small-reset'>", unsafe_allow_html=True)
    if st.button("🗑️ Reset Cache", key="btn_reset", use_container_width=True):
        if clear_db():
            st.success("Cache cleared")
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# Load data
# =========================================================
raw_data = get_all_open_from_db()
if not raw_data:
    st.warning("Database empty. Click Sync Dashboard to start.")
    st.stop()

# =========================================================
# Build rows + counts
# =========================================================
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

search_query = (st.session_state.get("search_query_input") or "").strip().lower()

for rma in raw_data:
    summary = rma.get("rmaSummary", {}) or {}
    shipments = rma.get("shipments", []) or []
    comments = rma.get("comments", []) or []

    status = summary.get("status", "Unknown")
    rma_id = str(summary.get("rmaId", ""))
    order_name = summary.get("order_name", summary.get("orderName", ""))

    track_nums = [s.get("trackingNumber") for s in shipments if s.get("trackingNumber")]
    track_str = ", ".join(track_nums) if track_nums else ""
    shipment_id = shipments[0].get("shipmentId") if shipments else ""

    local_tracking_status = (rma.get("_local_courier_status") or "")

    # Link values (LinkColumn uses this as the href; display text is extracted via regex)
    rma_url = f"https://app.returngo.ai/dashboard/returns?filter_status=open&rmaid={rma_id}"
    track_url = f"https://portal.thecourierguy.co.za/track?ref={track_nums[0]}" if track_nums else ""

    requested_iso = get_event_date_iso(rma, "RMA_CREATED") or (summary.get("createdAt") or rma.get("createdAt") or "")
    approved_iso = get_event_date_iso(rma, "RMA_APPROVED")
    received_iso = get_received_date_iso(rma)

    req_date_short = str(requested_iso)[:10] if requested_iso else ""
    app_date_short = str(approved_iso)[:10] if approved_iso else ""
    rcv_date_short = str(received_iso)[:10] if received_iso else ""

    resolution_type = get_resolution_type(rma)
    actioned_label = resolution_actioned_label(rma)

    if status in counts:
        counts[status] += 1

    is_nt = (status == "Approved" and not track_str)
    is_fg = any("flagged" in (c.get("htmlText", "").lower()) for c in comments)
    is_cc = ("courier cancelled" in local_tracking_status.lower())
    is_ad = (status == "Approved" and ("delivered" in local_tracking_status.lower()))

    is_ra = (actioned_label != "No")
    is_nra = (not is_ra) and (status == "Received")

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

    if search_query:
        if (search_query not in rma_id.lower()) and (search_query not in str(order_name).lower()) and (search_query not in str(track_str).lower()):
            continue

    processed_rows.append(
        {
            "RMA ID": rma_url,
            "Order": order_name,
            "Current Status": status,
            "Tracking Number": track_url,
            "Tracking Status": local_tracking_status,
            "Requested date": req_date_short or "N/A",
            "Approved date": app_date_short or "N/A",
            "Received date": rcv_date_short or "N/A",
            "Days since requested": days_since(req_date_short) if req_date_short else "",
            "resolutionType": resolution_type or "N/A",
            "Resolution actioned": actioned_label,
            "shipment_id": shipment_id,
            "DisplayTrack": track_str,
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
if df_view.empty:
    st.info("No matching records found.")
    st.stop()

# =========================================================
# Tiles (multi-select + mini refresh)
# =========================================================

def toggle_filter(key: str):
    sel = set(st.session_state.selected_filters)

    if key == "AllOpen":
        st.session_state.selected_filters = set(["AllOpen"])
        st.rerun()

    if "AllOpen" in sel:
        sel.remove("AllOpen")

    if key in sel:
        sel.remove(key)
        if not sel:
            sel = set(["AllOpen"])
    else:
        sel.add(key)

    st.session_state.selected_filters = sel
    st.rerun()


def is_selected(key: str) -> bool:
    return key in st.session_state.selected_filters


def tile(title: str, key: str, count: int, refresh_fn):
    last = get_last_sync(key) or get_last_sync(",".join(ACTIVE_STATUSES))
    last_txt = last.strftime("%H:%M") if last else "-"

    tile_cls = "rg-tile selected" if is_selected(key) else "rg-tile"
    st.markdown(f"<div class='{tile_cls}'>", unsafe_allow_html=True)
    st.markdown(f"<div class='rg-updated-pill'>UPDATED: {last_txt}</div>", unsafe_allow_html=True)

    if st.button(f"{title} [**{count}**]", key=f"tile_{key}", use_container_width=True):
        toggle_filter(key)

    st.markdown("<div class='rg-mini'>", unsafe_allow_html=True)
    if st.button("↻ Refresh", key=f"sync_{key}"):
        refresh_fn()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


row0 = st.columns(5)
with row0[0]:
    tile("📋 VIEW ALL OPEN", "AllOpen", len(df_view), lambda: perform_sync(ACTIVE_STATUSES, force=True, list_full=True))
with row0[1]:
    tile("⏳ PENDING", "Pending", counts["Pending"], lambda: perform_sync(["Pending"], force=True, list_full=True))
with row0[2]:
    tile("✅ APPROVED", "Approved", counts["Approved"], lambda: perform_sync(["Approved"], force=True, list_full=True))
with row0[3]:
    tile("📦 RECEIVED", "Received", counts["Received"], lambda: perform_sync(["Received"], force=True, list_full=True))
with row0[4]:
    tile("🚫 NO TRACKING", "NoTrack", counts["NoTrack"], lambda: perform_sync(["Approved"], force=True, list_full=True))

row1 = st.columns(4)
with row1[0]:
    tile("🚩 FLAGGED", "Flagged", counts["Flagged"], lambda: perform_sync(ACTIVE_STATUSES, force=True, list_full=False))
with row1[1]:
    tile("🧾 COURIER CANCELLED", "CourierCancelled", counts["CourierCancelled"], lambda: perform_sync(ACTIVE_STATUSES, force=True, list_full=False))
with row1[2]:
    tile("📬 APPROVED > DELIVERED", "ApprovedDelivered", counts["ApprovedDelivered"], lambda: perform_sync(["Approved"], force=True, list_full=True))
with row1[3]:
    tile("💸 RESOLUTION ACTIONED", "ResolutionActioned", counts["ResolutionActioned"], lambda: perform_sync(ACTIVE_STATUSES, force=True, list_full=False))

row2 = st.columns(4)
with row2[0]:
    tile("🕒 NO RESOLUTION ACTIONED", "NoResolutionActioned", counts["NoResolutionActioned"], lambda: perform_sync(["Received"], force=True, list_full=True))
with row2[1]:
    st.write("")
with row2[2]:
    st.write("")
with row2[3]:
    st.write("")

st.write("")

# =========================================================
# Search + filters (expanded by default)
# =========================================================
st.markdown("<div class='rg-search-wrap'>", unsafe_allow_html=True)

sc1, sc2, sc3 = st.columns([8, 1, 2], vertical_alignment="center")
with sc1:
    st.text_input(
        "Search",
        placeholder="Search Order, RMA, or Tracking...",
        label_visibility="collapsed",
        key="search_query_input",
    )
with sc2:
    def clear_search():
        st.session_state.search_query_input = ""
    st.button("✖", help="Clear search", on_click=clear_search, use_container_width=True)
with sc3:
    if st.button("🧹 Clear filters", use_contai
