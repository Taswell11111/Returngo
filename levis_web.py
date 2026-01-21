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
from io import BytesIO

# ==========================================
# 1. CONFIGURATION
# ==========================================
st.set_page_config(page_title="Levi's ReturnGO Ops", layout="wide", page_icon="📤")

# --- Preserve scroll position across reruns (prevents page jump after dialogs close) ---
# Add delayed restore too (Streamlit can re-render after initial JS runs)
components.html(
    """
    <script>
    (function() {
      function restoreScroll() {
        const y = window.localStorage.getItem("scrollY");
        if (y) window.scrollTo(0, parseInt(y));
      }
      // Restore early + delayed (helps after dialogs/reruns)
      restoreScroll();
      setTimeout(restoreScroll, 150);
      setTimeout(restoreScroll, 450);

      window.addEventListener("scroll", () => {
        window.localStorage.setItem("scrollY", window.scrollY);
      }, { passive: true });

      // Before unload as well
      window.addEventListener("beforeunload", () => {
        window.localStorage.setItem("scrollY", window.scrollY);
      });
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
CACHE_EXPIRY_HOURS = 24          # requested: 24 hours
COURIER_REFRESH_HOURS = 12
MAX_WORKERS = 3
RG_RPS = 2
SYNC_OVERLAP_MINUTES = 5

OPEN_STATUSES = ["Pending", "Approved", "Received"]
DONE_STATUSES = ["Done"]
DASHBOARD_STATUSES = OPEN_STATUSES + DONE_STATUSES

# Thread-safe UI signal for 429s (never call st.* from worker threads)
RATE_LIMIT_HIT = threading.Event()

# ==========================================
# 1b. STYLING (enhanced layout + mini refresh)
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

      .app-subtitle {
        color: #cbd5e1;
        font-size: 0.95rem;
        margin-top: -0.35rem;
      }

      .card {
        background: rgba(17, 24, 39, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 14px;
        padding: 14px 14px 10px 14px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.25);
      }

      .metric-card { padding: 12px 12px 10px 12px; }

      .updated-pill {
        display:inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        border: 1px solid rgba(148,163,184,0.25);
        background: rgba(31, 41, 55, 0.55);
        font-size: 0.78rem;
        color: rgba(148,163,184,0.95);
        margin-bottom: 8px;
      }

      /* Main filter buttons */
      div.stButton > button {
        width: 100%;
        border: 1px solid rgba(148, 163, 184, 0.25) !important;
        background: rgba(31, 41, 55, 0.9) !important;
        color: #e5e7eb !important;
        border-radius: 10px !important;
        padding: 12px 14px !important;
        font-size: 15px !important;
        font-weight: 800 !important;
        transition: 0.15s ease-in-out;
      }
      div.stButton > button:hover {
        border-color: rgba(196,18,48,0.7) !important;
        color: #fff !important;
        transform: translateY(-1px);
      }

      /* Mini refresh buttons: smaller + "touch" main button visually */
      .mini-btn div.stButton > button {
        padding: 6px 10px !important;
        font-size: 12px !important;
        font-weight: 800 !important;
        border-radius: 10px !important;
        margin-top: -6px !important; /* touch */
      }

      /* Data editor tweaks */
      [data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid rgba(148,163,184,0.18);
      }

      /* Encourage autofit-ish behaviour */
      [data-testid="stDataFrame"] * {
        white-space: nowrap;
      }

      /* Dialog box styling */
      div[data-testid="stDialog"] {
        background-color: rgba(17, 24, 39, 0.95);
        border: 1px solid rgba(196,18,48,0.55);
        border-radius: 16px;
      }

      /* Compact action bar */
      .action-bar {
        display: flex;
        gap: 10px;
        justify-content: flex-end;
        align-items: center;
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

        existing_cols = {row[1] for row in c.execute("PRAGMA table_info(rmas)").fetchall()}

        def add_col(col_name: str, col_type: str):
            if col_name not in existing_cols:
                try:
                    c.execute(f"ALTER TABLE rmas ADD COLUMN {col_name} {col_type}")
                except Exception:
                    pass

        add_col("courier_last_checked", "TEXT")
        add_col("received_first_seen", "TEXT")

        c.execute("""
            CREATE TABLE IF NOT EXISTS sync_logs (
                scope TEXT PRIMARY KEY,
                last_sync_iso TEXT
            )
        """)

        sync_cols = {row[1] for row in c.execute("PRAGMA table_info(sync_logs)").fetchall()}
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


def get_all_dashboard_from_db():
    with DB_LOCK:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        placeholders = ",".join("?" for _ in DASHBOARD_STATUSES)
        c.execute(
            f"""
            SELECT json_data, courier_status, courier_last_checked, received_first_seen
            FROM rmas
            WHERE store_url=? AND status IN ({placeholders})
            """,
            (STORE_URL, *DASHBOARD_STATUSES),
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

        if res.status_code == 200:
            # JSON first
            try:
                data = res.json()
                if isinstance(data, dict) and data.get("history"):
                    v = data["history"][0].get("status") or data["history"][0].get("description")
                    return v or "Unknown"
                v = data.get("status") or data.get("currentStatus")
                return v or "Unknown"
            except Exception:
                pass

            # HTML fallback
            content = res.text
            clean_html = re.sub(r"<(script|style).*?</\1>", "", content, flags=re.DOTALL | re.IGNORECASE)
            history_section = re.search(
                r"<table[^>]*?tracking-history.*?>(.*?)</table>",
                clean_html,
                re.DOTALL | re.IGNORECASE
            )
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
# 5. RETURNGO FETCHING (incremental + cached + dirty refresh)
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


def maybe_refresh_courier(rma_payload: dict, *, force: bool = False) -> Tuple[Optional[str], Optional[str]]:
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

    if (not force) and cached_checked:
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


def fetch_rma_detail(rma_id: str, *, force: bool = False, force_courier: bool = False):
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

    courier_status, courier_checked = maybe_refresh_courier(data, force=force_courier)

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


def perform_sync(
    statuses=None,
    *,
    full=False,
    scope_name: Optional[str] = None,
    force_ids=None,
    force_courier=False
):
    """
    Correct sequence + "dirty refresh":
    - If incremental (since_dt exists), ALL returned RMAs are refreshed (even if cache is fresh).
    - force_ids can force-refresh specific RMAs (bucket refresh).
    """
    status_msg = st.empty()
    status_msg.info("⏳ Connecting to ReturnGO...")

    if statuses is None:
        statuses = DASHBOARD_STATUSES

    force_ids = set(force_ids or [])

    since_dt = get_incremental_since(statuses, full)

    list_bar = st.progress(0, text="Fetching RMA list from ReturnGO...")
    summaries = fetch_rma_list(statuses, since_dt)
    list_bar.progress(1.0, text=f"Fetched {len(summaries)} RMAs")
    time.sleep(0.10)
    list_bar.empty()

    api_ids = {s.get("rmaId") for s in summaries if s.get("rmaId")}

    # tidy cache only for OPEN statuses (Pending/Approved/Received)
    if set(statuses) == set(OPEN_STATUSES) and not full:
        local_open_ids = set()
        for stt in OPEN_STATUSES:
            local_open_ids |= get_local_ids_for_status(stt)
        stale = local_open_ids - api_ids
        delete_rmas(stale)

    # Decide what to refresh
    to_fetch = set()
    if since_dt is not None:
        # list already represents "updated since last sync" -> refresh all
        to_fetch |= api_ids
        force_flag = True
    else:
        to_fetch |= {rid for rid in api_ids if should_refresh_detail(rid)}
        force_flag = False

    # also force-refresh bucket members that are still in this status list
    to_fetch |= (force_ids & api_ids)

    total = len(to_fetch)
    status_msg.info(f"⏳ Syncing {total} records...")

    if total > 0:
        bar = st.progress(0, text="Downloading Details...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = []
            for rid in to_fetch:
                futures.append(
                    ex.submit(
                        fetch_rma_detail,
                        rid,
                        force=(force_flag or (rid in force_ids)),
                        force_courier=force_courier,
                    )
                )
            done = 0
            for _ in concurrent.futures.as_completed(futures):
                done += 1
                bar.progress(done / total, text=f"Syncing: {done}/{total}")
        bar.empty()

    now = _now_utc()
    scope = scope_name or ",".join(statuses)
    set_last_sync(scope, now)
    for s in statuses:
        set_last_sync(s, now)

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
        res = rg_request(
            "PUT",
            f"https://api.returngo.ai/shipment/{shipment_id}",
            headers=headers,
            timeout=15,
            json_body=payload,
        )
        if res.status_code == 200:
            fresh = fetch_rma_detail(rma_id, force=True, force_courier=True)
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
            fresh = fetch_rma_detail(rma_id, force=True)
            return True, "Success" if fresh else "Posted"
        return False, f"API Error {res.status_code}: {res.text}"
    except Exception as e:
        return False, str(e)


# ==========================================
# 7. Helpers: dates + resolution type + resolution actioned
# ==========================================
def get_event_date_iso(rma_payload: dict, event_name: str) -> str:
    summary = rma_payload.get("rmaSummary", {}) or {}
    for e in (summary.get("events") or []):
        if e.get("eventName") == event_name and e.get("eventDate"):
            return str(e["eventDate"])
    return ""


def get_received_date_iso(rma_payload: dict) -> str:
    # 1) Events (if ReturnGO ever includes them)
    for name in ("SHIPMENT_RECEIVED", "RMA_RECEIVED", "RMA_STATUS_RECEIVED", "RECEIVED"):
        dt = get_event_date_iso(rma_payload, name)
        if dt:
            return dt

    # 2) Comments: look for 'Shipment RECEIVED' automation/system comment (your examples use this)
    comments = rma_payload.get("comments", []) or []
    received_hits = []
    for c in comments:
        html = (c.get("htmlText") or "").lower()
        if "shipment" in html and "received" in html:
            if c.get("datetime"):
                received_hits.append(c["datetime"])

    if received_hits:
        def _dt(v):
            try:
                return datetime.fromisoformat(v.replace("Z", "+00:00"))
            except Exception:
                return datetime.max.replace(tzinfo=timezone.utc)
        received_hits.sort(key=_dt)
        return str(received_hits[0])

    # 3) Fallback: local first-seen timestamp
    local = rma_payload.get("_local_received_first_seen")
    if local:
        return str(local)

    return ""


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


def get_resolution_actioned(rma_payload: dict) -> str:
    """
    Resolution actioned:
    - Refund/store credit: transactions[] with status Success
    - Exchange: exchangeOrders[] exists OR items[].exchangeItems exists
    - Else: No
    """
    txs = rma_payload.get("transactions", []) or []
    success = [t for t in txs if str(t.get("status", "")).lower() == "success"]

    if success:
        def _dt(v: str):
            try:
                return datetime.fromisoformat((v or "").replace("Z", "+00:00"))
            except Exception:
                return datetime.min.replace(tzinfo=timezone.utc)

        success.sort(key=lambda t: _dt(t.get("createdAt") or ""), reverse=True)
        t = success[0]

        tx_type = (t.get("type") or "").strip()
        tx_id = t.get("id")
        amt = (t.get("amount") or {}).get("amount")
        cur = (t.get("amount") or {}).get("currency") or ""

        label_map = {
            "RefundToPaymentMethod": "Refund processed",
            "RefundToStoreCredit": "Store credit issued",
            "RefundToGiftCard": "Gift card issued",
            "Refund": "Refund processed",
        }
        label = label_map.get(tx_type, tx_type or "Transaction success")

        extras = []
        if amt is not None:
            extras.append(f"R{amt} {cur}".strip())
        if tx_id:
            extras.append(f"TX:{tx_id}")

        return f"{label}" + (f" ({', '.join(extras)})" if extras else "")

    ex_orders = rma_payload.get("exchangeOrders", []) or []
    if ex_orders:
        names = [o.get("orderName") for o in ex_orders if o.get("orderName")]
        if names:
            return f"Exchange processed ({', '.join(names)})"
        return "Exchange processed"

    items = rma_payload.get("items", []) or []
    if any((i.get("exchangeItems") or []) for i in items if isinstance(i, dict)):
        return "Exchange processed"

    return "No"


# ==========================================
# 8. UI STATE
# ==========================================
if "modal_rma" not in st.session_state:
    st.session_state.modal_rma = None
if "modal_action" not in st.session_state:
    st.session_state.modal_action = None
if "table_key" not in st.session_state:
    st.session_state.table_key = 0

if "filter_status" not in st.session_state:
    st.session_state.filter_status = "AllOpen"

if st.session_state.get("show_toast"):
    st.toast("✅ Sync Complete!", icon="🔄")
    st.session_state["show_toast"] = False

if RATE_LIMIT_HIT.is_set():
    st.warning(
        "ReturnGO rate limit reached (429). Sync is slowing down and retrying. "
        "If this happens often, sync less frequently or request a higher quota key."
    )
    RATE_LIMIT_HIT.clear()


def set_filter(f):
    st.session_state.filter_status = f
    st.rerun()


# ==========================================
# 9. Dialogs
# ==========================================
@st.dialog("Update Tracking")
def show_update_tracking_dialog(record):
    st.markdown(f"### Update Tracking for `{record['RMA_ID_RAW']}`")
    with st.form("upd_track"):
        new_track = st.text_input("New Tracking Number", value=record.get("DisplayTrack", ""))
        if st.form_submit_button("Save Changes"):
            if not record.get("shipment_id"):
                st.error("No Shipment ID.")
            else:
                ok, msg = push_tracking_update(record["RMA_ID_RAW"], record["shipment_id"], new_track)
                if ok:
                    st.success("Updated!")
                    time.sleep(0.20)
                    st.rerun()
                else:
                    st.error(msg)


@st.dialog("View Timeline")
def show_timeline_dialog(record):
    st.markdown(f"### Timeline for `{record['RMA_ID_RAW']}`")

    with st.expander("➕ Add Comment", expanded=False):
        with st.form("add_comm"):
            comment_text = st.text_area("New Note")
            if st.form_submit_button("Post Comment"):
                ok, msg = push_comment_update(record["RMA_ID_RAW"], comment_text)
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


# Trigger modal after rerun (keeps Streamlit stable)
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
# 10. Header + global actions (no "full rebuild")
# ==========================================
col1, col2 = st.columns([3, 1], vertical_alignment="top")

with col1:
    st.title("Levi's ReturnGO Ops Dashboard")
    st.markdown(
        f"<div class='app-subtitle'><b>CONNECTED TO:</b> {STORE_URL.upper()} &nbsp;|&nbsp; "
        f"<b>CACHE:</b> {CACHE_EXPIRY_HOURS}h</div>",
        unsafe_allow_html=True,
    )

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    if st.button("🔄 Sync Dashboard", key="btn_sync_all", use_container_width=True):
        perform_sync(DASHBOARD_STATUSES, scope_name="Dashboard")
    if st.button("🗑️ Reset Cache", key="btn_reset", use_container_width=True):
        if clear_db():
            st.success("Cache cleared!")
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

st.write("")

# ==========================================
# 11. Load + process cached RMAs (all for counts, then search for view)
# ==========================================
raw_data = get_all_dashboard_from_db()

def rma_dashboard_url(rma_id: str) -> str:
    return f"https://app.returngo.ai/dashboard/returns?filter_status=open&rmaid={rma_id}"

processed_rows_all = []

counts = {
    "Pending": 0, "Approved": 0, "Received": 0, "Done": 0,
    "NoTrack": 0, "Flagged": 0,
    "CourierCancelled": 0, "ApprovedDelivered": 0,
    "ResolutionActioned": 0, "NoResolutionActioned": 0,
}

search_query = st.session_state.get("search_query_input", "")

for rma in raw_data:
    summary = rma.get("rmaSummary", {}) or {}
    shipments = rma.get("shipments", []) or []
    comments = rma.get("comments", []) or []

    status = summary.get("status", "Unknown")
    rma_id = str(summary.get("rmaId", "N/A"))
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

    resolution_actioned_text = get_resolution_actioned(rma)
    is_actioned = (resolution_actioned_text not in ("No", "N/A", ""))

    is_nt = (status == "Approved" and not track_str)
    is_fg = any("flagged" in (c.get("htmlText", "").lower()) for c in comments)
    is_cc = bool(local_tracking_status) and ("courier cancelled" in local_tracking_status.lower())
    is_ad = (status == "Approved") and ("delivered" in (local_tracking_status or "").lower())
    is_no_actioned = (status == "Received") and (not is_actioned)

    if status in counts:
        counts[status] += 1
    if is_nt:
        counts["NoTrack"] += 1
    if is_fg:
        counts["Flagged"] += 1
    if is_cc:
        counts["CourierCancelled"] += 1
    if is_ad:
        counts["ApprovedDelivered"] += 1
    if is_actioned:
        counts["ResolutionActioned"] += 1
    if is_no_actioned:
        counts["NoResolutionActioned"] += 1

    processed_rows_all.append(
        {
            "No": "",
            # We display URL but keep raw ID separately for logic/search/modals
            "RMA ID": rma_dashboard_url(rma_id),
            "RMA_ID_RAW": rma_id,
            "Order": order_name,
            "Current Status": status,
            "Tracking Number": track_link_url,
            "Tracking Status": local_tracking_status,
            "Requested date": str(requested_iso)[:10] if requested_iso else "N/A",
            "Approved date": str(approved_iso)[:10] if approved_iso else "N/A",
            "Received date": str(received_iso)[:10] if received_iso else "N/A",
            "resolutionType": resolution_type if resolution_type else "N/A",
            "Resolution actioned": resolution_actioned_text if resolution_actioned_text else "No",
            "Update Tracking Number": False,
            "View Timeline": False,
            "DisplayTrack": track_str,
            "shipment_id": shipment_id,
            "full_data": rma,
            "is_nt": is_nt,
            "is_fg": is_fg,
            "is_cc": is_cc,
            "is_ad": is_ad,
            "is_actioned": is_actioned,
            "is_no_actioned": is_no_actioned,
        }
    )

df_all = pd.DataFrame(processed_rows_all)

# Search is applied to view only (counts stay accurate)
if search_query:
    sq = search_query.lower()
    df_view = df_all[
        df_all["RMA_ID_RAW"].astype(str).str.lower().str.contains(sq, na=False)
        | df_all["Order"].astype(str).str.lower().str.contains(sq, na=False)
        | df_all["DisplayTrack"].astype(str).str.lower().str.contains(sq, na=False)
    ].copy()
else:
    df_view = df_all.copy()

# ==========================================
# 12. Bucket refresh helpers (mini refresh buttons)
# ==========================================
def updated_pill(scope: str) -> str:
    ts = get_last_sync(scope)
    if not ts:
        return "<div class='updated-pill'>UPDATED: -</div>"
    return f"<div class='updated-pill'>UPDATED: {ts.strftime('%H:%M')}</div>"


def bucket_force_ids(df_source: pd.DataFrame, bucket: str) -> set:
    if df_source.empty:
        return set()
    col = "RMA_ID_RAW"
    if bucket == "NoTrack":
        return set(df_source[df_source["is_nt"] == True][col].astype(str))
    if bucket == "Flagged":
        return set(df_source[df_source["is_fg"] == True][col].astype(str))
    if bucket == "CourierCancelled":
        return set(df_source[df_source["is_cc"] == True][col].astype(str))
    if bucket == "ApprovedDelivered":
        return set(df_source[df_source["is_ad"] == True][col].astype(str))
    if bucket == "ResolutionActioned":
        return set(df_source[df_source["is_actioned"] == True][col].astype(str))
    if bucket == "NoResolutionActioned":
        return set(df_source[df_source["is_no_actioned"] == True][col].astype(str))
    if bucket in OPEN_STATUSES or bucket in DONE_STATUSES:
        return set(df_source[df_source["Current Status"] == bucket][col].astype(str))
    return set()


def refresh_bucket(bucket: str, df_source: pd.DataFrame):
    # Choose statuses likely to affect the bucket
    if bucket in ("NoTrack",):
        statuses = ["Approved"]
        force_courier = False
    elif bucket in ("CourierCancelled", "ApprovedDelivered"):
        statuses = ["Approved"]
        force_courier = True  # refresh ParcelNinja status
    elif bucket == "Flagged":
        statuses = OPEN_STATUSES
        force_courier = False
    elif bucket == "ResolutionActioned":
        statuses = ["Received", "Done"]
        force_courier = False
    elif bucket == "NoResolutionActioned":
        statuses = ["Received"]
        force_courier = False
    elif bucket in OPEN_STATUSES:
        statuses = [bucket]
        force_courier = False
    elif bucket == "Done":
        statuses = ["Done"]
        force_courier = False
    else:
        statuses = DASHBOARD_STATUSES
        force_courier = False

    force_ids = bucket_force_ids(df_source, bucket)
    perform_sync(statuses, scope_name=bucket, force_ids=force_ids, force_courier=force_courier)


def metric_block(title: str, count: int, filter_key: str, refresh_bucket_name: str):
    st.markdown("<div class='card metric-card'>", unsafe_allow_html=True)
    st.markdown(updated_pill(refresh_bucket_name), unsafe_allow_html=True)

    if st.button(f"{title}\n{count}", key=f"btn_{filter_key}", use_container_width=True):
        set_filter(filter_key)

    st.markdown("<div class='mini-btn'>", unsafe_allow_html=True)
    if st.button("🔄 Refresh", key=f"ref_{filter_key}", help="Refresh this list", use_container_width=True):
        refresh_bucket(refresh_bucket_name, df_all)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ==========================================
# 13. Metrics / Preset filters (with mini refresh buttons)
# ==========================================
r1 = st.columns(5)
with r1[0]:
    metric_block("PENDING", counts["Pending"], "Pending", "Pending")
with r1[1]:
    metric_block("APPROVED", counts["Approved"], "Approved", "Approved")
with r1[2]:
    metric_block("RECEIVED", counts["Received"], "Received", "Received")
with r1[3]:
    metric_block("NO TRACKING", counts["NoTrack"], "NoTrack", "NoTrack")
with r1[4]:
    metric_block("🚩 FLAGGED", counts["Flagged"], "Flagged", "Flagged")

st.write("")

r2 = st.columns(4)
with r2[0]:
    metric_block("COURIER CANCELLED", counts["CourierCancelled"], "CourierCancelled", "CourierCancelled")
with r2[1]:
    metric_block("APPROVED > DELIVERED", counts["ApprovedDelivered"], "ApprovedDelivered", "ApprovedDelivered")
with r2[2]:
    metric_block("RESOLUTION ACTIONED", counts["ResolutionActioned"], "ResolutionActioned", "ResolutionActioned")
with r2[3]:
    metric_block("NO RESOLUTION ACTIONED", counts["NoResolutionActioned"], "NoResolutionActioned", "NoResolutionActioned")

st.write("")

# ==========================================
# 14. Search + Filters bar + Export/Copy (replaces "top right corner" menu)
# ==========================================
top_left, top_right = st.columns([6, 4], vertical_alignment="center")

with top_left:
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

with top_right:
    st.markdown("<div class='action-bar'>", unsafe_allow_html=True)

    # Export XLSX (current filtered view, excluding non-display helper columns)
    def _export_xlsx_bytes(df_export: pd.DataFrame) -> bytes:
        out = BytesIO()
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            df_export.to_excel(writer, index=False, sheet_name="Levis_RMAs")
        out.seek(0)
        return out.read()

    # We will decide export frame later (after filtering)
    st.markdown("</div>", unsafe_allow_html=True)

st.divider()

# ==========================================
# 15. Column filters (header-like filters)
# ==========================================
with st.expander("🔎 Column Filters", expanded=False):
    f1, f2, f3, f4, f5 = st.columns([1.2, 1.6, 1.8, 1.8, 2.0], vertical_alignment="center")

    with f1:
        status_filter = st.multiselect(
            "Status",
            options=["Pending", "Approved", "Received", "Done"],
            default=[],
        )
    with f2:
        resolution_filter = st.multiselect(
            "resolutionType",
            options=sorted([x for x in df_view["resolutionType"].dropna().unique().tolist() if x]),
            default=[],
        )
    with f3:
        actioned_filter = st.selectbox(
            "Resolution actioned?",
            options=["All", "Yes", "No"],
            index=0
        )
    with f4:
        track_status_contains = st.text_input("Tracking Status contains", value="")
    with f5:
        order_contains = st.text_input("Order contains", value="")

# ==========================================
# 16. Apply preset filter + column filters
# ==========================================
if df_all.empty:
    st.warning("Database empty. Click Sync Dashboard to start.")
    st.stop()

f_stat = st.session_state.filter_status
display_df = df_view

# Preset filters
if f_stat == "Pending":
    display_df = display_df[display_df["Current Status"] == "Pending"]
elif f_stat == "Approved":
    display_df = display_df[display_df["Current Status"] == "Approved"]
elif f_stat == "Received":
    display_df = display_df[display_df["Current Status"] == "Received"]
elif f_stat == "Done":
    display_df = display_df[display_df["Current Status"] == "Done"]
elif f_stat == "NoTrack":
    display_df = display_df[display_df["is_nt"] == True]
elif f_stat == "Flagged":
    display_df = display_df[display_df["is_fg"] == True]
elif f_stat == "CourierCancelled":
    display_df = display_df[display_df["is_cc"] == True]
elif f_stat == "ApprovedDelivered":
    display_df = display_df[display_df["is_ad"] == True]
elif f_stat == "ResolutionActioned":
    display_df = display_df[display_df["is_actioned"] == True]
elif f_stat == "NoResolutionActioned":
    display_df = display_df[display_df["is_no_actioned"] == True]
elif f_stat == "AllOpen":
    display_df = display_df[display_df["Current Status"].isin(OPEN_STATUSES)]
else:
    # default: show open
    display_df = display_df[display_df["Current Status"].isin(OPEN_STATUSES)]

# Column filters
if status_filter:
    display_df = display_df[display_df["Current Status"].isin(status_filter)]

if resolution_filter:
    display_df = display_df[display_df["resolutionType"].isin(resolution_filter)]

if actioned_filter == "Yes":
    display_df = display_df[display_df["is_actioned"] == True]
elif actioned_filter == "No":
    display_df = display_df[display_df["is_actioned"] == False]

if track_status_contains.strip():
    q = track_status_contains.strip().lower()
    display_df = display_df[display_df["Tracking Status"].astype(str).str.lower().str.contains(q, na=False)]

if order_contains.strip():
    q = order_contains.strip().lower()
    display_df = display_df[display_df["Order"].astype(str).str.lower().str.contains(q, na=False)]

if display_df.empty:
    st.info("No matching records found.")
    st.stop()

# Sort and number
display_df = display_df.sort_values(by="Requested date", ascending=False).reset_index(drop=True)
display_df["No"] = (display_df.index + 1).astype(str)

# ==========================================
# 17. Export / Copy controls (top right corner equivalent)
# ==========================================
export_cols = st.columns([1, 1, 6], vertical_alignment="center")
with export_cols[0]:
    export_df = display_df[
        [
            "RMA_ID_RAW",
            "Order",
            "Current Status",
            "Tracking Number",
            "Tracking Status",
            "Requested date",
            "Approved date",
            "Received date",
            "resolutionType",
            "Resolution actioned",
        ]
    ].copy()
    export_df.rename(columns={"RMA_ID_RAW": "RMA ID"}, inplace=True)
    xlsx_bytes = _export_xlsx_bytes(export_df)
    st.download_button(
        "⬇️ Export XLSX",
        data=xlsx_bytes,
        file_name="levis_returngo_dashboard.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
        key="btn_export_xlsx"
    )

with export_cols[1]:
    # Copy table (TSV) to clipboard via HTML button
    tsv = export_df.to_csv(sep="\t", index=False)
    # keep it safe for JS string
    tsv_js = json.dumps(tsv)
    components.html(
        f"""
        <div>
          <button id="copyBtn" style="
            width:100%;
            border:1px solid rgba(148,163,184,0.25);
            background:rgba(31,41,55,0.9);
            color:#e5e7eb;
            border-radius:10px;
            padding:10px 12px;
            font-size:13px;
            font-weight:800;
            cursor:pointer;
          ">📋 Copy table</button>
          <script>
            const data = {tsv_js};
            const btn = document.getElementById("copyBtn");
            btn.onclick = async () => {{
              try {{
                await navigator.clipboard.writeText(data);
                btn.innerText = "✅ Copied!";
                setTimeout(() => btn.innerText = "📋 Copy table", 1200);
              }} catch (e) {{
                btn.innerText = "❌ Copy failed";
                setTimeout(() => btn.innerText = "📋 Copy table", 1200);
              }}
            }};
          </script>
        </div>
        """,
        height=55,
    )

st.write("")

# ==========================================
# 18. Data table (RMA ID hyperlinked, includes Resolution actioned)
# ==========================================
edited = st.data_editor(
    display_df[
        [
            "No",
            "RMA ID",  # URL; shown as RMA ID via LinkColumn display_text
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
        "RMA ID": st.column_config.LinkColumn(
            "RMA ID",
            display_text=r"rmaid=(\d+)",
            width="small",
        ),
        "Order": st.column_config.TextColumn("Order", width="small"),
        "Current Status": st.column_config.TextColumn("Current Status", width="small"),
        "Tracking Number": st.column_config.LinkColumn("Tracking Number", display_text=r"ref=(.*)", width="medium"),
        "Tracking Status": st.column_config.TextColumn("Tracking Status", width="medium"),
        "Requested date": st.column_config.TextColumn("Requested date", width="small"),
        "Approved date": st.column_config.TextColumn("Approved date", width="small"),
        "Received date": st.column_config.TextColumn("Received date", width="small"),
        "resolutionType": st.column_config.TextColumn("resolutionType", width="small"),
        "Resolution actioned": st.column_config.TextColumn("Resolution actioned", width="medium"),
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
        "Resolution actioned",
    ],
)

# ONE-CLICK LOGIC (keeps scroll via localStorage + delayed restore)
editor_key = f"main_table_{st.session_state.table_key}"
if editor_key in st.session_state:
    edits = st.session_state[editor_key].get("edited_rows", {})
    for row_idx, changes in edits.items():
        idx = int(row_idx)
        row = display_df.iloc[idx]

        if "Update Tracking Number" in changes and changes["Update Tracking Number"]:
            st.session_state.modal_rma = row
            st.session_state.modal_action = "edit"
            st.session_state.table_key += 1
            st.rerun()

        if "View Timeline" in changes and changes["View Timeline"]:
            st.session_state.modal_rma = row
            st.session_state.modal_action = "view"
            st.session_state.table_key += 1
            st.rerun()
