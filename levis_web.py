import streamlit as st
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

# ==========================================
# 1. CONFIGURATION
# ==========================================
st.set_page_config(page_title="Levi's ReturnGO Ops", layout="wide", page_icon="📤")

# --- ACCESS SECRETS ---
try:
    MY_API_KEY = st.secrets["RETURNGO_API_KEY"]
except (FileNotFoundError, KeyError):
    MY_API_KEY = os.environ.get("RETURNGO_API_KEY")

if not MY_API_KEY:
    st.error("API Key not found! Please set 'RETURNGO_API_KEY' in Streamlit secrets or env vars.")
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
CACHE_EXPIRY_HOURS = 4
COURIER_REFRESH_HOURS = 12
MAX_WORKERS = 3            # keep concurrency low to avoid 429
RG_RPS = 2                 # soft pacing
SYNC_OVERLAP_MINUTES = 5

ACTIVE_STATUSES = ["Pending", "Approved", "Received"]  # "Open RMAs"

# ==========================================
# 1b. STYLING
# ==========================================
st.markdown(
    """
    <style>
      :root{
        --bg:#0b0f16;
        --panel:#111827;
        --panel2:#0f172a;
        --border:#273244;
        --text:#e5e7eb;
        --muted:#9ca3af;
        --accent:#c41230;
        --accent2:#ef4444;
        --ok:#22c55e;
      }

      .stApp { background: radial-gradient(1200px 600px at 20% 0%, #111827 0%, var(--bg) 55%); color: var(--text); }

      /* Top header */
      .hdr-title { font-size: 44px; font-weight: 800; letter-spacing: -0.02em; margin: 0 0 6px 0; }
      .hdr-sub { color: var(--muted); font-weight: 600; margin-top: 0; }

      /* Action buttons (right side) */
      div.stButton > button {
        width: 100%;
        border: 1px solid var(--border);
        background: linear-gradient(180deg, #111827 0%, #0b1220 100%);
        color: var(--text);
        border-radius: 12px;
        padding: 12px 14px;
        font-size: 15px;
        font-weight: 800;
      }
      div.stButton > button:hover {
        border-color: var(--accent);
        color: var(--accent);
      }

      /* Metric tile buttons */
      .tile-time {
        display:flex;
        justify-content:center;
        font-size: 12px;
        font-weight: 800;
        color: var(--muted);
        margin: 2px 0 6px 0;
      }

      /* Reduce the tiny sync buttons by ~50% */
      button[kind="secondary"]{
        padding: 6px 8px !important;
        font-size: 12px !important;
        border-radius: 10px !important;
      }

      /* Make the small sync button sit tight under the tile button */
      .tight-under { margin-top: -6px; }

      /* Filters row chips */
      .chip-row {
        display:flex; gap:10px; align-items:center; margin: 8px 0 0 0;
      }
      .chip-hint { color: var(--muted); font-weight: 700; font-size: 13px; margin-bottom: 6px; }

      /* Data editor tweaks */
      [data-testid="stDataEditor"]{
        border: 1px solid var(--border);
        border-radius: 14px;
        overflow: hidden;
        background: rgba(17,24,39,0.65);
      }

      /* Dialog styling */
      div[data-testid="stDialog"] {
        background: linear-gradient(180deg, #0b1220 0%, #0b0f16 100%);
        border: 1px solid var(--accent);
        border-radius: 14px;
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


def _extract_rate_headers(headers: dict) -> dict:
    """Try to surface any rate-limit headers if ReturnGO provides them."""
    keep = {}
    for k, v in headers.items():
        lk = k.lower()
        if "rate" in lk or "retry-after" in lk:
            keep[k] = v
    return keep


def rg_request(method: str, url: str, *, headers=None, timeout=15, json_body=None):
    """ReturnGO request wrapper with pacing + 429 backoff."""
    session = get_thread_session()
    headers = headers or rg_headers()

    backoff = 1
    last_rate_hdrs = None

    for attempt in range(1, 6):
        _sleep_for_rate_limit()
        res = session.request(method, url, headers=headers, timeout=timeout, json=json_body)
        last_rate_hdrs = _extract_rate_headers(res.headers)

        if res.status_code != 429:
            return res, last_rate_hdrs

        # 429
        ra = res.headers.get("Retry-After")
        if ra and ra.isdigit():
            sleep_s = int(ra)
        else:
            sleep_s = backoff
            backoff = min(backoff * 2, 30)

        if attempt == 1:
            st.warning(
                "ReturnGO rate limit reached (429). Slowing down and retrying. "
                "Consider fewer sync clicks (or ask ReturnGO to increase quota)."
            )
            if last_rate_hdrs:
                st.caption(f"Rate headers: {last_rate_hdrs}")

        time.sleep(sleep_s)

    return res, last_rate_hdrs


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
                courier_status TEXT,
                courier_last_checked TEXT
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

        # Migrations (if DB existed)
        cols_rmas = {row[1] for row in c.execute("PRAGMA table_info(rmas)").fetchall()}
        for col, coldef in [
            ("courier_status", "TEXT"),
            ("courier_last_checked", "TEXT"),
            ("last_fetched", "TEXT"),
        ]:
            if col not in cols_rmas:
                try:
                    c.execute(f"ALTER TABLE rmas ADD COLUMN {col} {coldef}")
                except Exception:
                    pass

        # Handle old sync_logs schemas if they exist
        cols_sync = {row[1] for row in c.execute("PRAGMA table_info(sync_logs)").fetchall()}
        if ("status" in cols_sync) or ("last_sync" in cols_sync):
            # rebuild to new schema
            c.execute("CREATE TABLE IF NOT EXISTS sync_logs_new (scope TEXT PRIMARY KEY, last_sync_iso TEXT)")
            try:
                c.execute(
                    "INSERT OR REPLACE INTO sync_logs_new (scope, last_sync_iso) SELECT status, last_sync FROM sync_logs"
                )
            except Exception:
                pass
            try:
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


def upsert_rma(rma_id: str, status: str, created_at: str, payload: dict, courier_status=None, courier_checked_iso=None):
    with DB_LOCK:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        now_iso = _iso_utc(_now_utc())

        if courier_status is None or courier_checked_iso is None:
            c.execute("SELECT courier_status, courier_last_checked FROM rmas WHERE rma_id=?", (str(rma_id),))
            row = c.fetchone()
            if row:
                if courier_status is None:
                    courier_status = row[0]
                if courier_checked_iso is None:
                    courier_checked_iso = row[1]

        c.execute(
            """
            INSERT OR REPLACE INTO rmas
            (rma_id, store_url, status, created_at, json_data, last_fetched, courier_status, courier_last_checked)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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
            "SELECT json_data, last_fetched, courier_status, courier_last_checked FROM rmas WHERE rma_id=?",
            (str(rma_id),),
        )
        row = c.fetchone()
        conn.close()

    if not row:
        return None

    payload = json.loads(row[0])
    payload["_local_courier_status"] = row[2]
    payload["_local_courier_checked"] = row[3]
    return payload, row[1]


def get_all_open_from_db():
    """Open = Pending/Approved/Received (what you want)."""
    with DB_LOCK:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        placeholders = ",".join(["?"] * len(ACTIVE_STATUSES))
        sql = f"""
            SELECT json_data, courier_status, courier_last_checked
            FROM rmas
            WHERE store_url=?
              AND status IN ({placeholders})
        """
        c.execute(sql, (STORE_URL, *ACTIVE_STATUSES))
        rows = c.fetchall()
        conn.close()

    results = []
    for js, cstat, cchk in rows:
        data = json.loads(js)
        data["_local_courier_status"] = cstat
        data["_local_courier_checked"] = cchk
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


def get_last_sync(scope: str):
    with DB_LOCK:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        try:
            c.execute("SELECT last_sync_iso FROM sync_logs WHERE scope=?", (scope,))
            row = c.fetchone()
        except Exception:
            row = None
        conn.close()

    if not row or not row[0]:
        return None
    try:
        return datetime.fromisoformat(row[0])
    except Exception:
        return None


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
                    top = data["history"][0]
                    return top.get("status") or top.get("description") or "Unknown"
                return data.get("status") or data.get("currentStatus") or "Unknown"
            except Exception:
                pass

            # fallback HTML parse (best-effort)
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

def fetch_rma_list(statuses, since_dt: datetime | None):
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

        res, _rate = rg_request("GET", url, timeout=20)
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
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=timezone.utc)
    except Exception:
        return True

    return (_now_utc() - last_dt) > timedelta(hours=CACHE_EXPIRY_HOURS)


def maybe_refresh_courier(rma_payload: dict) -> tuple[str | None, str | None]:
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
    res, _rate = rg_request("GET", url, timeout=20)
    if res.status_code != 200:
        return cached[0] if cached else None

    data = res.json() if res.content else {}
    summary = data.get("rmaSummary", {}) or {}

    if cached:
        data["_local_courier_status"] = cached[0].get("_local_courier_status")
        data["_local_courier_checked"] = cached[0].get("_local_courier_checked")

    courier_status, courier_checked = maybe_refresh_courier(data)

    upsert_rma(
        rma_id=str(rma_id),
        status=summary.get("status", "Unknown"),
        created_at=summary.get("createdAt") or data.get("createdAt") or "",
        payload=data,
        courier_status=courier_status,
        courier_checked_iso=courier_checked,
    )

    data["_local_courier_status"] = courier_status
    data["_local_courier_checked"] = courier_checked
    return data


def perform_sync(statuses=None):
    status_msg = st.empty()
    status_msg.info("⏳ Connecting to ReturnGO...")

    if statuses is None:
        statuses = ACTIVE_STATUSES

    scope = ",".join(statuses)
    since_dt = get_last_sync(scope)

    list_bar = st.progress(0, text="Fetching RMA list from ReturnGO...")
    summaries = fetch_rma_list(statuses, since_dt)
    list_bar.progress(1.0, text=f"Fetched {len(summaries)} RMAs")
    time.sleep(0.15)
    list_bar.empty()

    api_ids = {s.get("rmaId") for s in summaries if s.get("rmaId")}

    # tidy locally cached open RMAs if we synced all open
    if set(statuses) == set(ACTIVE_STATUSES):
        local_open_ids = set()
        for stt in ACTIVE_STATUSES:
            local_open_ids |= get_local_ids_for_status(stt)
        stale = local_open_ids - api_ids
        delete_rmas(stale)

    # fetch details only if stale
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
    set_last_sync(scope, now)

    # Set "last full sync" only when syncing all open statuses
    if set(statuses) == set(ACTIVE_STATUSES):
        st.session_state["last_full_sync"] = now.strftime("%Y-%m-%d %H:%M")

    st.session_state["show_toast"] = True
    status_msg.success("✅ Sync Complete!")
    st.rerun()


# ==========================================
# 6. MUTATIONS (Tracking + Notes)
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

    res, _rate = rg_request("PUT", f"https://api.returngo.ai/shipment/{shipment_id}", headers=headers, timeout=15, json_body=payload)
    if res.status_code == 200:
        fetch_rma_detail(rma_id)  # refresh this one
        return True, "Success"
    return False, f"API Error {res.status_code}: {res.text}"


def push_comment_update(rma_id, comment_text):
    headers = {**rg_headers(), "Content-Type": "application/json"}
    payload = {"text": comment_text, "isPublic": False}

    res, _rate = rg_request("POST", f"https://api.returngo.ai/rma/{rma_id}/comment", headers=headers, timeout=15, json_body=payload)
    if res.status_code in (200, 201):
        fetch_rma_detail(rma_id)
        return True, "Success"
    return False, f"API Error {res.status_code}: {res.text}"


# ==========================================
# 7. EXTRACTION HELPERS (dates + resolution)
# ==========================================

def _try_parse_dt(val) -> datetime | None:
    if not val:
        return None
    try:
        s = str(val)
        s = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _find_event_date(payload: dict, want: list[str]) -> datetime | None:
    """
    Best-effort: scan common places for event/status timelines.
    want: list of keywords (any match)
    """
    want_l = [w.lower() for w in want]

    def match_name(x: str) -> bool:
        xl = (x or "").lower()
        return any(w in xl for w in want_l)

    # Common containers
    candidates = []
    summary = payload.get("rmaSummary") or {}
    for key in ["events", "statusHistory", "timeline", "history"]:
        v = summary.get(key)
        if isinstance(v, list):
            candidates.extend(v)

    for key in ["events", "statusHistory", "timeline", "history"]:
        v = payload.get(key)
        if isinstance(v, list):
            candidates.extend(v)

    # Sometimes shipment-level events exist
    shipments = payload.get("shipments") or []
    for sh in shipments:
        for key in ["events", "statusHistory", "history", "timeline"]:
            v = sh.get(key)
            if isinstance(v, list):
                candidates.extend(v)

    # comments might contain datetime + text
    comments = payload.get("comments") or []
    for c in comments:
        candidates.append(c)

    found = []
    for e in candidates:
        if not isinstance(e, dict):
            continue
        name = (
            e.get("eventName")
            or e.get("name")
            or e.get("type")
            or e.get("status")
            or e.get("event")
            or ""
        )
        if not match_name(str(name)):
            continue

        for dkey in ["eventDate", "datetime", "createdAt", "date", "timestamp", "time"]:
            dt = _try_parse_dt(e.get(dkey))
            if dt:
                found.append(dt)
                break

    if not found:
        return None
    # earliest is usually the moment it happened
    return sorted(found)[0]


def _resolution_types(payload: dict) -> str:
    """
    Best-effort: find resolutionType per item. If multiple distinct -> Mix.
    """
    types = set()

    # Try common item arrays
    for key in ["items", "rmaItems", "returnedItems", "products", "lineItems"]:
        arr = payload.get(key)
        if isinstance(arr, list):
            for it in arr:
                if not isinstance(it, dict):
                    continue
                for rk in ["resolutionType", "resolution", "resolution_type", "refundType", "refundMethod"]:
                    v = it.get(rk)
                    if v:
                        types.add(str(v).strip())
                # Sometimes nested
                reso = it.get("resolution") if isinstance(it.get("resolution"), dict) else None
                if reso:
                    for rk in ["type", "resolutionType", "method", "name"]:
                        v = reso.get(rk)
                        if v:
                            types.add(str(v).strip())

    # also check summary-level
    summ = payload.get("rmaSummary") or {}
    for rk in ["resolutionType", "resolution", "resolution_type"]:
        v = summ.get(rk)
        if v:
            types.add(str(v).strip())

    if not types:
        return "Unknown"
    if len(types) == 1:
        return next(iter(types))
    return "Mix"


def _resolution_actioned(payload: dict) -> str:
    """
    Best-effort indicator:
    - Looks for transaction/refund completed, or exchange/replacement created.
    """
    text = json.dumps(payload).lower()

    # Refund indicators
    if any(x in text for x in ["refundstatus", "refunded", "refund paid", "refund completed", "refund_complete", "refund succeeded"]):
        return "Yes"
    if any(x in text for x in ["paymentstatus", "paid", "settled"]) and "refund" in text:
        return "Yes"

    # Exchange indicators
    if any(x in text for x in ["exchangeorder", "replacementorder", "new order", "exchange created", "replacement created"]):
        return "Yes"

    # If resolution exists but no clear indicator
    if "resolution" in text or "refund" in text or "exchange" in text:
        return "Unknown"

    return "No"


# ==========================================
# 8. UI STATE
# ==========================================

if "filter_mode" not in st.session_state:
    st.session_state.filter_mode = "OPEN"   # default view
if "filter_status" not in st.session_state:
    st.session_state.filter_status = "OPEN" # OPEN / Pending / Approved / Received
if "search_query_input" not in st.session_state:
    st.session_state.search_query_input = ""
if "show_toast" not in st.session_state:
    st.session_state.show_toast = False

# Modal state
if "modal_action" not in st.session_state:
    st.session_state.modal_action = None
if "modal_record" not in st.session_state:
    st.session_state.modal_record = None

# ==========================================
# 9. DIALOGS (NO FORCED RERUN ON OPEN)
# ==========================================

@st.dialog("Update Tracking")
def show_update_tracking_dialog(record):
    st.markdown(f"### Update Tracking for `{record['RMA ID']}`")
    with st.form("upd_track"):
        new_track = st.text_input("New Tracking Number", value=record.get("DisplayTrack",""))
        submitted = st.form_submit_button("Save Changes")
        if submitted:
            if not record.get("shipment_id"):
                st.error("No Shipment ID.")
            else:
                ok, msg = push_tracking_update(record["RMA ID"], record["shipment_id"], new_track)
                if ok:
                    st.success("Updated!")
                    time.sleep(0.4)
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
                    time.sleep(0.2)
                    st.rerun()
                else:
                    st.error(msg)

    full = record.get("full_data") or {}
    timeline = full.get("comments", []) or []
    if not timeline:
        st.info("No timeline events found.")
    else:
        for t in timeline:
            d_str = (t.get("datetime", "") or "")[:16].replace("T", " ")
            st.markdown(f"**{d_str}** | `{t.get('triggeredBy', 'System')}`\n> {t.get('htmlText', '')}")
            st.divider()

# ==========================================
# 10. HEADER
# ==========================================

col1, col2 = st.columns([3.2, 1.1])

with col1:
    st.markdown('<div class="hdr-title">Levi\'s ReturnGO Ops Dashboard</div>', unsafe_allow_html=True)
    last_full = st.session_state.get("last_full_sync", "N/A")
    st.markdown(
        f'<div class="hdr-sub">CONNECTED TO: {STORE_URL.upper()} | LAST FULL SYNC: <span style="color:var(--ok)">{last_full}</span></div>',
        unsafe_allow_html=True
    )

with col2:
    if st.button("🔄 Sync Open RMAs", type="primary"):
        perform_sync(ACTIVE_STATUSES)

    if st.button("🗑️ Reset Cache", type="secondary"):
        if clear_db():
            st.success("Cache cleared!")
            st.rerun()

# Toast
if st.session_state.get("show_toast"):
    st.toast("✅ Sync Complete!", icon="🔄")
    st.session_state["show_toast"] = False

# ==========================================
# 11. LOAD + PROCESS DATA
# ==========================================

raw_data = get_all_open_from_db()
processed_rows = []

counts = {"Pending": 0, "Approved": 0, "Received": 0}
for rma in raw_data:
    summary = rma.get("rmaSummary", {}) or {}
    shipments = rma.get("shipments", []) or []
    comments = rma.get("comments", []) or []

    status = summary.get("status", "Unknown")
    if status in counts:
        counts[status] += 1

    rma_id = summary.get("rmaId", "N/A")
    order_name = summary.get("order_name", summary.get("orderName", "N/A"))

    # Tracking
    track_nums = [s.get("trackingNumber") for s in shipments if s.get("trackingNumber")]
    track_str = ", ".join(track_nums) if track_nums else ""
    shipment_id = shipments[0].get("shipmentId") if shipments else None

    local_status = rma.get("_local_courier_status", "") or ""
    track_link_url = f"https://portal.thecourierguy.co.za/track?ref={track_nums[0]}" if track_nums else ""

    # Dates (best-effort)
    req_dt = _find_event_date(rma, ["rma_created", "created", "rma created"]) or _try_parse_dt(summary.get("createdAt"))
    appr_dt = _find_event_date(rma, ["rma_approved", "approved"]) or _try_parse_dt(summary.get("approvedAt"))
    recv_dt = _find_event_date(rma, ["shipment_received", "received", "shipment received", "rma_received"])

    # resolutionType + actioned
    res_type = _resolution_types(rma)
    res_actioned = _resolution_actioned(rma)

    # Flags
    is_nt = (status == "Approved" and not track_str)
    is_fg = any("flagged" in (c.get("htmlText", "").lower()) for c in comments)

    processed_rows.append(
        {
            "No": "",
            "RMA ID": rma_id,
            "RMA URL": f"https://app.returngo.ai/dashboard/returns?filter_status=open&rmaid={rma_id}",
            "Order": order_name,
            "Current Status": status,
            "Tracking Number": track_link_url,
            "Tracking Status": local_status,
            "Requested date": req_dt.date().isoformat() if req_dt else "N/A",
            "Approved date": appr_dt.date().isoformat() if appr_dt else "N/A",
            "Received date": recv_dt.date().isoformat() if recv_dt else "N/A",
            "resolutionType": res_type,
            "Resolution actioned": res_actioned,
            "Days since updated": "",  # optional placeholder
            "Update Tracking": False,
            "View Timeline": False,
            "DisplayTrack": track_str,
            "shipment_id": shipment_id,
            "full_data": rma,
            "is_nt": is_nt,
            "is_fg": is_fg,
        }
    )

df_view = pd.DataFrame(processed_rows)

# ==========================================
# 12. METRIC TILES (with sync buttons under)
# ==========================================

def _tile_time(scope_key: str) -> str:
    dt = get_last_sync(scope_key)
    if not dt:
        return "UPDATED: -"
    return f"UPDATED: {dt.astimezone().strftime('%H:%M')}"

b1, b2, b3, b4 = st.columns([1, 1, 1, 2.1])

def set_status_filter(v):
    st.session_state.filter_status = v
    st.session_state.filter_mode = "OPEN"
    st.rerun()

with b1:
    st.markdown(f'<div class="tile-time">{_tile_time("Pending")}</div>', unsafe_allow_html=True)
    if st.button(f"PENDING  {counts['Pending']}", key="tile_pending"):
        set_status_filter("Pending")
    st.markdown('<div class="tight-under"></div>', unsafe_allow_html=True)
    if st.button("🔄 Sync Pending", key="sync_pending", type="secondary"):
        perform_sync(["Pending"])

with b2:
    st.markdown(f'<div class="tile-time">{_tile_time("Approved")}</div>', unsafe_allow_html=True)
    if st.button(f"APPROVED  {counts['Approved']}", key="tile_approved"):
        set_status_filter("Approved")
    st.markdown('<div class="tight-under"></div>', unsafe_allow_html=True)
    if st.button("🔄 Sync Approved", key="sync_approved", type="secondary"):
        perform_sync(["Approved"])

with b3:
    st.markdown(f'<div class="tile-time">{_tile_time("Received")}</div>', unsafe_allow_html=True)
    if st.button(f"RECEIVED  {counts['Received']}", key="tile_received"):
        set_status_filter("Received")
    st.markdown('<div class="tight-under"></div>', unsafe_allow_html=True)
    if st.button("🔄 Sync Received", key="sync_received", type="secondary"):
        perform_sync(["Received"])

with b4:
    st.markdown("&nbsp;", unsafe_allow_html=True)
    if st.button("📋 List all Open RMAs", key="list_open"):
        set_status_filter("OPEN")

st.divider()

# ==========================================
# 13. PRE-FILTERS + SEARCH + EXPORT
# ==========================================

if "prefilter" not in st.session_state:
    st.session_state.prefilter = "None"

pf1, pf2, pf3, pf4 = st.columns([1.2, 1.5, 2.8, 1.5])

with pf1:
    if st.button("Courier Cancelled", key="pf_cancelled"):
        st.session_state.prefilter = "Courier Cancelled"
        st.rerun()

with pf2:
    if st.button("Approved > Delivered", key="pf_appr_deliv"):
        st.session_state.prefilter = "Approved > Delivered"
        st.rerun()

with pf3:
    search_query = st.text_input(
        "Search",
        placeholder="🔍 Search Order, RMA, Tracking, Status...",
        label_visibility="collapsed",
        key="search_query_input",
    )

with pf4:
    # Export button (Excel)
    def _to_excel_bytes(df: pd.DataFrame) -> bytes:
        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Open RMAs")
        return output.getvalue()

    st.download_button(
        "⬇️ Export Excel",
        data=_to_excel_bytes(df_view.drop(columns=["full_data", "DisplayTrack", "shipment_id"], errors="ignore")),
        file_name=f"levis_open_rmas_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

# Column filter controls (Streamlit doesn't support true header filters in st.data_editor)
with st.expander("Filters", expanded=False):
    c1, c2, c3, c4 = st.columns(4)

    statuses_sel = []
    if not df_view.empty:
        statuses_sel = sorted(df_view["Current Status"].dropna().unique().tolist())

    with c1:
        f_status = st.multiselect("Current Status", options=statuses_sel, default=[])
    with c2:
        f_track_contains = st.text_input("Tracking Status contains", value="")
    with c3:
        f_res_type = st.multiselect("resolutionType", options=sorted(df_view["resolutionType"].dropna().unique().tolist()) if not df_view.empty else [], default=[])
    with c4:
        f_actioned = st.multiselect("Resolution actioned", options=sorted(df_view["Resolution actioned"].dropna().unique().tolist()) if not df_view.empty else [], default=[])

# ==========================================
# 14. APPLY FILTERS
# ==========================================

display_df = df_view.copy()

# Base view: OPEN means all three statuses
if st.session_state.filter_status in ("Pending", "Approved", "Received"):
    display_df = display_df[display_df["Current Status"] == st.session_state.filter_status]
else:
    display_df = display_df[display_df["Current Status"].isin(ACTIVE_STATUSES)]

# Prefilters
pf = st.session_state.get("prefilter", "None")
if pf == "Courier Cancelled":
    display_df = display_df[display_df["Tracking Status"].astype(str).str.contains("Courier Cancelled", case=False, na=False)]
elif pf == "Approved > Delivered":
    display_df = display_df[
        (display_df["Current Status"] == "Approved")
        & (display_df["Tracking Status"].astype(str).str.contains("Delivered", case=False, na=False))
    ]

# Search filter
sq = (search_query or "").strip().lower()
if sq:
    display_df = display_df[
        display_df.apply(
            lambda r: sq in str(r.get("RMA ID","")).lower()
                      or sq in str(r.get("Order","")).lower()
                      or sq in str(r.get("Tracking Number","")).lower()
                      or sq in str(r.get("Tracking Status","")).lower()
                      or sq in str(r.get("resolutionType","")).lower(),
            axis=1
        )
    ]

# Column filters
if f_status:
    display_df = display_df[display_df["Current Status"].isin(f_status)]
if f_track_contains:
    display_df = display_df[display_df["Tracking Status"].astype(str).str.contains(f_track_contains, case=False, na=False)]
if f_res_type:
    display_df = display_df[display_df["resolutionType"].isin(f_res_type)]
if f_actioned:
    display_df = display_df[display_df["Resolution actioned"].isin(f_actioned)]

# Sort
if not display_df.empty:
    display_df = display_df.sort_values(by="Requested date", ascending=False).reset_index(drop=True)
    display_df["No"] = (display_df.index + 1).astype(str)

# ==========================================
# 15. TABLE (RMA ID is hyperlink + actions)
# ==========================================

if display_df.empty:
    st.info("No matching records found.")
    st.stop()

edited = st.data_editor(
    display_df[
        [
            "No",
            "RMA URL",
            "Order",
            "Current Status",
            "Tracking Number",
            "Tracking Status",
            "Requested date",
            "Approved date",
            "Received date",
            "resolutionType",
            "Resolution actioned",
            "Update Tracking",
            "View Timeline",
        ]
    ],
    width="stretch",
    height=700,
    hide_index=True,
    key="main_table",
    column_config={
        "No": st.column_config.TextColumn("No", width="small"),
        "RMA URL": st.column_config.LinkColumn(
            "RMA ID",
            display_text=r"rmaid=(\d+)",
            width="small"
        ),
        "Order": st.column_config.TextColumn("Order", width="medium"),
        "Current Status": st.column_config.TextColumn("Current Status", width="small"),
        "Tracking Number": st.column_config.LinkColumn("Tracking Number", display_text=r"ref=(.*)", width="medium"),
        "Tracking Status": st.column_config.TextColumn("Tracking Status", width="medium"),
        "Requested date": st.column_config.TextColumn("Requested date", width="small"),
        "Approved date": st.column_config.TextColumn("Approved date", width="small"),
        "Received date": st.column_config.TextColumn("Received date", width="small"),
        "resolutionType": st.column_config.TextColumn("resolutionType", width="medium"),
        "Resolution actioned": st.column_config.TextColumn("Resolution actioned", width="small"),
        "Update Tracking": st.column_config.CheckboxColumn("Update Tracking", width="small"),
        "View Timeline": st.column_config.CheckboxColumn("View Timeline", width="small"),
    },
    disabled=[
        "No",
        "RMA URL",
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

# IMPORTANT:
# We do NOT st.rerun() to open dialogs (prevents jumping back to the top / row 1).
# The checkbox may stay ticked until the next rerun; that's the trade-off to preserve position.
state = st.session_state.get("main_table", {})
edits = state.get("edited_rows", {}) if isinstance(state, dict) else {}

for row_idx, changes in edits.items():
    idx = int(row_idx)
    record = display_df.iloc[idx].to_dict()

    if changes.get("Update Tracking"):
        # open dialog without forcing a rerun
        record["RMA ID"] = record.get("RMA URL","").split("rmaid=")[-1]
        # rehydrate full_data etc from original df_view row
        full_row = df_view[df_view["RMA ID"] == record["RMA ID"]]
        if not full_row.empty:
            record["full_data"] = full
