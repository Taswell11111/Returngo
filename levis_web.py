import streamlit as st
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

# ==========================================
# 1. CONFIGURATION
# ==========================================
st.set_page_config(page_title="Levi's RMA Ops", layout="wide", page_icon="📤")

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
CACHE_EXPIRY_HOURS = 4                    # don't refetch ReturnGO detail if cached within this window
COURIER_REFRESH_HOURS = 12                # don't re-check courier tracking too often
MAX_WORKERS = 3                           # ReturnGO docs recommend minimizing concurrency
RG_RPS = 2                                # soft rate-limit (requests per second) to avoid 429
SYNC_OVERLAP_MINUTES = 5                  # pull a small overlap on incremental sync

ACTIVE_STATUSES = ["Pending", "Approved", "Received"]

# ==========================================
# 1b. STYLING
# ==========================================
st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117; color: white; }

    div.stButton > button {
        width: 100%;
        border: 1px solid #4b5563;
        background-color: #1f2937;
        color: white;
        border-radius: 8px;
        padding: 12px 24px;
        font-size: 16px;
        font-weight: bold;
    }
    div.stButton > button:hover {
        border-color: #c41230;
        color: #c41230;
    }
    div[data-testid="stDialog"] {
        background-color: #1a1a1a;
        border: 1px solid #c41230;
    }
    .sync-time {
        font-size: 0.8em;
        color: #888;
        text-align: center;
        margin-top: -10px;
        margin-bottom: 10px;
    }
    div[data-testid="column"] button[kind="secondary"] {
        padding: 12px 10px;
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
    # header keys are case-insensitive; using lowercase aligns with API docs
    return {"x-api-key": MY_API_KEY, "x-shop-name": STORE_URL}


def rg_request(method: str, url: str, *, headers=None, timeout=15, json_body=None):
    """ReturnGO request wrapper with pacing and 429-aware backoff."""
    session = get_thread_session()
    headers = headers or rg_headers()

    # Small loop to handle 429s even when Retry doesn't get a Retry-After header
    backoff = 1
    for attempt in range(1, 6):
        _sleep_for_rate_limit()
        res = session.request(method, url, headers=headers, timeout=timeout, json=json_body)

        if res.status_code != 429:
            return res

        # 429: try server guidance, otherwise exponential backoff
        ra = res.headers.get("Retry-After")
        if ra and ra.isdigit():
            sleep_s = int(ra)
        else:
            sleep_s = backoff
            backoff = min(backoff * 2, 30)

        # Only surface a small, non-spammy notice
        if attempt == 1:
            st.warning(
                "ReturnGO rate limit reached (429). Slowing down and retrying. "
                "If this happens often, reduce sync frequency or request a bulk quota key."
            )
        time.sleep(sleep_s)

    return res


# ==========================================
# 3. DATABASE
# ==========================================

def init_db():
    with DB_LOCK:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()

        # --- RMAs table (as you already have it) ---
        c.execute("""
            CREATE TABLE IF NOT EXISTS rmas (
                rma_id TEXT PRIMARY KEY,
                store_url TEXT,
                status TEXT,
                created_at TEXT,
                json_data TEXT,
                last_fetched TIMESTAMP,
                courier_status TEXT,
                courier_last_checked TIMESTAMP
            )
        """)

        # --- sync_logs: create if missing ---
        c.execute("""
            CREATE TABLE IF NOT EXISTS sync_logs (
                scope TEXT PRIMARY KEY,
                last_sync_iso TEXT
            )
        """)

        # --- MIGRATION: if old sync_logs schema exists, adapt it ---
        cols = {row[1] for row in c.execute("PRAGMA table_info(sync_logs)").fetchall()}

        # Old column names: status, last_sync
        has_old = ("status" in cols) or ("last_sync" in cols)
        has_new = ("scope" in cols) and ("last_sync_iso" in cols)

        if not has_new and has_old:
            # Create a new table with the new schema
            c.execute("""
                CREATE TABLE IF NOT EXISTS sync_logs_new (
                    scope TEXT PRIMARY KEY,
                    last_sync_iso TEXT
                )
            """)
            # Copy what we can from old -> new
            # If old had (status,last_sync), map status->scope and store as text
            try:
                c.execute("""
                    INSERT OR REPLACE INTO sync_logs_new (scope, last_sync_iso)
                    SELECT status, last_sync FROM sync_logs
                """)
            except Exception:
                pass

            # Swap tables
            c.execute("DROP TABLE sync_logs")
            c.execute("ALTER TABLE sync_logs_new RENAME TO sync_logs")

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

        # Preserve courier fields unless overridden
        if courier_status is None or courier_checked_iso is None:
            c.execute(
                "SELECT courier_status, courier_last_checked FROM rmas WHERE rma_id=?",
                (str(rma_id),),
            )
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
        c.executemany("DELETE FROM rmas WHERE rma_id=? AND store_url=?", [(str(i), STORE_URL) for i in rma_ids])
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


def get_all_active_from_db():
    with DB_LOCK:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        placeholders = ",".join("?" for _ in ACTIVE_STATUSES)
        c.execute(
            f"SELECT json_data, courier_status, courier_last_checked FROM rmas WHERE store_url=? AND status IN ({placeholders})",
            (STORE_URL, *ACTIVE_STATUSES),
        )
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
            # New schema
            c.execute("SELECT last_sync_iso FROM sync_logs WHERE scope=?", (scope,))
            row = c.fetchone()
            if row and row[0]:
                try:
                    return datetime.fromisoformat(row[0])
                except Exception:
                    return None
            return None
        except sqlite3.OperationalError:
            # Fallback to old schema if still present for any reason
            try:
                c.execute("SELECT last_sync FROM sync_logs WHERE status=?", (scope,))
                row = c.fetchone()
                if row and row[0]:
                    # old stored as string like "YYYY-mm-dd HH:MM" most likely
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
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Authorization": f"Bearer {PARCEL_NINJA_TOKEN}",
        }
        res = requests.get(url, headers=headers, timeout=10)
        final_status = "Unknown"

        if res.status_code == 200:
            # Try JSON first
            try:
                data = res.json()
                if isinstance(data, dict) and "history" in data and data["history"]:
                    final_status = data["history"][0].get("status") or data["history"][0].get("description")
                else:
                    final_status = data.get("status") or data.get("currentStatus") or "Unknown"
                return final_status or "Unknown"
            except Exception:
                pass

            # Fallback HTML parse
            content = res.text
            clean_html = re.sub(r"<(script|style).*?</\1>", "", content, flags=re.DOTALL | re.IGNORECASE)
            history_section = re.search(
                r"<table[^>]*?tracking-history.*?>(.*?)</table>",
                clean_html,
                re.DOTALL | re.IGNORECASE,
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

        elif res.status_code == 404:
            return "Tracking Not Found"

        return f"Error {res.status_code}"

    except Exception:
        return "Check Failed"


# ==========================================
# 5. RETURNGO FETCHING (incremental + cached)
# ==========================================

def fetch_rma_list(statuses, since_dt: datetime | None):
    """Fetch RMA summaries for one or many statuses using cursor pagination."""

    all_summaries = []
    cursor = None

    status_param = ",".join(statuses)

    # Optional incremental filter (supported by docs)
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


def maybe_refresh_courier(rma_payload: dict) -> tuple[str | None, str | None]:
    shipments = rma_payload.get("shipments", []) or []
    track_no = None
    for s in shipments:
        if s.get("trackingNumber"):
            track_no = s.get("trackingNumber")
            break

    if not track_no:
        return None, None

    # Check last courier check
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

    # Refresh courier
    status = check_courier_status(track_no)
    checked_iso = _iso_utc(_now_utc())
    return status, checked_iso


def fetch_rma_detail(rma_id: str):
    # If fresh enough, use cache
    cached = get_rma(rma_id)
    if cached and not should_refresh_detail(rma_id):
        return cached[0]

    url = f"https://api.returngo.ai/rma/{rma_id}"
    res = rg_request("GET", url, timeout=20)
    if res.status_code != 200:
        return cached[0] if cached else None

    data = res.json() if res.content else {}
    summary = data.get("rmaSummary", {}) or {}

    # Courier refresh is cached separately
    # For the refresh decision, pass cached courier fields if we have them
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


def perform_sync(statuses=None, *, full=False):
    """Efficient sync:

    - Uses /rmas cursor pagination with pagesize=500.
    - Uses incremental sync via rma_updated_at filter (unless full=True).
    - Only fetches /rma/{id} details when cache is stale.
    - Minimizes concurrency and applies a soft RPS limit.
    """

    status_msg = st.empty()
    status_msg.info("⏳ Connecting to ReturnGO...")

    if statuses is None:
        statuses = ACTIVE_STATUSES

    scope = ",".join(statuses)
    since_dt = None if full else get_last_sync(scope)

    # 1) Fetch list (one call stream, cursor paginated)
    list_bar = st.progress(0, text="Fetching RMA list from ReturnGO...")
    summaries = fetch_rma_list(statuses, since_dt)
    list_bar.progress(1.0, text=f"Fetched {len(summaries)} RMAs")
    time.sleep(0.2)
    list_bar.empty()

    api_ids = {s.get("rmaId") for s in summaries if s.get("rmaId")}

    # 2) Keep local cache tidy for active statuses
    # If we're syncing active statuses, delete any locally-active RMAs not returned in the current active list.
    if set(statuses) == set(ACTIVE_STATUSES) and not full:
        local_active_ids = set()
        for stt in ACTIVE_STATUSES:
            local_active_ids |= get_local_ids_for_status(stt)
        stale = local_active_ids - api_ids
        delete_rmas(stale)

    # 3) Fetch details (only when needed)
    to_fetch = [rid for rid in api_ids if should_refresh_detail(rid)]
    total = len(to_fetch)

    status_msg.info(f"⏳ Syncing {total} records...")

    if total > 0:
        bar = st.progress(0, text="Downloading Details...")

        # Reduce concurrency to avoid ReturnGO 429s
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = [ex.submit(fetch_rma_detail, rid) for rid in to_fetch]
            done = 0
            for _ in concurrent.futures.as_completed(futures):
                done += 1
                bar.progress(done / total, text=f"Syncing: {done}/{total}")

        bar.empty()

    # Save sync marker
    now = _now_utc()
    set_last_sync(scope, now)
    st.session_state["last_sync"] = now.strftime("%Y-%m-%d %H:%M")
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

    try:
        res = rg_request("PUT", f"https://api.returngo.ai/shipment/{shipment_id}", headers=headers, timeout=15, json_body=payload)
        if res.status_code == 200:
            # Refresh this RMA only
            fresh = fetch_rma_detail(rma_id)
            return True, "Success" if fresh else "Updated"
        return False, f"API Error {res.status_code}: {res.text}"
    except Exception as e:
        return False, str(e)


def push_comment_update(rma_id, comment_text):
    headers = {**rg_headers(), "Content-Type": "application/json"}

    # Docs show /comment, your old code used /note. Use /comment to match API docs.
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
# 7. FRONTEND UI
# ==========================================

# Initialize modal states
if "modal_rma" not in st.session_state:
    st.session_state.modal_rma = None
if "modal_action" not in st.session_state:
    st.session_state.modal_action = None
if "table_key" not in st.session_state:
    st.session_state.table_key = 0


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
                    time.sleep(0.5)
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


if "filter_status" not in st.session_state:
    st.session_state.filter_status = "All"
if st.session_state.get("show_toast"):
    st.toast("✅ API Sync Complete!", icon="🔄")
    st.session_state["show_toast"] = False


def set_filter(f):
    st.session_state.filter_status = f
    st.rerun()


# --- Header ---
col1, col2 = st.columns([3, 1])
with col1:
    st.title("Levi's ReturnGO Ops Dashboard")
    st.markdown(
        f"**CONNECTED TO:** {STORE_URL.upper()} | **LAST SYNC:** :green[{st.session_state.get('last_sync', 'N/A')}]"
    )

with col2:
    if st.button("🔄 Sync All Data", type="primary"):
        perform_sync()
    if st.button("🧾 Full Rebuild (Slow)", type="secondary", help="Ignores incremental sync and rebuilds cache"):
        perform_sync(full=True)
    if st.button("🗑️ Reset Cache", type="secondary"):
        if clear_db():
            st.success("Cache cleared!")
            st.rerun()


# --- Data Processing ---
search_query = st.session_state.get("search_query_input", "")
raw_data = get_all_active_from_db()
processed_rows = []
counts = {"Pending": 0, "Approved": 0, "Received": 0, "NoTrack": 0, "Flagged": 0}

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

    local_status = rma.get("_local_courier_status", "")
    track_link_url = f"https://portal.thecourierguy.co.za/track?ref={track_nums[0]}" if track_nums else ""

    created_at = summary.get("createdAt")
    if not created_at:
        for evt in summary.get("events", []) or []:
            if evt.get("eventName") == "RMA_CREATED":
                created_at = evt.get("eventDate")
                break

    u_at = rma.get("lastUpdated")
    d_since = 0
    if u_at:
        try:
            d_since = (
                _now_utc().date()
                - datetime.fromisoformat(str(u_at).replace("Z", "+00:00")).date()
            ).days
        except Exception:
            pass

    if status in counts:
        counts[status] += 1

    is_nt = (status == "Approved" and not track_str)
    is_fg = any("flagged" in (c.get("htmlText", "").lower()) for c in comments)

    if is_nt:
        counts["NoTrack"] += 1
    if is_fg:
        counts["Flagged"] += 1

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
                "Status": status,
                "Tracking Number": track_link_url,
                "Tracking Status": local_status,
                "Created": str(created_at)[:10] if created_at else "N/A",
                "Updated": str(u_at)[:10] if u_at else "N/A",
                "Days since updated": str(d_since),
                "Update Tracking Number": False,
                "View Timeline": False,
                "DisplayTrack": track_str,
                "shipment_id": shipment_id,
                "full_data": rma,
                "is_nt": is_nt,
                "is_fg": is_fg,
            }
        )


# --- Metrics with Split Buttons (Filter vs Sync) ---
b1, b2, b3, b4, b5 = st.columns(5)


def get_status_time(s):
    try:
        ts = get_last_sync(s)
        if not ts:
            return "<div class='sync-time'>UPDATED: -</div>"
        return f"<div class='sync-time'>UPDATED: {ts.strftime('%H:%M')}</div>"
    except Exception:
        return "<div class='sync-time'>UPDATED: -</div>"

with b1:
    bc1, bc2 = st.columns([3, 1])
    with bc1:
        if st.button(f"PENDING\n{counts['Pending']}", key="btn_p"):
            set_filter("Pending")
    with bc2:
        if st.button("🔄", key="sync_p", help="Sync Pending Only"):
            perform_sync(["Pending"])
    st.markdown(get_status_time("Pending"), unsafe_allow_html=True)

with b2:
    bc1, bc2 = st.columns([3, 1])
    with bc1:
        if st.button(f"APPROVED\n{counts['Approved']}", key="btn_a"):
            set_filter("Approved")
    with bc2:
        if st.button("🔄", key="sync_a", help="Sync Approved Only"):
            perform_sync(["Approved"])
    st.markdown(get_status_time("Approved"), unsafe_allow_html=True)

with b3:
    bc1, bc2 = st.columns([3, 1])
    with bc1:
        if st.button(f"RECEIVED\n{counts['Received']}", key="btn_r"):
            set_filter("Received")
    with bc2:
        if st.button("🔄", key="sync_r", help="Sync Received Only"):
            perform_sync(["Received"])
    st.markdown(get_status_time("Received"), unsafe_allow_html=True)

with b4:
    if st.button(f"NO TRACKING\n{counts['NoTrack']} "):
        set_filter("NoTrack")

with b5:
    if st.button(f"🚩 FLAGGED\n{counts['Flagged']} "):
        set_filter("Flagged")


# --- Search Bar & Clear Button ---
st.divider()


def clear_search_cb():
    st.session_state.search_query_input = ""


sc1, sc2 = st.columns([9, 1])
with sc1:
    search_query = st.text_input(
        "Search",
        placeholder="🔍 Search Order, RMA, or Tracking...",
        label_visibility="collapsed",
        key="search_query_input",
    )
with sc2:
    st.button("❌", help="Clear Search", key="clear_search_btn", on_click=clear_search_cb)

st.write("")


# --- Table Display ---
df_view = pd.DataFrame(processed_rows)

if not df_view.empty:
    f_stat = st.session_state.filter_status

    if f_stat == "Pending":
        display_df = df_view[df_view["Status"] == "Pending"]
    elif f_stat == "Approved":
        display_df = df_view[df_view["Status"] == "Approved"]
    elif f_stat == "Received":
        display_df = df_view[df_view["Status"] == "Received"]
    elif f_stat == "NoTrack":
        display_df = df_view[df_view["is_nt"] == True]
    elif f_stat == "Flagged":
        display_df = df_view[df_view["is_fg"] == True]
    else:
        display_df = df_view

    if not display_df.empty:
        display_df = display_df.sort_values(by="Created", ascending=False).reset_index(drop=True)
        display_df["No"] = (display_df.index + 1).astype(str)
        display_df["Days since updated"] = display_df["Days since updated"].astype(str)

        edited = st.data_editor(
            display_df[
                [
                    "No",
                    "RMA ID",
                    "Order",
                    "Status",
                    "Tracking Number",
                    "Tracking Status",
                    "Created",
                    "Updated",
                    "Days since updated",
                    "Update Tracking Number",
                    "View Timeline",
                ]
            ],
            width="stretch",
            height=700,
            hide_index=True,
            key=f"main_table_{st.session_state.table_key}",
            column_config={
                "No": st.column_config.TextColumn("No", width="small"),
                "RMA ID": st.column_config.TextColumn("RMA ID", width="small"),
                "Order": st.column_config.TextColumn("Order", width="small"),
                "Tracking Number": st.column_config.LinkColumn(
                    "Tracking Number", display_text=r"ref=(.*)", width="medium"
                ),
                "Tracking Status": st.column_config.TextColumn("Tracking Status", width="medium"),
                "Update Tracking Number": st.column_config.CheckboxColumn(
                    "Update Tracking Number", width="small"
                ),
                "View Timeline": st.column_config.CheckboxColumn("View Timeline", width="small"),
                "Days since updated": st.column_config.TextColumn("Days since updated", width="small"),
            },
            disabled=[
                "No",
                "RMA ID",
                "Order",
                "Status",
                "Tracking Number",
                "Tracking Status",
                "Created",
                "Updated",
                "Days since updated",
            ],
        )

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

    else:
        st.info("No matching records found in cache.")

else:
    st.warning("Database empty. Click Sync All Data to start.")
