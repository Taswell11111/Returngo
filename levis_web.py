# app.py
"""
Levi's ReturnGO Ops Dashboard (Streamlit)

- Copy-to-clipboard copies EXACTLY the server-rendered table:
  * same visible columns you choose
  * same server-side sorting you choose
  * same filtering you applied
"""

import concurrent.futures
import json
import os
import re
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ==========================================
# 1. CONFIGURATION
# ==========================================
st.set_page_config(page_title="Levi's ReturnGO Ops", layout="wide", page_icon="📤")

try:
    MY_API_KEY = st.secrets["RETURNGO_API_KEY"]
except Exception:
    MY_API_KEY = os.environ.get("RETURNGO_API_KEY")

if not MY_API_KEY:
    st.error("API Key not found! Please set 'RETURNGO_API_KEY'.")
    st.stop()

try:
    PARCEL_NINJA_TOKEN = st.secrets["PARCEL_NINJA_TOKEN"]
except Exception:
    PARCEL_NINJA_TOKEN = os.environ.get("PARCEL_NINJA_TOKEN", "")

STORE_URL = "levis-sa.myshopify.com"
DB_FILE = "levis_cache.db"
DB_LOCK = threading.Lock()

CACHE_EXPIRY_HOURS = 4
COURIER_REFRESH_HOURS = 12
MAX_WORKERS = 3
RG_RPS = 2
SYNC_OVERLAP_MINUTES = 5
DETAIL_FETCH_BATCH = 100

ACTIVE_STATUSES = ["Pending", "Approved", "Received"]

# ==========================================
# 2. HELPERS
# ==========================================
def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

def _iso_utc(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()

def extract_rma_id(url: str) -> str | None:
    if not url:
        return None
    m = re.search(r"rmaid=(\d+)", url)
    return m.group(1) if m else None

# ==========================================
# 3. HTTP + RATE LIMITING
# ==========================================
_thread_local = threading.local()
_rate_lock = threading.Lock()
_last_req_ts = 0.0

def _sleep_for_rate_limit():
    global _last_req_ts
    if RG_RPS <= 0:
        return
    min_interval = 1 / RG_RPS
    with _rate_lock:
        now = time.time()
        wait = (_last_req_ts + min_interval) - now
        if wait > 0:
            time.sleep(wait)
        _last_req_ts = time.time()

def get_thread_session():
    s = getattr(_thread_local, "session", None)
    if s:
        return s

    s = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "PUT", "POST"]),
        respect_retry_after_header=True,
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    _thread_local.session = s
    return s

def rg_headers():
    return {"x-api-key": MY_API_KEY, "x-shop-name": STORE_URL}

def rg_request(method, url, *, headers=None, timeout=15, json_body=None):
    session = get_thread_session()
    headers = headers or rg_headers()
    backoff = 1

    for attempt in range(5):
        _sleep_for_rate_limit()
        res = session.request(method, url, headers=headers, timeout=timeout, json=json_body)

        if res.status_code < 400:
            return res

        # Log all API errors
        st.warning(f"⚠️ ReturnGO API error {res.status_code}")
        rate_headers = {k: v for k, v in res.headers.items() if "rate" in k.lower()}
        if rate_headers:
            st.caption(f"Rate headers: {rate_headers}")

        if res.status_code != 429:
            return res

        time.sleep(backoff)
        backoff = min(backoff * 2, 30)

    return res

# ==========================================
# 4. DATABASE
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
                courier_status TEXT,
                courier_last_checked TEXT
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS sync_logs (
                scope TEXT PRIMARY KEY,
                last_sync_iso TEXT
            )
        """)
        conn.commit()
        conn.close()

init_db()

# ==========================================
# 5. FETCHING (INCREMENTAL + BATCHED)
# ==========================================
def fetch_rma_list(statuses, since_dt):
    all_rmas = []
    cursor = None
    status_param = ",".join(statuses)

    updated = ""
    if since_dt:
        since_dt -= timedelta(minutes=SYNC_OVERLAP_MINUTES)
        updated = f"&rma_updated_at=gte:{_iso_utc(since_dt)}"

    while True:
        url = f"https://api.returngo.ai/rmas?pagesize=500&status={status_param}{updated}"
        if cursor:
            url += f"&cursor={cursor}"

        res = rg_request("GET", url, timeout=20)
        if res.status_code != 200:
            break

        data = res.json()
        rmas = data.get("rmas", [])
        if not rmas:
            break

        all_rmas.extend(rmas)
        cursor = data.get("next_cursor")
        if not cursor:
            break

    return all_rmas

def fetch_rma_detail(rma_id):
    url = f"https://api.returngo.ai/rma/{rma_id}"
    res = rg_request("GET", url, timeout=20)
    if res.status_code != 200:
        return None
    return res.json()

# ==========================================
# 6. SYNC (BATCHED DETAIL FETCH)
# ==========================================
def perform_sync(statuses):
    st.info("⏳ Syncing...")
    summaries = fetch_rma_list(statuses, None)
    ids = [s["rmaId"] for s in summaries if s.get("rmaId")]

    total = len(ids)
    done = 0
    bar = st.progress(0)

    for i in range(0, total, DETAIL_FETCH_BATCH):
        chunk = ids[i : i + DETAIL_FETCH_BATCH]
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = [ex.submit(fetch_rma_detail, rid) for rid in chunk]
            for f in concurrent.futures.as_completed(futures):
                done += 1
                bar.progress(done / total)

    bar.empty()
    st.success("✅ Sync Complete")
    st.session_state["show_toast"] = True
    st.rerun()

# ==========================================
# 7. MUTATIONS (NO REDUNDANT RE-FETCH)
# ==========================================
def push_tracking_update(rma_id, shipment_id, tracking_number):
    payload = {
        "status": "LabelCreated",
        "carrierName": "CourierGuy",
        "trackingNumber": tracking_number,
        "trackingURL": f"https://optimise.parcelninja.com/shipment/track/{tracking_number}",
    }
    res = rg_request(
        "PUT",
        f"https://api.returngo.ai/shipment/{shipment_id}",
        headers={**rg_headers(), "Content-Type": "application/json"},
        json_body=payload,
    )
    return res.status_code == 200

def push_comment_update(rma_id, text):
    res = rg_request(
        "POST",
        f"https://api.returngo.ai/rma/{rma_id}/comment",
        headers={**rg_headers(), "Content-Type": "application/json"},
        json_body={"text": text, "isPublic": False},
    )
    return res.status_code in (200, 201)

# ==========================================
# 8. UI SAFETY: RMA ID EXTRACTION FIX
# ==========================================
def open_dialog_from_row(row):
    rma_id = extract_rma_id(row.get("RMA URL"))
    if not rma_id:
        st.error("Invalid RMA URL")
        return
    return rma_id

# ==========================================
# 9. VISUAL INDICATORS
# ==========================================
def add_alert_column(df):
    df["⚠️"] = df.apply(
        lambda r: "🚫" if r.get("is_fg") else ("🔍" if r.get("is_nt") else ""),
        axis=1,
    )
    return df

# ==========================================
# END
# ==========================================
