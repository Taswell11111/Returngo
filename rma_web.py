import streamlit as st
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime, timezone
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="RMA Central | Web", layout="wide", page_icon="📦")

# --- CSS STYLING ---
st.markdown(
    """
    <style>
    /* Dark Theme Base */
    .stApp { background-color: #0e1117; color: white; }
    
    /* Blue Header Boxes */
    div[data-testid="column"] {
        background-color: #1a1c24;
        border-radius: 5px;
        padding: 10px;
        border: 1px solid #333;
    }
    
    /* Live Logger Style */
    .stTextArea textarea {
        background-color: #000000 !important;
        color: #00ff00 !important;
        font-family: 'Consolas', monospace !important;
        font-size: 12px !important;
    }
    
    /* Timeline Block */
    .timeline-box {
        background-color: #2b2b2b;
        padding: 10px;
        margin: 5px 0 5px 20px;
        border-radius: 5px;
        border-left: 3px solid #666;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- CONFIGURATION ---
MY_API_KEY = os.environ.get("ReturnGo_API", "")
if not MY_API_KEY:
    # For local testing you may set env var ReturnGo_API or change this default
    MY_API_KEY = os.environ.get("RETURNGO_API", "")

STORES = [
    {"name": "Diesel", "url": "diesel-dev-south-africa.myshopify.com"},
    {"name": "Hurley", "url": "hurley-dev-south-africa.myshopify.com"},
    {"name": "Jeep Apparel", "url": "jeep-apparel-dev-south-africa.myshopify.com"},
    {"name": "Reebok", "url": "reebok-dev-south-africa.myshopify.com"},
    {"name": "Superdry", "url": "superdry-dev-south-africa.myshopify.com"},
]

# --- SESSION STATE INITIALIZATION ---
if "logs" not in st.session_state:
    st.session_state["logs"] = []
if "rma_list" not in st.session_state:
    st.session_state["rma_list"] = []
if "counts" not in st.session_state:
    st.session_state["counts"] = {
        s["url"]: {"Pending": 0, "Approved": 0, "Received": 0, "NoTrack": 0}
        for s in STORES
    }

# --- LOGGING ---
def add_log(message: str, is_error: bool = False) -> str:
    ts = datetime.now().strftime("%H:%M:%S")
    icon = "❌" if is_error else "ℹ️"
    entry = f"[{ts}] {icon} {message}"
    st.session_state["logs"].append(entry)
    return "\n".join(st.session_state["logs"])


# --- NETWORK SESSION ---
@st.cache_resource
def get_session():
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


# --- SYNC ALL STORES (FULL) ---
def sync_counts(log_placeholder):
    session = get_session()
    st.session_state["logs"] = []

    log_text = add_log("Starting Sync Process...")
    log_placeholder.text_area("Live System Log", value=log_text, height=150, disabled=True)

    new_counts = {k: v.copy() for k, v in st.session_state["counts"].items()}

    with st.spinner("Syncing stores..."):
        for store in STORES:
            add_log(f"Syncing {store['name']}...")
            log_placeholder.text_area("Live System Log", value="\n".join(st.session_state["logs"]), height=150, disabled=True)

            headers = {"X-API-KEY": MY_API_KEY, "x-shop-name": store["url"]}
            for status in ["Pending", "Approved", "Received"]:
                try:
                    resp = session.get(
                        f"https://api.returngo.ai/rmas?status={status}&pagesize=50",
                        headers=headers,
                        timeout=8,
                    )
                    if resp.status_code == 200:
                        rmas = resp.json().get("rmas", [])
                        new_counts[store["url"]][status] = len(rmas)

                        if status == "Approved" and len(rmas) > 0:
                            no_track = 0
                            for r in rmas:
                                try:
                                    det = session.get(f"https://api.returngo.ai/rma/{r['rmaId']}", headers=headers, timeout=6)
                                    if det.status_code == 200:
                                        det_json = det.json()
                                        shipments = det_json.get("shipments", [])
                                        if not shipments or all(not s.get("trackingNumber") for s in shipments):
                                            no_track += 1
                                except Exception:
                                    # ignore individual rma failures
                                    pass
                            new_counts[store["url"]]["NoTrack"] = no_track
                            if no_track > 0:
                                add_log(f"{store['name']}: {no_track} missing tracking", is_error=True)
                                log_placeholder.text_area("Live System Log", value="\n".join(st.session_state["logs"]), height=150, disabled=True)
                except Exception as e:
                    add_log(f"Error syncing {store['name']} {status}: {e}", is_error=True)
                    log_placeholder.text_area("Live System Log", value="\n".join(st.session_state["logs"]), height=150, disabled=True)

    st.session_state["counts"] = new_counts
    add_log("Full sync complete.")
    log_placeholder.text_area("Live System Log", value="\n".join(st.session_state["logs"]), height=150, disabled=True)


# --- TARGETED SYNC FOR A SINGLE STORE/STATUS ---
def targeted_sync(store_url: str, sync_status: str | None = None, log_placeholder=None):
    session = get_session()
    headers = {"X-API-KEY": MY_API_KEY, "x-shop-name": store_url}
    counts = st.session_state["counts"].get(store_url, {"Pending": 0, "Approved": 0, "Received": 0, "NoTrack": 0}).copy()

    statuses = ["Pending", "Approved", "Received"] if sync_status is None else [sync_status]

    for status in statuses:
        try:
            resp = session.get(
                f"https://api.returngo.ai/rmas?status={status}&pagesize=50",
                headers=headers,
                timeout=8,
            )
            if resp.status_code == 200:
                rmas = resp.json().get("rmas", [])
                counts[status] = len(rmas)

                if status == "Approved":
                    no_track = 0
                    for r in rmas:
                        try:
                            det = session.get(f"https://api.returngo.ai/rma/{r['rmaId']}", headers=headers, timeout=6)
                            if det.status_code == 200:
                                det_json = det.json()
                                shipments = det_json.get("shipments", [])
                                if not shipments or all(not s.get("trackingNumber") for s in shipments):
                                    no_track += 1
                        except Exception:
                            pass
                    counts["NoTrack"] = no_track
        except Exception as e:
            if log_placeholder is not None:
                add_log(f"Targeted sync error for {store_url} {status}: {e}", is_error=True)
                log_placeholder.text_area("Live System Log", value="\n".join(st.session_state["logs"]), height=150, disabled=True)

    st.session_state["counts"][store_url] = counts


# --- FETCH TABLE DATA ---
def load_rmas(url: str, status_type: str, log_placeholder):
    session = get_session()

    add_log(f"Fetching {status_type} list for {url}...")
    log_placeholder.text_area("Live System Log", value="\n".join(st.session_state["logs"]), height=150, disabled=True)

    st.session_state["rma_list"] = []

    headers = {"X-API-KEY": MY_API_KEY, "x-shop-name": url}
    api_status = "Approved" if status_type == "NoTrack" else status_type

    try:
        resp = session.get(
            f"https://api.returngo.ai/rmas?status={api_status}&pagesize=50&sort_by=+rma_created_at",
            headers=headers,
            timeout=12,
        )
        if resp.status_code == 200:
            rmas = resp.json().get("rmas", [])
            add_log(f"Found {len(rmas)} items. Downloading details...")
            log_placeholder.text_area("Live System Log", value="\n".join(st.session_state["logs"]), height=150, disabled=True)

            temp = []
            for r in rmas:
                try:
                    det_resp = session.get(f"https://api.returngo.ai/rma/{r['rmaId']}", headers=headers, timeout=8)
                    if det_resp.status_code != 200:
                        continue
                    det = det_resp.json()

                    shipments = det.get("shipments", [])
                    track_nums = [s.get("trackingNumber") for s in shipments if s.get("trackingNumber")]
                    track_str = ", ".join(track_nums) if track_nums else "N/A"

                    if status_type == "NoTrack" and track_nums:
                        # skip rmas that have tracking when user asked for NoTrack
                        continue

                    raw_create = det.get("createdAt") or r.get("createdAt")
                    created_display = str(raw_create)[:10] if raw_create else "N/A"

                    raw_update = det.get("lastUpdated")
                    updated_display = str(raw_update)[:10] if raw_update else "N/A"

                    days_since = "0"
                    if raw_update:
                        try:
                            dt_obj = datetime.fromisoformat(raw_update.replace("Z", "+00:00"))
                            now = datetime.now(dt_obj.tzinfo or timezone.utc)
                            delta = now - dt_obj
                            days_since = str(max(0, delta.days))
                        except Exception:
                            pass

                    temp.append(
                        {
                            "id": r.get("rmaId"),
                            "order": r.get("order_name"),
                            "tracking": track_str,
                            "created": created_display,
                            "updated": updated_display,
                            "status": r.get("status"),
                            "days": days_since,
                            "timeline": det.get("comments", []),
                        }
                    )
                except Exception:
                    # ignore individual item errors
                    pass

            st.session_state["rma_list"] = temp
            add_log("Table loaded successfully.")
            log_placeholder.text_area("Live System Log", value="\n".join(st.session_state["logs"]), height=150, disabled=True)
        else:
            add_log(f"Failed fetching list: HTTP {resp.status_code}", is_error=True)
            log_placeholder.text_area("Live System Log", value="\n".join(st.session_state["logs"]), height=150, disabled=True)
    except Exception as e:
        add_log(f"Fetch error: {e}", is_error=True)
        log_placeholder.text_area("Live System Log", value="\n".join(st.session_state["logs"]), height=150, disabled=True)


# --- MAIN UI LAYOUT ---
st.title("🚀 RMA Central")

# 1. STATUS BOXES
cols = st.columns(len(STORES))
log_col_area = st.container()

for i, store in enumerate(STORES):
    c = st.session_state["counts"].get(store["url"], {"Pending": 0, "Approved": 0, "Received": 0, "NoTrack": 0})
    with cols[i]:
        st.markdown(f"**{store['name']}**")

        # unique keys for buttons using index
        if st.button(f"Pending: {c['Pending']}", key=f"pending_{i}"):
            st.session_state["trigger"] = ("load", store["url"], "Pending")
        if st.button(f"Approved: {c['Approved']}", key=f"approved_{i}"):
            st.session_state["trigger"] = ("load", store["url"], "Approved")
        if st.button(f"Received: {c['Received']}", key=f"received_{i}"):
            st.session_state["trigger"] = ("load", store["url"], "Received")

        label = f"No ID: {c['NoTrack']}"
        # no-track button key also unique
        if st.button(label, key=f"notrack_{i}"):
            st.session_state["trigger"] = ("load", store["url"], "NoTrack")

st.markdown("---")

# 2. LOGGING & ACTIONS AREA
col_act, col_log = st.columns([1, 3])

with col_act:
    if st.button("🔄 Sync All", use_container_width=True, key="sync_all_btn"):
        st.session_state["trigger"] = ("sync", None, None)

    st.markdown("**Live System Log:**")
    log_placeholder = st.empty()
    current_logs = "\n".join(st.session_state["logs"]) if st.session_state["logs"] else "Ready..."
    log_placeholder.text_area("Live System Log", value=current_logs, height=150, disabled=True, label_visibility="collapsed")

# 3. HANDLE TRIGGERS
if "trigger" in st.session_state:
    action, url, status = st.session_state["trigger"]
    del st.session_state["trigger"]

    if action == "sync":
        sync_counts(log_placeholder)
        st.experimental_rerun()
    elif action == "load":
        # targeted sync then load records
        targeted_sync(url, status, log_placeholder)
        load_rmas(url, status, log_placeholder)
        st.experimental_rerun()

# 4. DATA TABLE
with col_log:
    st.subheader(f"📋 Active List ({len(st.session_state['rma_list'])})")

    search = st.text_input("🔍 Filter (RMA, Order, or Tracking)", placeholder="Type to search...", key="search_input")

    # Header - Added "Updated" Column and Row No
    # Weights: No, ID, Order, Track, Created, Updated, Days, Action
    h0, h1, h2, h3, h4, h5, h6, h7 = st.columns([1, 2, 2, 3, 2, 2, 1, 2])
    h0.markdown("**No.**")
    h1.markdown("**RMA ID**")
    h2.markdown("**Order #**")
    h3.markdown("**Tracking**")
    h4.markdown("**Created**")
    h5.markdown("**Updated**")
    h6.markdown("**Days**")
    h7.markdown("**Action**")
    st.divider()

    # Rows (with numbering)
    for idx, row in enumerate(st.session_state["rma_list"], 1):
        search_str = f"{row.get('id','')} {row.get('order','')} {row.get('tracking','')}".lower()
        if search and search.lower() not in search_str:
            continue

        c0, c1, c2, c3, c4, c5, c6, c7 = st.columns([1, 2, 2, 3, 2, 2, 1, 2])
        c0.markdown(f"**{idx}**")
        c1.code(row.get("id", ""))
        c2.write(row.get("order", ""))
        c3.write(row.get("tracking", ""))
        c4.write(row.get("created", ""))
        c5.write(row.get("updated", ""))

        days_val = row.get("days", "0")
        days_int = int(days_val) if str(days_val).isdigit() else 0
        if days_int > 7:
            c6.error(f"{days_int}")
        else:
            c6.success(f"{days_int}")

        with c7:
            with st.expander("Timeline"):
                for t in row.get("timeline", []):
                    triggered_by = t.get("triggeredBy", "System")
                    datetime_str = (t.get("datetime") or "")[:16]
                    html = t.get("htmlText", "")
                    st.markdown(
                        f"""
                        <div class="timeline-box">
                            <small style="color:#aaa">{triggered_by} | {datetime_str}</small><br>
                            {html}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        st.markdown("<hr style='margin: 5px 0; opacity: 0.2'>", unsafe_allow_html=True)

if not st.session_state["rma_list"]:
    st.info("Select a status above to load records.")
