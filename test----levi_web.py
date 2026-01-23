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
@@ -178,96 +180,119 @@ st.markdown(

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
@@ -316,59 +341,96 @@ def get_thread_session() -> requests.Session:
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

@@ -891,142 +953,194 @@ def get_event_date_iso(rma_payload: dict, event_name: str) -> str:
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


def get_status_time_html(s: str) -> str:
    try:
        ts = get_last_sync(s)
        if not ts:
            return "UPDATED: -"
        return f"UPDATED: {ts.strftime('%H:%M')}"
    except Exception:
        return "UPDATED: -"


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
@@ -1096,60 +1210,50 @@ for rma in raw_data:
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
@@ -1388,151 +1492,56 @@ def show_rma_actions_dialog(row: pd.Series):
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
