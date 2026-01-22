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
        --greenbg: rgba(34,197,94,0.22);48,0.13);
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
        bord       box-shadow: 0 0 0 1px rgba(34,197,94,0.25), 0 12px 28px rgba(0,0,0,0.28);
      }absolut.rg-tile::before {
        content: "";
        position: absolute;
        left: 10px; right: 10px; top: 28px;
        height: 10px;
        border-radia(34,197,94,0.45);
      }14px;,2.rg-updated-pill {
        position: absolute;
        left: 14px;
        top: 8px;      .rg-api {
        display:flex;
        gap:10px2px;
        border-radius: 14px;
        background: rgba(17,24,39,0.72);
        border: 1px solid  .rg-mini {
        margin-top: -14px;,240,0.95);
      }
      .rg-api b { font-weight: 900; }

      /* Righ/* API pill */
      .rg-api {
        display:flex;
        gap:10px2px;
        border-radius: 14px;
        background: rgba(17,24,39,0.72);
        border: 1px solid rgba(148,163,184,0.18);
        box-shadow: 0 10px 26px rgba(0,0,0,0.25);
        margin-bottom: 10px;
        font-size: .rg-mini {
        margin-top: -14px;,240,0.95);
      }
      .rg-api b { font-weight: 900; }

      /* Righ/* API pill */
      .rg-api {
        display:flex;
        gap:10px;
        align-items:center;
        justify-content:center;
        padding: 10px 12px;
        border-radius: 14px;
        background: rgba(17,24,39,0.72);
        border: 1px solid rgba(148,163,184,0.18);
        box-shadow: 0 10px 26px rgba(0,0,0,0.25);
        margin-bottom: 10px;
        font-size: .rg-mini {
        margin-top: -14px;,240,0.95);
      }
      .rg-api b { font-weight: 900; }

      /* Righ/* API pill */
      .rg-api {
        display:flex;
        gap:10px;
        align-items:center;
        justify-content:center;
        padding: 10px 12px;
        border-radius: 14px;
        background: rgba(17,24,39,0.72);
        border: 1px solid rgba(148,163,184,0.18);
        box-shadow: 0 10px 26px rgba(0,0,0,0.25);
        margin-bottom: 10px;
        font-size: 0.92rem;
        color: rgba(226,232,240,0.95);
      }
      .rg-api b { font-weight: 900; }

      /* Right control card */ 0.0

= 0.0

= 0.0

# Best-effort capture o_last_req_ts = 0.0

# Best-effort capture o_last_req_ts = 0.0

# Best-effort capture o_last_req_ts = 0.0

# Best-effort capture o_last_req_ts = 0.0

# Best-effort capture o_last_req_ts = 0.0

# Best-effort capture o_last_req_ts = 0.0

# Best-effort capture o_last_req_ts = 0.0

# Best-effort capture o_last_req_ts = 0.0

# Best-effort capture o_last_req_ts = 0.0

# Best-effort capture o_last_req_ts = 0.0

# Best-effort capture o_last_req_ts = 0.0

# Best-effort capture o_last_req_ts = 0.0

# Best-effort capture o_last_req_ts = 0.0

# Best-effort capture o_last_req_ts = 0.0

# Best-effort capture o_last_req_ts = 0.0

# Best-effort capture o_last_req_ts = 0.0

# Best-effort capture o_last_req_ts = 0.0

# Best-effort capture o_last_req_ts = 0.0

# Best-effort capture o_last_req_ts = 0.0

# Best-effort capture o_last_req_ts = 0.0

# Best-effort capture o_last_req_ts = 0.0

# Best-effort capture o_last_req_ts = 0.0

# Best-effort capture o_last_req_ts = 0.0

# Best-effort capture of ReturnGO rate-limit headers
_rateinfo_lock = threading.Lock()
_rate_info = {"limit": None, "remaining": None, "reset": None, "ts": None}
it headers
_rateinfo_lock = threading.Lock()
_rate_info = {"limit": None, "remaining": None, "reset": None, "ts": None}
it headers
_rateinfo_lock = threading.Lock()
_rate_info = {"limit": None, "remaining": None, "reset": None, "ts": None}
it headers
_rateinfo_lock = threading.Lock()
_rate_info = {"limit": None, "remaining": None, "reset": None, "ts": None}
it headers
_rateinfo_lock = threading.Lock()
_rate_info = {"limit": None, "remaining": None, "reset": None, "ts": None}
it headers
_rateinfo_lock = threading.Lock()
_rate_info = {"limit": None, "remaining": None, "reset": None, "ts": None}
it headers
_rateinfo_lock = threading.Lock()
_rate_info = {"limit": None, "remaining": None, "reset": None, "ts": None}
it headers
_rateinfo_lock = threading.Lock()
_rate_info = {"limit": None, "remaining": None, "reset": None, "ts": None}
it headers
_rateinfo_lock = threading.Lock()
_rate_info = {"limit": None, "remaining": None, "reset": None, "ts": None}
it headers
_rateinfo_lock = threading.Lock()
_rate_info = {"limit": None, "remaining": None, "reset": None, "ts": None}
it headers
_rateinfo_lock = threading.Lock()
_rate_info = {"limit": None, "remaining": None, "reset": None, "ts": None}
itdef _capture_rate_headers(res: requests.Response):
    """Capture rate-limit headers (best-effort). Safe to call from worker threads."""
    try:
        h = res.headers or {}
        # Common header variants
        limit = h.get("X-RateLimit-Limit") or h.get("x-ratelimit-limit")
        remaining = h.get("X-RateLimit-Remaining") or h.get("x-ratelimit-remaining")
        reset = h.get("X-RateLimit-Reset") or h.get("x-ratelimit-reset")
        with _rateinfo_lock:
            _rate_info["limit"] = limit
            _rate_info["remaining"] = remaining
            _rate_info["reset"] = reset
            _rate_info["ts"] = _iso_utc(_now_utc())
    except Exception:
        pass


def rg_request(method: str, url: str, *, headers=None, timeout=15, json_body=None):
    session = get_thread_session()
    headers = headers or rg_headers()

    backoff = 1
    for _attempt in range(1, 6):
        _sleep_for_rate_limit()
        res = session.request(method, url, headers=headers, timeout=timeout, json=json_body)

        # Capture rate headers on every response
        _capture_rate_headers(res)

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

    return resmit-limit")
        remaining = h.get("X-RateLimit-Remaining") or h.get("x-ratelimit-remaining")
        reset = h.get("X-RateLimit-Reset") or h.get("x-ratelimit-reset")
        with _rateinfo_lock:
            _rate_info["limit"] = limit
            _rate_info["remaining"] = remaining
            _rate_info["reset"] = reset
            _rate_info["ts"] = _iso_utc(_now_utc())
    except Exception:
        pass


def rg_request(method: str, url: str, *, headers=None, timeout=15, json_body=None):
    session = get_thread_session()
    headers = headers or rg_headers()

    backoff = 1
    for _attempt in range(1, 6):
        _sleep_for_rate_limit()
        res = session.request(method, url, headers=headers, timeout=timeout, json=json_body)

        # Capture rate headers on every response
        _capture_rate_headers(res)

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

    return resmit-limit")
        remaining = h.get("X-RateLimit-Remaining") or h.get("x-ratelimit-remaining")
        reset = h.get("X-RateLimit-Reset") or h.get("x-ratelimit-reset")
        with _rateinfo_lock:
            _rate_info["limit"] = limit
            _rate_info["remaining"] = remaining
            _rate_info["reset"] = reset
            _rate_info["ts"] = _iso_utc(_now_utc())
    except Exception:
        pass


def rg_request(method: str, url: str, *, headers=None, timeout=15, json_body=None):
    session = get_thread_session()
    headers = headers or rg_headers()

    backoff = 1
    for _attempt in range(1, 6):
        _sleep_for_rate_limit()
        res = session.request(method, url, headers=headers, timeout=timeout, json=json_body)

        # Capture rate headers on every response
        _capture_rate_headers(res)

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

    return resmit-limit")
        remaining = h.get("X-RateLimit-Remaining") or h.get("x-ratelimit-remaining")
        reset = h.get("X-RateLimit-Reset") or h.get("x-ratelimit-reset")
        with _rateinfo_lock:
            _rate_info["limit"] = limit
            _rate_info["remaining"] = remaining
            _rate_info["reset"] = reset
            _rate_info["ts"] = _iso_utc(_now_utc())
    except Exception:
        pass


def rg_request(method: str, url: str, *, headers=None, timeout=15, json_body=None):
    session = get_thread_session()
    headers = headers or rg_headers()

    backoff = 1
    for _attempt in range(1, 6):
        _sleep_for_rate_limit()
        res = session.request(method, url, headers=headers, timeout=timeout, json=json_body)

        # Capture rate headers on every response
        _capture_rate_headers(res)

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

    return resmit-limit")
        remaining = h.get("X-RateLimit-Remaining") or h.get("x-ratelimit-remaining")
        reset = h.get("X-RateLimit-Reset") or h.get("x-ratelimit-reset")
        with _rateinfo_lock:
            _rate_info["limit"] = limit
            _rate_info["remaining"] = remaining
            _rate_info["reset"] = reset
            _rate_info["ts"] = _iso_utc(_now_utc())
    except Exception:
        pass


def rg_request(method: str, url: str, *, headers=None, timeout=15, json_body=None):
    session = get_thread_session()
    headers = headers or rg_headers()

    backoff = 1
    for _attempt in range(1, 6):
        _sleep_for_rate_limit()
        res = session.request(method, url, headers=headers, timeout=timeout, json=json_body)

        # Capture rate headers on every response
        _capture_rate_headers(res)

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

    return resmit-limit")
        remaining = h.get("X-RateLimit-Remaining") or h.get("x-ratelimit-remaining")
        reset = h.get("X-RateLimit-Reset") or h.get("x-ratelimit-reset")
        with _rateinfo_lock:
            _rate_info["limit"] = limit
            _rate_info["remaining"] = remaining
            _rate_info["reset"] = reset
            _rate_info["ts"] = _iso_utc(_now_utc())
    except Exception:
        pass


def rg_request(method: str, url: str, *, headers=None, timeout=15, json_body=None):
    session = get_thread_session()
    headers = headers or rg_headers()

    backoff = 1
    for _attempt in range(1, 6):
        _sleep_for_rate_limit()
        res = session.request(method, url, headers=headers, timeout=timeout, json=json_body)

        # Capture rate headers on every response
        _capture_rate_headers(res)

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

    return resmit-limit")
        remaining = h.get("X-RateLimit-Remaining") or h.get("x-ratelimit-remaining")
        reset = h.get("X-RateLimit-Reset") or h.get("x-ratelimit-reset")
        with _rateinfo_lock:
            _rate_info["limit"] = limit
            _rate_info["remaining"] = remaining
            _rate_info["reset"] = reset
            _rate_info["ts"] = _iso_utc(_now_utc())
    except Exception:
        pass


def rg_request(method: str, url: str, *, headers=None, timeout=15, json_body=None):
    session = get_thread_session()
    headers = headers or rg_headers()

    backoff = 1
    for _attempt in range(1, 6):
        _sleep_for_rate_limit()
        res = session.request(method, url, headers=headers, timeout=timeout, json=json_body)

        # Capture rate headers on every response
        _capture_rate_headers(res)

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

    return resmit-limit")
        remaining = h.get("X-RateLimit-Remaining") or h.get("x-ratelimit-remaining")
        reset = h.get("X-RateLimit-Reset") or h.get("x-ratelimit-reset")
        with _rateinfo_lock:
            _rate_info["limit"] = limit
            _rate_info["remaining"] = remaining
            _rate_info["reset"] = reset
            _rate_info["ts"] = _iso_utc(_now_utc())
    except Exception:
        pass


def rg_request(method: str, url: str, *, headers=None, timeout=15, json_body=None):
    session = get_thread_session()
    headers = headers or rg_headers()

    backoff = 1
    for _attempt in range(1, 6):
        _sleep_for_rate_limit()
        res = session.request(method, url, headers=headers, timeout=timeout, json=json_body)

        # Capture rate headers on every response
        _capture_rate_headers(res)

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

    return resmit-limit")
        remaining = h.get("X-RateLimit-Remaining") or h.get("x-ratelimit-remaining")
        reset = h.get("X-RateLimit-Reset") or h.get("x-ratelimit-reset")
        with _rateinfo_lock:
            _rate_info["limit"] = limit
            _rate_info["remaining"] = remaining
            _rate_info["reset"] = reset
            _rate_info["ts"] = _iso_utc(_now_utc())
    except Exception:
        pass


def rg_request(method: str, url: str, *, headers=None, timeout=15, json_body=None):
    session = get_thread_session()
    headers = headers or rg_headers()

    backoff = 1
    for _attempt in range(1, 6):
        _sleep_for_rate_limit()
        res = session.request(method, url, headers=headers, timeout=timeout, json=json_body)

        # Capture rate headers on every response
        _capture_rate_headers(res)

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

    return resmit-limit")
        remaining = h.get("X-RateLimit-Remaining") or h.get("x-ratelimit-remaining")
        reset = h.get("X-RateLimit-Reset") or h.get("x-ratelimit-reset")
        with _rateinfo_lock:
            _rate_info["limit"] = limit
            _rate_info["remaining"] = remaining
            _rate_info["reset"] = reset
            _rate_info["ts"] = _iso_utc(_now_utc())
    except Exception:
        pass


def rg_request(method: str, url: str, *, headers=None, timeout=15, json_body=None):
    session = get_thread_session()
    headers = headers or rg_headers()

    backoff = 1
    for _attempt in range(1, 6):
        _sleep_for_rate_limit()
        res = session.request(method, url, headers=headers, timeout=timeout, json=json_body)

        # Capture rate headers on every response
        _capture_rate_headers(res)

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

    return resmit-limit")
        remaining = h.get("X-RateLimit-Remaining") or h.get("x-ratelimit-remaining")
        reset = h.get("X-RateLimit-Reset") or h.get("x-ratelimit-reset")
        with _rateinfo_lock:
            _rate_info["limit"] = limit
            _rate_info["remaining"] = remaining
            _rate_info["reset"] = reset
            _rate_info["ts"] = _iso_utc(_now_utc())
    except Exception:
        pass


def rg_request(method: str, url: str, *, headers=None, timeout=15, json_body=None):
    session = get_thread_session()
    headers = headers or rg_headers()

    backoff = 1
    for _attempt in range(1, 6):
        _sleep_for_rate_limit()
        res = session.request(method, url, headers=headers, timeout=timeout, json=json_body)

        # Capture rate headers on every response
        _capture_rate_headers(res)

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

    return resmit-limit")
        remaining = h.get("X-RateLimit-Remaining") or h.get("x-ratelimit-remaining")
        reset = h.get("X-RateLimit-Reset") or h.get("x-ratelimit-reset")
        with _rateinfo_lock:
            _rate_info["limit"] = limit
            _rate_info["remaining"] = remaining
            _rate_info["reset"] = reset
            _rate_info["ts"] = _iso_utc(_now_utc())
    except Exception:
        pass


def rg_request(method: str, url: str, *, headers=None, timeout=15, json_body=None):
    session = get_thread_session()
    headers = headers or rg_headers()

    backoff = 1
    for _attempt in range(1, 6):
        _sleep_for_rate_limit()
        res = session.request(method, url, headers=headers, timeout=timeout, json=json_body)

        # Capture rate headers on every response
        _capture_rate_headers(res)

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

    return resmit-limit")
        remaining = h.get("X-RateLimit-Remaining") or h.get("x-ratelimit-remaining")
        reset = h.get("X-RateLimit-Reset") or h.get("x-ratelimit-reset")
        with _rateinfo_lock:
            _rate_info["limit"] = limit
            _rate_info["remaining"] = remaining
            _rate_info["reset"] = reset
            _rate_info["ts"] = _iso_utc(_now_utc())
    except Exception:
        pass


def rg_request(method: str, url: str, *, headers=None, timeout=15, json_body=None):
    session = get_thread_session()
    headers = headers or rg_headers()

    backoff = 1
    for _attempt in range(1, 6):
        _sleep_for_rate_limit()
        res = session.request(method, url, headers=headers, timeout=timeout, json=json_body)

        # Capture rate headers on every response
        _capture_rate_headers(res)

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

    return resmit-limit")
        remaining = h.get("X-RateLimit-Remaining") or h.get("x-ratelimit-remaining")
        reset = h.get("X-RateLimit-Reset") or h.get("x-ratelimit-reset")
        with _rateinfo_lock:
            _rate_info["limit"] = limit
            _rate_info["remaining"] = remaining
            _rate_info["reset"] = reset
            _rate_info["ts"] = _iso_utc(_now_utc())
    except Exception:
        pass


def rg_request(method: str, url: str, *, headers=None, timeout=15, json_body=None):
    session = get_thread_session()
    headers = headers or rg_headers()

    backoff = 1
    for _attempt in range(1, 6):
        _sleep_for_rate_limit()
        res = session.request(method, url, headers=headers, timeout=timeout, json=json_body)

        # Capture rate headers on every response
        _capture_rate_headers(res)

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

    return resmit-limit")
        remaining = h.get("X-RateLimit-Remaining") or h.get("x-ratelimit-remaining")
        reset = h.get("X-RateLimit-Reset") or h.get("x-ratelimit-reset")
        with _rateinfo_lock:
            _rate_info["limit"] = limit
            _rate_info["remaining"] = remaining
            _rate_info["reset"] = reset
            _rate_info["ts"] = _iso_utc(_now_utc())
    except Exception:
        pass


def rg_request(method: str, url: str, *, headers=None, timeout=15, json_body=None):
    session = get_thread_session()
    headers = headers or rg_headers()

    backoff = 1
    for _attempt in range(1, 6):
        _sleep_for_rate_limit()
        res = session.request(method, url, headers=headers, timeout=timeout, json=json_body)

        # Capture rate headers on every response
        _capture_rate_headers(res)

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

    return resmit-limit")
        remaining = h.get("X-RateLimit-Remaining") or h.get("x-ratelimit-remaining")
        reset = h.get("X-RateLimit-Reset") or h.get("x-ratelimit-reset")
        with _rateinfo_lock:
            _rate_info["limit"] = limit
            _rate_info["remaining"] = remaining
            _rate_info["reset"] = reset
            _rate_info["ts"] = _iso_utc(_now_utc())
    except Exception:
        pass


def rg_request(method: str, url: str, *, headers=None, timeout=15, json_body=None):
     rg_headers()

    backoff = 1
    for _attempt in range(1, 6):
        _sleep_for_rate_limit()
        res = session.request(method, url, headers=headers, timeout=timeout, json=json_body)

        # Capture rate headers on every response
        _capture_rate_headers(res)

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

    return resmit-limit")
        remaining = h.get("X-RateLimit-Remaining") or h.get("x-ratelimit-remaining")
        reset = h.get("X-RateLimit-Reset") or h.get("x-ratelimit-reset")
        with _rateinfo_lock:
            _rate_info["limit"] = limit
            _rate_info["remaining"] = remaining
  def pretty_resolution(rt: str) -> str:
    if not rt:
        return ""
    # e.g. RefundToPaymentMethod -> Refund To Payment Method
    out = re.sub(r"([a-z])([A-Z])", r"\ \", rt).strip()
    return out.replace("To ", "to ")headers=None, timeout=15, json_body=None):
     rg_headers()

    backoff = 1
    for _attempt in range(1, 6):
        _sleep_for_rate_limit()
        res = session.request(method, url, headers=headers, timeout=timeout, json=json_body)

        # Capture rate headers on every response
        _capture_rate_headers(res)

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

    return resmit-limit")
        remaining = h.get("X-RateLimit-Remaining") or h.get("x-ratelimit-remaining")
        reset = h.get("X-RateLimit-Reset") or h.get("x-ratelimit-reset")
        with _rateinfo_lock:
            _rate_info["limit"] = limit
            _rate_info["remaining"] = remaining
  def pretty_resolution(rt: str) -> str:
    if not rt:
        return ""
    # e.g. RefundToPaymentMethod -> Refund To Payment Method
    out = re.sub(r"([a-z])([A-Z])", r"\ \", rt).strip()
    return out.replace("To ", "to ")headers=None, timeout=15, json_body=None):
     rg_headers()

    backoff = 1
    for _attempt in range(1, 6):
        _sleep_for_rate_limit()
        res = session.request(method, url, headers=headers, timeout=timeout, json=json_body)

        # Capture rate headers on every response
        _capture_rate_headers(res)

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

    return resmit-limit")
        remaining = h.get("X-RateLimit-Remaining") or h.get("x-ratelimit-remaining")
        reset = h.get("X-RateLimit-Reset") or h.get("x-ratelimit-reset")
        with _rateinfo_lock:
            _rate_info["limit"] = limit
            _rate_info["remaining"] = remaining
  def pretty_resolution(rt: str) -> str:
    if not rt:
        return ""
    # e.g. RefundToPaymentMethod -> Refund To Payment Method
    out = re.sub(r"([a-z])([A-Z])", r"\ \", rt).strip()
    return out.replace("To ", "to ")headers=None, timeout=15, json_body=None):
     rg_headers()

    backoff = 1
    for _attempt in range(1, 6):
        _sleep_for_rate_limit()
        res = session.request(method, url, headers=headers, timeout=timeout, json=json_body)

        # Capture rate headers on every response
        _capture_rate_headers(res)

        if res.status_code != 429:
            return res

        RATE_LIMIT_HIT.set()
        ra = res.headers.get("Retry-After")
        if ra and ra.isdigit():
            sleep_s = int(ra)
        else:
            sleep_s = backoff
            backoff = min(backoff * 2, 30)

        timewith top_right:
    st.markdown("<div class='rg-right-card'>", unsafe_allow_html=True)

    def _api_box_text() -> str:
        with _rateinfo_lock:
            lim = _rate_info.get("limit")
            rem = _rate_info.get("remaining")
            reset = _rate_info.get("reset")
            ts = _rate_info.get("ts")

        # Format reset if it looks like epoch seconds
        reset_txt = "-"
        try:
            if reset and str(reset).isdigit():
                r = int(str(reset))
                # If it's too small, treat as seconds-from-now
                if r < 10_000_000:
                    dt = _now_utc() + timedelta(seconds=r)
                else:
                    dt = datetime.fromtimestamp(r, tz=timezone.utc)
                reset_txt = dt.strftime("%H:%M")
        except Exception:
            reset_txt = "-"

        lim_txt = str(lim) if lim else "-"
        rem_txt = str(rem) if rem else "-"
        ts_txt = ts[11:16] if isinstance(ts, str) and len(ts) >= 16 else "-"
        return f"<div class='rg-api'><span>API Remaining:</span> <b>{rem_txt}/{lim_txt}</b><span style='opacity:.55'>|</span><span>Updated:</span> <b>{ts_txt}</b><span style='opacity:.55'>|</span><span>Reset:</span> <b>{reset_txt}</b></div>"

    st.markdown(_api_box_text(), unsafe_allow_html=True)

    if st.button("🔄 Sync Dashboard", key="btn_sync_all", use_container_width=True):
        perform_sync()
    if st.button("🗑️ Reset Cache", key="btn_reset", use_container_width=True):
        if clear_db():
            st.success("Cache cleared!")
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)PaymentMethod -> Refund To Payment Method
    out = re.sub(r"([a-z])([A-Z])", r"\ \", rt).strip()
    return out.replace("To ", "to ")headers=None, timeout=15, json_body=None):
     rg_headers()

    backoff = 1
    for _attempt in range(1, 6):
        _sleep_for_rate_limit()
        res = session.request(method, url, headers=headers, timeout=timeout, json=json_body)

        # Capture rate headers on every response
        _capture_rate_headers(res)

        if res.status_code != 429:
            return res

        RATE_LIMIT_HIT.set()
        ra = res.headers.get("Retry-After")
        if ra and ra.isdigit():
            sleep_s = int(ra)
        else:
            sleep_s = backoff
            backoff = min(backoff * 2, 30)

        timewith top_right:
    st.markdown("<div class='rg-right-card'>", unsafe_allow_html=True)

    def _api_box_text() -> str:
        with _rateinfo_lock:
            lim = _rate_info.get("limit")
            rem = _rate_info.get("remaining")
            reset = _rate_info.get("reset")
            ts = _rate_info.get("ts")

        # Format reset if it looks like epoch seconds
        reset_txt = "-"
        try:
            if reset and str(reset).isdigit():
                r = int(str(reset))
                # If it's too small, treat as seconds-from-now
                if r < 10_000_000:
                    dt = _now_utc() + timedelta(seconds=r)
                else:
                    dt = datetime.fromtimestamp(r, tz=timezone.utc)
                reset_txt = dt.strftime("%H:%M")
        except Exception:
            reset_txt = "-"

        lim_txt = str(lim) if lim else "-"
        rem_txt = str(rem) if rem else "-"
        ts_txt = ts[11:16] if isinstance(ts, str) and len(ts) >= 16 else "-"
        return f"<div class='rg-api'><span>API Remaining:</span> <b>{rem_txt}/{lim_txt}</b><span style='opacity:.55'>|</span><span>Updated:</span> <b>{ts_txt}</b><span style='opacity:.55'>|</span><span>Reset:</span> <b>{reset_txt}</b></div>"

    st.markdown(_api_box_text(), unsafe_allow_html=True)

    if st.button("🔄 Sync Dashboard", key="btn_sync_all", use_container_width=True):
        perform_sync()
    if st.button("🗑️ Reset Cache", key="btn_reset", use_container_width=True):
        if clear_db():
            st.success("Cache cleared!")
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)PaymentMethod -> Refund To Payment Method
    out = re.sub(r"([a-z])([A-Z])", r"\ \", rt).strip()
    return out.replace("To ", "to ")headers=None, timeout=15, json_body=None):
     rg_headers()

    backoff = 1
    for _attempt in range(1, 6):
        _sleep_for_rate_limit()
        res = session.request(method, url, headers=headers, timeout=timeout, json=json_body)

        # Capture rate headers on every response
        _capture_rate_headers(res)

        if res.status_code != 429:
            return res

        RATE_LIMIT_HIT.set()
        ra = res.headers.get("Retry-After")
        if ra and ra.isdigit():
            sleep_s = int(ra)
        else:
            sleep_s = backoff
            backoff = min(backoff * 2, 30)

        timewith top_right:
    st.markdown("<div class='rg-right-card'>", unsafe_allow_html=True)

    def _api_box_text() -> str:
        with _rateinfo_lock:
            lim = _rate_info.get("limit")
            rem = _rate_info.get("remaining")
            reset = _rate_info.get("reset")
            ts = _rate_info.get("ts")

        # Format reset if it looks like epoch seconds
        reset_txt = "-"
        try:
            if reset and str(reset).isdigit():
                r = int(str(reset))
                # If it's too small, treat as seconds-from-now
                if r < 10_000_000:
                    dt = _now_utc() + timedelta(seconds=r)
                else:
                    dt = datetime.fromtimestamp(r, tz=timezone.utc)
                reset_txt = dt.strftime("%H:%M")
        except Exception:
            reset_txt = "-"

        lim_txt = str(lim) if lim else "-"
        rem_txt = str(rem) if rem else "-"
        ts_txt = ts[11:16] if isinstance(ts, str) and len(ts) >= 16 else "-"
        return f"<div class='rg-api'><span>API Remaining:</span> <b>{rem_txt}/{lim_txt}</b><span style='opacity:.55'>|</span><span>Updated:</span> <b>{ts_txt}</b><span style='opacity:.55'>|</span><span>Reset:</span> <b>{reset_txt}</b></div>"

    st.markdown(_api_box_text(), unsafe_allow_html=True)

    if st.button("🔄 Sync Dashboard", key="btn_sync_all", use_container_width=True):
        perform_sync()
    if st.button("🗑️ Reset Cache", key="btn_reset", use_container_width=True):
        if clear_db():
            st.success("Cache cleared!")
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)PaymentMethod -> Refund To Payment Method
    out = re.sub(r"([a-z])([A-Z])", r"\ \", rt).strip()
    return out.replace("To ", "to ")headers=None, timeout=15, json_body=None):
     rg_headers()

    backoff = 1
    for _attempt in range(1, 6):
        _sleep_for_rate_limit()
        res = session.request(method, url, headers=headers, timeout=timeout, json=json_body)

        # Capture rate headers on every response
        _capture_rate_headers(res)

        if res.status_code != 429:
            return res

        RATE_LIMIT_HIT.set()
        ra = res.headers.get("Retry-After")
        if ra and ra.isdigit():
            sleep_s = int(ra)
        else:
            sleep_s = backoff
            backoff = min(backoff * 2, 30)

        timewith top_right:
    st.markdown("<div class='rg-right-card'>", unsafe_allow_html=True)

    def _api_box_text() -> str:
        with _rateinfo_lock:
            lim = _rate_info.get("limit")
            rem = _rate_info.get("remaining")
            reset = _rate_info.get("reset")
            ts = _rate_info.get("ts")

        # Format reset if it looks like epoch seconds
        reset_txt = "-"
        try:
            if reset and str(reset).isdigit():
                r = int(str(reset))
                # If it's too small, treat as seconds-from-now
                if r < 10_000_000:
                    dt = _now_utc() + timedelta(seconds=r)
                else:
                    dt = datetime.fromtimestamp(r, tz=timezone.utc)
                reset_txt = dt.strftime("%H:%M")
        except Exception:
            reset_txt = "-"

        lim_txt = str(lim) if lim else "-"
        rem_txt = str(rem) if rem else "-"
        ts_txt = ts[11:16] if isinstance(ts, str) and len(ts) >= 16 else "-"
        return f"<div class='rg-api'><span>API Remaining:</span> <b>{rem_txt}/{lim_txt}</b><span style='opacity:.55'>|</span><span>Updated:</span> <b>{ts_txt}</b><span style='opacity:.55'>|</span><span>Reset:</span> <b>{reset_txt}</b></div>"

    st.markdown(_api_box_text(), unsafe_allow_html=True)

    if st.button("🔄 Sync Dashboard", key="btn_sync_all", use_container_width=True):
        perform_sync()
    if st.button("🗑️ Reset Cache", key="btn_reset", use_container_width=True):
        if clear_db():
            st.success("Cache cleared!")
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)PaymentMethod -> Refund To Payment Method
    out = re.sub(r"([a-z])([A-Z])", r"\ \", rt).strip()
    return out.replace("To ", "to ")headers=None, timeout=15, json_body=None):
     rg_headers()

    backoff = 1
    for _attempt in range(1, 6):
        _sleep_for_rate_limit()
        res = session.request(method, url, headers=headers, timeout=timeout, json=json_body)

        # Capture rate headers on every response
        _capture_rate_headers(res)

        if res.status_code != 429:
            return res

        RATE_LIMIT_HIT.set()
        ra = res.headers.get("Retry-After")
        if ra and ra.isdigit():
            sleep_s = int(ra)
        else:
            sleep_s = backoff
            backoff = min(backoff * 2, 30)

        timewith top_right:
    st.markdown("<div class='rg-right-card'>", unsafe_allow_html=True)

    def _api_box_text() -> str:
        with _rateinfo_lock:
            lim = _rate_info.get("limit")
            rem = _rate_info.get("remaining")
            reset = _rate_info.get("reset")
            ts = _rate_info.get("ts")

        # Format reset if it looks like epoch seconds
        reset_txt = "-"
        try:
            if reset and str(reset).isdigit():
                r = int(str(reset))
                # If it's too small, treat as seconds-from-now
                if r < 10_000_000:
                    dt = _now_utc() + timedelta(seconds=r)
                else:
                    dt = da"RMA ID": st.column_config.LinkColumn(
        "RMA ID",
        # Extract the RMA id from the URL so the cell shows the ID (but remains clickable)
        display_text=r"rmaid=(\d+)",
        width="small",
    ),
