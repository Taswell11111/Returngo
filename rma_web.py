import streamlit as st
import sqlite3
import requests
import json
import pandas as pd
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import concurrent.futures

# --- SAFE IMPORT FOR BS4 ---
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    # Define a dummy BeautifulSoup to prevent NameErrors later in code if referenced directly
    BeautifulSoup = None

# ==========================================
# 1. CONFIGURATION
# ==========================================
st.set_page_config(page_title="Bounty Apparel ReturnGo RMAs", layout="wide", page_icon="🔄️")

# ACCESS SECRETS
try:
    MY_API_KEY = st.secrets["RETURNGO_API_KEY"]
except (FileNotFoundError, KeyError):
    MY_API_KEY = os.environ.get("RETURNGO_API_KEY")

if not MY_API_KEY:
    st.error("API Key not found! Please set 'RETURNGO_API_KEY' in secrets or env vars.")
    st.stop()

# PARCEL NINJA TOKEN (TASWELL)
# Ideally store this in secrets, but hardcoded here for your specific test request
PARCEL_NINJA_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbiI6IjJhMDI4MWNhNTBmNDEwOTRiZTkyNzdhNTQ0MDZhZGRkODMyOGExODhhYmNiZGViMiIsIm5iZiI6MTc2ODk1ODUwMSwiZXhwIjoxODYzNjUyOTAxLCJpYXQiOjE3Njg5NTg1MDEsImlzcyI6Imh0dHBzOi8vb3B0aW1pc2UucGFyY2VsbmluamEuY29tIiwiYXVkIjoiaHR0cHM6Ly9vcHRpbWlzZS5wYXJjZWxuaW5qYS5jb20ifQ.lgAi9s2INGKrzGYb3Qn_PY6N1ekh3fSBP7JBgMhX0Pk"

CACHE_EXPIRY_HOURS = 4
DB_FILE = "rma_cache.db"
# LOCK FOR THREAD-SAFE DB WRITES
DB_LOCK = threading.Lock()

STORES = [
    {"name": "Diesel", "url
