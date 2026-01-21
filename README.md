ReturnGo — RMA Dashboard & Tools

ReturnGo is a small collection of Streamlit-based dashboards and utilities for working with ReturnGo's RMA API. The repo provides an operations dashboard that can:

Sync RMA counts across multiple stores

Load RMA detail lists (Pending, Approved, Received, and "NoTrack" — approved RMAs missing tracking)

Inspect and export RMA data to CSV

Push tracking updates to the ReturnGo API

Cache RMA detail responses in a local SQLite DB (used by rma_web.py and levis_web.py)

The repository currently includes:

rma_web.py — Main central Streamlit dashboard for multiple stores (counts, live logs, data load/export, threading lock).

levis_web.py — Store-specific app (Levi's) with SQLite caching.

rma_dashboard.py — Legacy dashboard variant.

Features

Streamlit UI for quick operational workflows and drilldown

Robust HTTP session with retry adapters

Cursor-based pagination to fetch all pages of RMAs

Thread-safe SQLite caching to allow fast parallel downloads

Export to CSV / data editor UI

Requirements

Python 3.8+

Recommended packages:

streamlit

requests

pandas

sqlite3 (standard library)

urllib3 (for Retry)

concurrent.futures (standard library)

Quickstart (local)

Clone the repo:

git clone [https://github.com/Taswell11111/Returngo.git](https://github.com/Taswell11111/Returngo.git)
cd Returngo


Create and activate a virtual environment:

python -m venv .venv
source .venv/bin/activate   # macOS / Linux
.venv\Scripts\activate      # Windows (PowerShell)


Install dependencies:

pip install streamlit requests pandas


Configuration / Secrets

All apps now use the same configuration key.

You must set RETURNGO_API_KEY. You can do this in two ways:

Streamlit Secrets (Recommended for Local/Cloud):
Create a file at .streamlit/secrets.toml:

RETURNGO_API_KEY = "your_api_key_here"


Environment Variable:

export RETURNGO_API_KEY="your_api_key_here"


Running the apps

Run the main multi-store dashboard:

streamlit run rma_web.py


Run the store-specific (Levi's) app:

streamlit run levis_web.py


Caching & Database

rma_web.py uses rma_cache.db

levis_web.py uses levis_cache.db

Cache expiry is controlled via CACHE_EXPIRY_HOURS (default: 4 hours).

Note: The databases use a threading.Lock() to prevent write errors during high-speed parallel syncs.

Security & Rate Limits

Do not commit your API keys to source control.

Respect ReturnGo API rate limits: the code uses retries and timeouts.

License

MIT License
