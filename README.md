# ReturnGo — RMA Dashboard & Tools

ReturnGo is a small collection of Streamlit-based dashboards and utilities for working with ReturnGo's RMA API. The repo provides an operations dashboard that can:

- Sync RMA counts across multiple stores
- Load RMA detail lists (Pending, Approved, Received, and "NoTrack" — approved RMAs missing tracking)
- Inspect and export RMA data to CSV
- Push tracking updates to the ReturnGo API
- Cache RMA detail responses in a local SQLite DB (used by a store-specific app)

The repository currently includes:
- `rma_web.py` — central Streamlit dashboard for multiple stores (counts, live logs, data load/export)
- `rma_dashboard.py` — another Streamlit dashboard variant focused on a multi-store view / metrics
- `levis_web.py` — store-specific app (Levi's) with SQLite caching and deeper sync logic
- helper functions for requests sessions, retry logic, pagination, DB caching and push operations

Features
- Streamlit UI for quick operational workflows and drilldown
- Robust HTTP session with retry adapters
- Cursor-based pagination to fetch all pages of RMAs
- Optional local SQLite caching with expiry to reduce API calls
- Export to CSV / data editor UI
- Simple logging area in the UI for operational visibility

Requirements
- Python 3.8+
- Recommended packages:
  - streamlit
  - requests
  - pandas
  - sqlite3 (standard library)
  - urllib3 (for Retry)
  - concurrent.futures (standard library)

Quickstart (local)
1. Clone the repo:
   ```bash
   git clone https://github.com/Taswell11111/Returngo.git
   cd Returngo
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # macOS / Linux
   .venv\Scripts\activate      # Windows (PowerShell)
   ```

3. Install dependencies:
   ```bash
   pip install streamlit requests pandas
   ```

Configuration / Secrets
The apps need a ReturnGo API key to call the ReturnGo API.

There are two ways the code reads the key (depending on the file/app):
- Environment variable `ReturnGo_API` (used by `rma_dashboard.py` and `rma_web.py`)
  - Example:
    ```bash
    export ReturnGo_API="your_api_key_here"
    ```
- Streamlit secrets (used by `levis_web.py`)
  - Create `.streamlit/secrets.toml` with:
    ```toml
    RETURNGO_API_KEY = "your_api_key_here"
    ```

Confirm which file you intend to run and set the appropriate secret mechanism.

Running the apps
- Run the multi-store dashboard:
  ```bash
  streamlit run rma_web.py
  ```
- Run the alternative dashboard:
  ```bash
  streamlit run rma_dashboard.py
  ```
- Run the store-specific (Levi's) app:
  ```bash
  streamlit run levis_web.py
  ```

How to use the UI (high level)
- Click "Sync All" to refresh counts across configured stores.
- Use the per-store buttons to view lists filtered by status (Pending, Approved, Received).
- The "No ID" or "NoTrack" feature finds Approved RMAs that have no tracking numbers.
- Loaded results show in a data editor where you can copy or export as CSV.

Caching & Database
- `levis_web.py` uses a local SQLite cache (`levis_cache.db`) to store RMA details and avoid repeated API hits.
- Cache expiry is controlled via `CACHE_EXPIRY_HOURS` (default: 4 hours) within `levis_web.py`.

Code structure notes
- `get_session()` sets up a requests.Session with retry/backoff adapters.
- `fetch_all_pages()` handles cursor-based pagination to retrieve complete RMA lists.
- `fetch_rma_detail()` fetches detailed RMA objects and optionally persists them to the local DB.
- Push/update functions call the ReturnGo API endpoints for shipments and RMAs.

Security & Rate Limits
- Do not commit your API keys to source control.
- Respect ReturnGo API rate limits: the code uses retries and timeouts — adjust concurrency and timeouts if you hit rate throttling.
- If run in a production environment, provide the API secret via secure secret storage (Streamlit secrets for deployed Streamlit Cloud, environment variables in container orchestration, or a secret store).

Deployment suggestions
- Streamlit Cloud: add secrets via the app settings and set `RETURNGO_API_KEY` or `ReturnGo_API`.
- Docker: create a lightweight container and pass the API key as an environment variable. Example Dockerfile steps:
  - Use python:3.10-slim
  - Copy files, pip install requirements, expose port and run `streamlit run ...`
- When deploying, ensure the SQLite file is stored in a writable volume if caching persists between restarts.

Troubleshooting
- "API Key not found" — ensure the proper env var or secrets.toml entry is set.
- Timeouts / partial results — increase timeouts, reduce concurrent workers, or check API health.
- Missing createdAt in RMA summaries — the code attempts to fall back to event dates where available.

Extending / Contributing
- Add a Dockerfile and `requirements.txt` to standardize local/dev environments.
- Add unit tests around pagination and caching logic.
- Consider adding authentication and role-based access controls if multiple users will manage RMAs.
- Open issues or PRs with changes. If you want collaboration help, describe which file or feature you'd like to extend.

License
- No license file included by default. If you want the project to be open-source, consider adding an MIT license:
  ```text
  MIT License
  ```
  (Add a `LICENSE` file at repo root.)

Contact
- Repo owner / maintainer: Taswell11111 (on GitHub)

Acknowledgements
- Built around the ReturnGo API and Streamlit for quick operational tooling.

If you want, I can:
- Add a `requirements.txt` or `Dockerfile` to this repo
- Create a `LICENSE` (MIT) and a CONTRIBUTING.md
- Add example `.streamlit/secrets.toml` template (gitignored) for local development

```
