import os
import sys
import requests
import json
import logging
import threading
from datetime import datetime, timezone
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Optional, Dict, Union, Any, Mapping

# Setup basic logging for the temporary script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Replicating essential functions from rma_web.py for standalone execution ---

# Assuming BASE_URL and RMA_COMMENT_PATH are from returngo_api.py
BASE_URL = "https://api.returngo.ai"
RMA_COMMENT_PATH = "/rma/{rma_id}/comment"

def api_url(path: str) -> str:
    normalized = path.lstrip("/")
    return f"{BASE_URL}/{normalized}"

# Replace _thread_local with a simple global session for this standalone script
_global_session = None

def get_global_session() -> requests.Session:
    global _global_session
    if _global_session is None:
        _global_session = requests.Session()
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
        _global_session.mount("https://", HTTPAdapter(max_retries=retries))
    return _global_session

# Rate limiting variables (simplified for a single-run script)
_last_req_ts = 0.0
RG_RPS = 2 # Requests per second

def _sleep_for_rate_limit():
    global _last_req_ts
    if RG_RPS <= 0:
        return
    min_interval = 1.0 / float(RG_RPS)
    now = datetime.now().timestamp()
    wait = (_last_req_ts + min_interval) - now
    if wait > 0:
        import time
        time.sleep(wait)
    _last_req_ts = datetime.now().timestamp()

RATE_LIMIT_INFO: Dict[str, Union[int, str, datetime, None]] = {"remaining": None, "limit": None, "reset": None, "updated_at": None}
# No need for a lock in a single-threaded script

def update_rate_limit_info(headers: Mapping[str, Optional[str]]):
    if not headers:
        return
    lower = {str(k).lower(): v for k, v in headers.items()}
    remaining = lower.get("x-ratelimit-remaining") or lower.get("x-rate-limit-remaining") or lower.get("ratelimit-remaining")
    limit = lower.get("x-ratelimit-limit") or lower.get("x-rate-limit-limit") or lower.get("ratelimit-limit")
    reset = lower.get("x-ratelimit-reset") or lower.get("x-rate-limit-reset") or lower.get("ratelimit-reset")

    if remaining is None and limit is None and reset is None:
        return

    def to_int(val):
        if val is None: return None
        sval = str(val).strip()
        return int(sval) if sval.isdigit() else sval

    RATE_LIMIT_INFO["remaining"] = to_int(remaining)
    RATE_LIMIT_INFO["limit"] = to_int(limit)
    RATE_LIMIT_INFO["reset"] = to_int(reset)
    RATE_LIMIT_INFO["updated_at"] = datetime.now(timezone.utc)

def rg_request(method: str, url: str, store_url: str, *, headers=None, timeout=15, json_body=None):
    logger.info(f"rg_request: Making {method} request to {url} for store {store_url}")
    session: requests.Session = get_global_session()
    
    # Construct headers - MY_API_KEY needs to be available in this scope
    # For this temporary script, let's assume MY_API_KEY is defined globally or passed
    request_headers = {
        "x-api-key": MY_API_KEY_GLOBAL, # Use a global variable for the API key
        "X-API-KEY": MY_API_KEY_GLOBAL,
        "x-shop-name": store_url
    }
    if headers:
        request_headers.update(headers)

    backoff = 1
    res = None
    for _attempt in range(1, 6):
        _sleep_for_rate_limit()
        logger.info(f"rg_request: Attempt {_attempt}...")
        try:
            res = session.request(method, url, headers=request_headers, timeout=timeout, json=json_body)
            logger.info(f"rg_request: Request completed with status {res.status_code}")
            update_rate_limit_info(res.headers)
            if res.status_code != 429:
                return res
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            res = None # Ensure res is None on exception

        if res is None: # Handle cases where request failed or returned 429
            sleep_s = backoff
            backoff = min(backoff * 2, 30)
            logger.info(f"Waiting for {sleep_s} seconds before retrying...")
            import time
            time.sleep(sleep_s)
        elif res.status_code == 429:
            ra = res.headers.get("Retry-After")
            if ra and ra.isdigit():
                sleep_s = int(ra)
            else:
                sleep_s = backoff
                backoff = min(backoff * 2, 30)
            logger.info(f"Rate limit hit. Waiting for {sleep_s} seconds before retrying...")
            import time
            time.sleep(sleep_s)
            res = None # Reset res to None to force another attempt

    return None

def fetch_rma_detail(rma_id: str, store_url: str):
    logger.info(f"Fetching detail for RMA {rma_id} from {store_url}")
    url = api_url(f"/rma/{rma_id}")
    
    res = rg_request("GET", url, store_url, timeout=20)
    if res is None:
        logger.error(f"Failed to fetch detail for {rma_id}: No response after retries.")
        return None

    if res.status_code == 200:
        logger.info(f"Detail fetched for {rma_id}")
        return res.json()
    else:
        logger.warning(f"Failed to fetch detail for {rma_id}: Status {res.status_code}, Response: {res.text}")
        return None

def push_tracking_update(rma_id: str, shipment_id: str, new_tracking: str, store_url: str):
    logger.info(f"Pushing tracking update for RMA {rma_id}, shipment {shipment_id} to {new_tracking} for store {store_url}")
    url = api_url(f"/shipment/{shipment_id}")
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "status": "LabelCreated", # Assuming this is the status when tracking is updated
        "carrierName": "CourierGuy", # Placeholder, adjust as needed
        "trackingNumber": new_tracking,
        "trackingURL": f"https://optimise.parcelninja.com/shipment/track?WaybillNo={new_tracking}", # Placeholder
        "labelURL": "https://sellerportal.dpworld.com/api/file-download?link=null", # Placeholder
    }
    
    res = rg_request("PUT", url, store_url, headers=headers, json_body=payload, timeout=15)
    if res is None:
        return False, "API Error: No response from ReturnGO API after retries."

    if res.status_code in [200, 201]:
        logger.info(f"Tracking for {rma_id} updated successfully.")
        return True, "Success"
    else:
        error_msg = f"API Error {res.status_code}: {res.text}"
        logger.error(f"Failed to update tracking for {rma_id}: {error_msg}")
        return False, error_msg

def push_comment_update(rma_id: str, comment_text: str, store_url: str):
    logger.info(f"Adding comment to RMA {rma_id} for store {store_url}")
    url = api_url(RMA_COMMENT_PATH.format(rma_id=rma_id))
    headers = {
        "Content-Type": "application/json"
    }
    payload = {"htmlText": comment_text}
    try:
        res = rg_request("POST", url, store_url, headers=headers, json_body=payload, timeout=15)
        if res is None:
            return False, "API Error: No response from ReturnGO API after retries."

        if res.status_code in [200, 201]:
            logger.info(f"Comment added to RMA {rma_id} successfully.")
            return True, "Success"
        return False, f"API Error {res.status_code}: {res.text}"
    except Exception as e:
        logger.error(f"Exception adding comment to RMA {rma_id}: {e}")
        return False, str(e)


# --- Main execution logic ---
if __name__ == "__main__":
    # Ensure MY_API_KEY_GLOBAL is set from environment
    MY_API_KEY_GLOBAL = os.environ.get("RETURNGO_API_KEY_BOUNTY")
    if not MY_API_KEY_GLOBAL:
        logger.error("RETURNGO_API_KEY_BOUNTY environment variable not set.")
        sys.exit(1)

    # User provided data
    rma_to_update = "9113828"
    store_to_update = "superdry-dev-south-africa.myshopify.com"
    new_tracking_number = "OPT-782082327"
    
    logger.info(f"Attempting to update RMA: {rma_to_update}")
    logger.info(f"Store: {store_to_update}")
    logger.info(f"New Tracking Number: {new_tracking_number}")

    # Step 1: Fetch RMA details to get shipment_id
    rma_details = fetch_rma_detail(rma_to_update, store_to_update)

    shipment_id = None
    if rma_details and rma_details.get('shipments'):
        for shipment in rma_details['shipments']:
            # Assuming we need to update the first shipment or a specific one if there are multiple
            # For simplicity, taking the first shipment's ID
            shipment_id = shipment.get('shipmentId')
            if shipment_id:
                break
    
    if not shipment_id:
        logger.error(f"Could not find shipment ID for RMA {rma_to_update}.")
        sys.exit(1)

    logger.info(f"Found shipment ID: {shipment_id} for RMA {rma_to_update}.")

    # Step 2: Push tracking update
    success, message = push_tracking_update(rma_to_update, shipment_id, new_tracking_number, store_to_update)

    if success:
        logger.info(f"Successfully updated tracking for RMA {rma_to_update}.")
    else:
        logger.error(f"Failed to update tracking for RMA {rma_to_update}: {message}")
        sys.exit(1)

    # Optional: Add a comment to the RMA about the tracking update
    comment_text = f"Tracking number updated to: {new_tracking_number}"
    logger.info(f"Adding comment to RMA {rma_to_update}: {comment_text}")
    comment_success, comment_message = push_comment_update(rma_to_update, comment_text, store_to_update)

    if comment_success:
        logger.info(f"Successfully added comment to RMA {rma_to_update}.")
    else:
        logger.warning(f"Failed to add comment to RMA {rma_to_update}: {comment_message}")
        # This is a warning, not a critical failure, so don't exit.
