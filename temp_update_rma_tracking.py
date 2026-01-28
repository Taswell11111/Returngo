import os
import requests
import json
from typing import Optional, Dict

# --- Configuration ---
BASE_URL = "https://api.returngo.ai"

def api_url(path: str) -> str:
    normalized = path.lstrip("/")
    return f"{BASE_URL}/{normalized}"

MY_API_KEY = os.environ.get("RETURNGO_API_KEY")
if not MY_API_KEY:
    print("Error: RETURNGO_API_KEY_BOUNTY_BOUNTY environment variable not set.")
    exit(1)

def rg_headers(store_url: str) -> Dict[str, Optional[str]]:
    return {"x-api-key": MY_API_KEY, "X-API-KEY": MY_API_KEY, "x-shop-name": store_url}

# --- Simplified API Call for Debugging ---
def fetch_rma_detail(rma_id, store_url):
    url = api_url(f"/rma/{rma_id}")
    headers = rg_headers(store_url)
    print(f"fetch_rma_detail: Making a direct GET request to {url}")
    try:
        res = requests.get(url, headers=headers, timeout=20)
        print(f"fetch_rma_detail: Received status code: {res.status_code}")
        print(f"fetch_rma_detail: Response headers: {res.headers}")
        print(f"fetch_rma_detail: Response content: {res.text[:500]}") # Print first 500 chars

        if res.status_code == 200:
            return res.json()
        else:
            return None
    except requests.exceptions.RequestException as e:
        print(f"fetch_rma_detail: An exception occurred: {e}")
        return None
        
def push_tracking_update(rma_id, shipment_id, new_tracking, store_url):
    """Update tracking number for a shipment"""
    url = api_url(f"/shipment/{shipment_id}") # Changed to PUT /shipment/{shipmentId} as per API docs
    headers = {
        "x-api-key": MY_API_KEY,
        "X-API-KEY": MY_API_KEY,
        "x-shop-name": store_url,
        "Content-Type": "application/json"
    }
    payload = { # Added required fields for PUT /shipment/{shipmentId}
        "status": "LabelCreated", # Assuming this is the status when tracking is updated
        "carrierName": "CourierGuy", # Placeholder, adjust as needed
        "trackingNumber": new_tracking,
        "trackingURL": f"https://optimise.parcelninja.com/shipment/track?WaybillNo={new_tracking}", # Placeholder
        "labelURL": "https://sellerportal.dpworld.com/api/file-download?link=null", # Placeholder,
    }
    try:
        res = requests.put(url, headers=headers, json=payload, timeout=15)
        if res.status_code in [200, 201]:
            return True, "Success"
        return False, f"API Error {res.status_code}"
    except Exception as e:
        return False, str(e)

# --- Main execution logic ---
if __name__ == "__main__":
    target_rma_id = "9875594"
    new_tracking_number = "OPT-338683449"
    store_url_to_use = "diesel-dev-south-africa.myshopify.com"

    print(f"Attempting to fetch details for RMA {target_rma_id} from store {store_url_to_use}")

    rma_detail = fetch_rma_detail(target_rma_id, store_url_to_use)

    if rma_detail:
        print("\nSuccessfully fetched RMA details.")
        shipments = rma_detail.get('shipments', [])
        if shipments:
            shipment_id = shipments[0].get('shipmentId')
            if shipment_id:
                print(f"Found shipment ID: {shipment_id}")
                print(f"\nProceeding to update tracking...")
                success, message = push_tracking_update(target_rma_id, shipment_id, new_tracking_number, store_url_to_use)
                if success:
                    print(f"Tracking update successful: {message}")
                else:
                    print(f"Tracking update failed: {message}")
            else:
                print(f"Error: No shipmentId found for RMA {target_rma_id}")
        else:
            print(f"Error: No shipments found for RMA {target_rma_id}")
    else:
        print("\nFailed to fetch RMA details. Halting.")
