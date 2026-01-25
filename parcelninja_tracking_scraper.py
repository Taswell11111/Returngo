import re
from typing import Optional

import pandas as pd
import requests

BASE_URL = "https://optimise.parcelninja.com/shipment/track?WaybillNo="

# Regex for lines like:
# "Wed, 12 Nov 13:55 Courier Cancelled"
EVENT_LINE_RE = re.compile(
    r"\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),\s+\d{1,2}\s+[A-Za-z]{3}\s+\d{2}:\d{2}\s+.+"
)

# Regex to extract only the status part (everything after the time)
STATUS_EXTRACT_RE = re.compile(
    r"^(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),\s+\d{1,2}\s+[A-Za-z]{3}\s+\d{2}:\d{2}\s+(?P<status>.+)$"
)


def fetch_latest_tracking_status(tracking_number: str, fallback: str = "Unknown") -> str:
    """
    Fetch the ParcelNinja tracking page and return the latest (most recent) status text.
    IMPORTANT: URL is used only to retrieve the status and is not stored anywhere.
    """
    if not tracking_number or not isinstance(tracking_number, str):
        return fallback

    url = f"{BASE_URL}{tracking_number.strip()}"

    try:
        resp = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=20,
        )
        resp.raise_for_status()
        html = resp.text

        # Find all lines that look like event entries
        matches = EVENT_LINE_RE.findall(html)
        if not matches:
            return fallback

        # First match is the most recent line entry on the page
        latest_line = matches[0].strip()

        match = STATUS_EXTRACT_RE.match(latest_line)
        if not match:
            return fallback

        return match.group("status").strip()
    except requests.RequestException:
        return fallback


def update_table_with_tracking_status(df: pd.DataFrame, fallback: str = "Unknown") -> pd.DataFrame:
    """
    Updates df['tracking_status'] using df['tracking_number'].
    Does NOT store or link the tracking URL in the table.
    """
    if "tracking_number" not in df.columns:
        raise ValueError("DataFrame must include a 'tracking_number' column")

    if "tracking_status" not in df.columns:
        df["tracking_status"] = ""

    df["tracking_status"] = df["tracking_number"].apply(
        lambda tn: fetch_latest_tracking_status(str(tn), fallback=fallback)
    )

    return df


if __name__ == "__main__":
    # Example data table
    df = pd.DataFrame(
        {
            "order_id": ["A1001", "A1002"],
            "tracking_number": ["OPT-926690181", "OPT-000000000"],
            "tracking_status": ["", ""],
        }
    )

    updated = update_table_with_tracking_status(df)
    print(updated.to_string(index=False))
