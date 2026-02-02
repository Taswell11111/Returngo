def scrape_parcel_ninja_status(tracking_url: str) -> Optional[str]:
    """
    Scrapes the Parcel Ninja website to get the latest tracking status.
    Returns the exact status line with the pipe: "Thu, 22 Jan 12:35 | Delivered"
    """
    if not tracking_url:
        return None
    
    # Use the provided tracking URL directly
    url = tracking_url
    tracking_number = url.split('=')[-1] if '=' in url else url.split('/')[-1]

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        content = response.text

        # Try multiple regex patterns to match status
        patterns = [
            r"([A-Za-z]{3}, \d{2} [A-Za-z]{3} \d{2}:\d{2}\s+\|\s+[A-Za-z ]+)",  # Original
            r"(\d{2} [A-Za-z]{3} \d{2}:\d{2}\s+\|\s+[A-Za-z ]+)",  # Without day prefix
            r"([A-Za-z]+, \d{2} [A-Za-z]+ \d{4} \d{2}:\d{2}\s+\|\s+[A-Za-z ]+)",  # With year
            r"(Latest Status:\s*[A-Za-z ]+)",  # Simple "Latest Status: ..."
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content)
            if matches:
                # Return the exact status line
                full_status = matches[0].strip()
                logger.info(f"Scraped status '{full_status}' for tracking {tracking_number}")
                return full_status

        logger.warning(f"Could not extract status for {tracking_number}. No pattern matched.")

    except requests.exceptions.RequestException as e:
        logger.error(f"Error scraping PN tracking for {tracking_number}: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error scraping PN tracking for {tracking_number}: {e}", exc_info=True)
        
    return None
