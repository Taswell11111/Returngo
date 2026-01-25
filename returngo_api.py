BASE_URL = "https://api.returngo.ai"
RMA_COMMENT_PATH = "/rma/{rma_id}/comment"


def api_url(path: str) -> str:
    normalized = path.lstrip("/")
    return f"{BASE_URL}/{normalized}"
