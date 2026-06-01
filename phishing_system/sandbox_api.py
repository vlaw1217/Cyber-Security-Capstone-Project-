import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

HYBRID_ANALYSIS_API_KEY = os.getenv("HYBRID_ANALYSIS_API_KEY")
HYBRID_ANALYSIS_BASE_URL = os.getenv(
    "HYBRID_ANALYSIS_BASE_URL",
    "https://www.hybrid-analysis.com/api/v2"
)
HYBRID_ANALYSIS_ENVIRONMENT_ID = os.getenv(
    "HYBRID_ANALYSIS_ENVIRONMENT_ID",
    "160"
)


def get_headers():
    """
    Build the required HTTP headers for Hybrid Analysis API requests.
    """
    if not HYBRID_ANALYSIS_API_KEY:
        raise ValueError("HYBRID_ANALYSIS_API_KEY is missing. Please check your .env file.")

    return {
        "api-key": HYBRID_ANALYSIS_API_KEY,
        "User-Agent": "Falcon",
        "Accept": "application/json"
    }


def test_hybrid_analysis_connection():
    """
    Test whether the API key can connect to Hybrid Analysis.

    This does not submit any file.
    It only checks the current API key information.
    """
    url = f"{HYBRID_ANALYSIS_BASE_URL}/key/current"

    response = requests.get(
        url,
        headers=get_headers(),
        timeout=30
    )

    if response.status_code == 200:
        return {
            "success": True,
            "status_code": response.status_code,
            "message": "Hybrid Analysis API connection successful.",
            "data": response.json()
        }

    return {
        "success": False,
        "status_code": response.status_code,
        "message": "Hybrid Analysis API connection failed.",
        "data": response.text
    }


if __name__ == "__main__":
    result = test_hybrid_analysis_connection()

    print("Success:", result["success"])
    print("Status code:", result["status_code"])
    print("Message:", result["message"])

    # Do not print the real API key.
    # Only print safe API account details if returned.
    print("Response:", result["data"])