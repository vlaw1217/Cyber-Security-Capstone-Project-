import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

HYBRID_ANALYSIS_API_KEY = os.getenv("HYBRID_ANALYSIS_API_KEY")
HYBRID_ANALYSIS_BASE_URL = os.getenv(
    "HYBRID_ANALYSIS_BASE_URL",
    "https://hybrid-analysis.com/api/v2"
).rstrip("/")
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


def submit_file_to_hybrid_analysis(file_path, filename):
    """
    Submit a local attachment file to Hybrid Analysis / Falcon Sandbox.

    Returns the sandbox submission response.
    This function should only be called from the project workflow,
    not automatically every time this file is executed.
    """
    url = f"{HYBRID_ANALYSIS_BASE_URL}/submit/file"

    headers = get_headers()

    data = {
        "environment_id": HYBRID_ANALYSIS_ENVIRONMENT_ID,
        "comment": "Submitted from AI phishing detection capstone project",
        "submit_name": filename
    }

    with open(file_path, "rb") as file_obj:
        files = {
            "file": (filename, file_obj)
        }

        response = requests.post(
            url,
            headers=headers,
            data=data,
            files=files,
            timeout=120
        )

    if response.status_code in [200, 201]:
        return {
            "success": True,
            "status_code": response.status_code,
            "message": "File submitted to Hybrid Analysis successfully.",
            "data": response.json()
        }

    return {
        "success": False,
        "status_code": response.status_code,
        "message": "File submission to Hybrid Analysis failed.",
        "data": response.text
    }


if __name__ == "__main__":
    result = test_hybrid_analysis_connection()

    print("Success:", result["success"])
    print("Status code:", result["status_code"])
    print("Message:", result["message"])