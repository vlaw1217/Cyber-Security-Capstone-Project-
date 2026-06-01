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

def get_hybrid_analysis_overview(sha256_hash):
    """
    Retrieve Hybrid Analysis overview information for a submitted file by SHA-256 hash.

    This is used after file submission to check whether Hybrid Analysis has
    available report/overview data for the file.
    """
    url = f"{HYBRID_ANALYSIS_BASE_URL}/overview/{sha256_hash}"

    response = requests.get(
        url,
        headers=get_headers(),
        timeout=60
    )

    if response.status_code == 200:
        return {
            "success": True,
            "status_code": response.status_code,
            "message": "Hybrid Analysis overview retrieved successfully.",
            "data": response.json()
        }

    return {
        "success": False,
        "status_code": response.status_code,
        "message": "Hybrid Analysis overview retrieval failed.",
        "data": response.text
    }


def parse_hybrid_analysis_overview(overview_data):
    """
    Convert Hybrid Analysis overview data into simple fields for SQLite/dashboard.
    """

    if not overview_data:
        return {
            "sandbox_verdict": "unknown",
            "threat_score": None,
            "behavior_summary": "No overview data returned from Hybrid Analysis.",
            "network_indicators": None,
            "file_indicators": None,
            "report_url": None,
            "raw_result": None
        }

    # Hybrid Analysis responses may vary depending on account level and report status.
    sandbox_verdict = (
        overview_data.get("verdict")
        or overview_data.get("threat_level")
        or overview_data.get("vx_family")
        or "unknown"
    )

    threat_score = (
        overview_data.get("threat_score")
        or overview_data.get("av_detect")
        or overview_data.get("total_signatures")
    )

    behavior_parts = []

    if overview_data.get("verdict"):
        behavior_parts.append(f"Verdict: {overview_data.get('verdict')}")

    if overview_data.get("threat_level"):
        behavior_parts.append(f"Threat level: {overview_data.get('threat_level')}")

    if overview_data.get("vx_family"):
        behavior_parts.append(f"Malware family: {overview_data.get('vx_family')}")

    if overview_data.get("type"):
        behavior_parts.append(f"File type: {overview_data.get('type')}")

    if overview_data.get("environment_description"):
        behavior_parts.append(f"Environment: {overview_data.get('environment_description')}")

    behavior_summary = "; ".join(behavior_parts) if behavior_parts else "No major sandbox behavior summary available yet."

    report_url = overview_data.get("url") or overview_data.get("report_url")

    return {
        "sandbox_verdict": sandbox_verdict,
        "threat_score": threat_score,
        "behavior_summary": behavior_summary,
        "network_indicators": None,
        "file_indicators": None,
        "report_url": report_url,
        "raw_result": str(overview_data)
    }

if __name__ == "__main__":
    result = test_hybrid_analysis_connection()

    print("Success:", result["success"])
    print("Status code:", result["status_code"])
    print("Message:", result["message"])


