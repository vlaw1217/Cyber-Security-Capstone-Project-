import os
import requests
from dotenv import load_dotenv


# ---------------------------------------------------
# VirusTotal Hash Reputation Checker
# ---------------------------------------------------
# This module checks whether a SHA-256 hash has known
# malicious or suspicious detections in VirusTotal.
#
# Important:
# - This does NOT upload the attachment file.
# - This only sends the SHA-256 hash value.
# - This helps protect user privacy while still checking
#   known file reputation.
# ---------------------------------------------------


# Load environment variables from .env file.
load_dotenv()

# Read VirusTotal API key from .env.
VIRUSTOTAL_API_KEY = os.getenv("VIRUSTOTAL_API_KEY")


def check_virustotal_hash(sha256_hash):
    """
    Check a SHA-256 file hash against VirusTotal API v3.

    Parameters:
        sha256_hash (str): SHA-256 hash of the attachment.

    Returns:
        dict: VirusTotal reputation result.
    """

    # If there is no hash, return a safe default result.
    if not sha256_hash:
        return {
            "status": "Not checked",
            "malicious": 0,
            "suspicious": 0,
            "harmless": 0,
            "undetected": 0,
            "message": "No SHA-256 hash available"
        }
    
    # Remove spaces or hidden newline characters from the hash.
    sha256_hash = sha256_hash.strip()

    # Validate SHA-256 format before sending request to VirusTotal.
    if len(sha256_hash) != 64 or not all(c in "0123456789abcdefABCDEF" for c in sha256_hash):
        return {
            "status": "Error",
            "malicious": 0,
            "suspicious": 0,
            "harmless": 0,
            "undetected": 0,
            "message": "Invalid SHA-256 hash format"
    }

    # If API key is missing, return a safe default result.
    if not VIRUSTOTAL_API_KEY:
        return {
            "status": "Not checked",
            "malicious": 0,
            "suspicious": 0,
            "harmless": 0,
            "undetected": 0,
            "message": "VirusTotal API key not configured"
        }

    # VirusTotal API v3 file hash lookup endpoint.
    url = f"https://www.virustotal.com/api/v3/files/{sha256_hash}"

    # VirusTotal requires the API key in the x-apikey header.
    headers = {
        "x-apikey": VIRUSTOTAL_API_KEY
    }

    try:
        response = requests.get(url, headers=headers, timeout=15)

        # 404 means VirusTotal does not know this file hash.
        if response.status_code == 404:
            return {
                "status": "Hash not found",
                "malicious": 0,
                "suspicious": 0,
                "harmless": 0,
                "undetected": 0,
                "message": "Hash not found in VirusTotal"
            }

        # 401 means API key is invalid or missing.
        if response.status_code == 401:
            return {
                "status": "Error",
                "malicious": 0,
                "suspicious": 0,
                "harmless": 0,
                "undetected": 0,
                "message": "VirusTotal API key is invalid or unauthorized"
            }

        # 429 means rate limit reached.
        if response.status_code == 429:
            return {
                "status": "Rate limited",
                "malicious": 0,
                "suspicious": 0,
                "harmless": 0,
                "undetected": 0,
                "message": "VirusTotal API rate limit reached"
            }

        # Any other non-success response is handled safely.
        if response.status_code != 200:
            return {
                "status": "Error",
                "malicious": 0,
                "suspicious": 0,
                "harmless": 0,
                "undetected": 0,
                "message": f"VirusTotal lookup failed with status code {response.status_code}"
            }

        # Parse VirusTotal JSON response.
        data = response.json()

        # last_analysis_stats contains vendor detection counts.
        stats = (
            data.get("data", {})
            .get("attributes", {})
            .get("last_analysis_stats", {})
        )

        malicious = stats.get("malicious", 0)
        suspicious = stats.get("suspicious", 0)
        harmless = stats.get("harmless", 0)
        undetected = stats.get("undetected", 0)

        # Decide reputation status based on VirusTotal detections.
        if malicious > 0:
            status = "Malicious"
            message = f"VirusTotal reported {malicious} malicious detection(s)"

        elif suspicious > 0:
            status = "Suspicious"
            message = f"VirusTotal reported {suspicious} suspicious detection(s)"

        else:
            status = "Clean"
            message = "No malicious or suspicious detections found"

        return {
            "status": status,
            "malicious": malicious,
            "suspicious": suspicious,
            "harmless": harmless,
            "undetected": undetected,
            "message": message
        }

    except requests.exceptions.RequestException as e:
        return {
            "status": "Error",
            "malicious": 0,
            "suspicious": 0,
            "harmless": 0,
            "undetected": 0,
            "message": f"VirusTotal request error: {e}"
        }