import os
import hashlib


# ---------------------------------------------------
# Attachment Risk Rules
# ---------------------------------------------------
# This file contains rule-based logic for checking
# attachment metadata. It does not open, run, or execute
# any attachment file.
#
# The goal is to classify attachments as:
# - Low risk
# - Medium risk
# - High risk
#
# based on file extension, MIME type, and file size.
# ---------------------------------------------------


# High-risk file extensions.
# These file types can execute programs, scripts, or commands.
# They are commonly abused in phishing and malware delivery.
HIGH_RISK_EXTENSIONS = {
    ".exe",   # Windows executable
    ".scr",   # Screensaver executable
    ".bat",   # Batch script
    ".cmd",   # Windows command script
    ".js",    # JavaScript file
    ".vbs",   # VBScript file
    ".ps1",   # PowerShell script
    ".msi",   # Windows installer
    ".com"    # Command executable
}


# Medium-risk file extensions.
# These file types are not always malicious, but they are often
# used to hide or deliver malicious content.
MEDIUM_RISK_EXTENSIONS = {
    ".zip",   # Compressed archive
    ".rar",   # Compressed archive
    ".7z",    # Compressed archive
    ".iso",   # Disk image
    ".docm",  # Macro-enabled Word document
    ".xlsm",  # Macro-enabled Excel workbook
    ".pptm"   # Macro-enabled PowerPoint file
}


# Lower-risk file extensions.
# These file types are usually safer, but they are not guaranteed
# to be safe. They are marked low risk unless another rule increases risk.
LOW_RISK_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".xlsx",
    ".pptx",
    ".txt",
    ".csv",
    ".png",
    ".jpg",
    ".jpeg"
}


# MIME types that are suspicious because they may represent
# executable or unknown binary content.
SUSPICIOUS_MIME_TYPES = {
    "application/x-msdownload",
    "application/octet-stream",
    "application/x-msdos-program"
}


def get_file_extension(filename):
    """
    Extract the file extension from a filename.

    Example:
        invoice.pdf  -> .pdf
        payment.docm -> .docm

    If the filename is missing or has no extension,
    the function returns an empty string.
    """

    if not filename:
        return ""

    # Convert filename to lowercase so extension matching is consistent.
    # Example: INVOICE.PDF becomes invoice.pdf
    filename = filename.lower()

    # os.path.splitext separates the filename and extension.
    # Example: "invoice.pdf" -> ("invoice", ".pdf")
    return os.path.splitext(filename)[1]

def calculate_sha256(file_bytes):
    """
    Calculate the SHA-256 hash of attachment bytes.

    This function does not open, run, or execute the attachment.
    It only creates a cryptographic fingerprint of the file content.

    Parameters:
        file_bytes (bytes): Raw attachment bytes.

    Returns:
        str or None: SHA-256 hash value, or None if bytes are missing.
    """

    # If there is no file content, return None safely.
    if not file_bytes:
        return None

    # Create and return SHA-256 hexadecimal digest.
    return hashlib.sha256(file_bytes).hexdigest()

def analyze_attachment_metadata(filename, mime_type, size_bytes):
    """
    Analyze attachment metadata using rule-based detection.

    This function does NOT open or execute the attachment.
    It only checks:
    - filename
    - file extension
    - MIME type
    - file size

    Parameters:
        filename (str): Attachment filename from Microsoft Graph.
        mime_type (str): Attachment MIME type from Microsoft Graph.
        size_bytes (int): Attachment size in bytes.

    Returns:
        dict: Attachment analysis result.
    """

    # Extract file extension from the filename.
    extension = get_file_extension(filename)

    # Default risk is Low unless a rule increases it.
    risk_level = "Low"

    # Store all reasons that explain the final risk level.
    reasons = []

    # ---------------------------------------------------
    # Rule 1: High-risk executable or script extensions
    # ---------------------------------------------------
    if extension in HIGH_RISK_EXTENSIONS:
        risk_level = "High"
        reasons.append(f"High-risk executable or script file extension: {extension}")

    # ---------------------------------------------------
    # Rule 2: Medium-risk archive or macro-enabled files
    # ---------------------------------------------------
    elif extension in MEDIUM_RISK_EXTENSIONS:
        risk_level = "Medium"
        reasons.append(f"Potentially risky attachment file extension: {extension}")

    # ---------------------------------------------------
    # Rule 3: Unknown or uncommon file extension
    # ---------------------------------------------------
    elif extension and extension not in LOW_RISK_EXTENSIONS:
        risk_level = "Medium"
        reasons.append(f"Unknown or uncommon file extension: {extension}")

    # ---------------------------------------------------
    # Rule 4: Missing file extension
    # ---------------------------------------------------
    elif not extension:
        risk_level = "Medium"
        reasons.append("Attachment has no file extension")

    # ---------------------------------------------------
    # Rule 5: Large file size
    # ---------------------------------------------------
    # If the file is larger than 10 MB, increase risk to Medium
    # unless it is already High.
    if size_bytes and size_bytes > 10 * 1024 * 1024:
        if risk_level == "Low":
            risk_level = "Medium"
        reasons.append("Large attachment size over 10 MB")

    # ---------------------------------------------------
    # Rule 6: Suspicious MIME type
    # ---------------------------------------------------
    # Some MIME types are generic binary types and may indicate
    # executable or unknown content.
    if mime_type in SUSPICIOUS_MIME_TYPES:
        risk_level = "High"
        reasons.append(f"Suspicious MIME type: {mime_type}")

    # ---------------------------------------------------
    # No risk indicators found
    # ---------------------------------------------------
    if not reasons:
        reasons.append("No obvious attachment risk detected")

    # Return structured analysis result.
    return {
        "filename": filename,
        "extension": extension,
        "mime_type": mime_type,
        "size_bytes": size_bytes,
        "risk_level": risk_level,
        "risk_reason": "; ".join(reasons)
    }