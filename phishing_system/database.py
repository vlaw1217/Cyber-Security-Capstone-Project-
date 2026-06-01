import sqlite3

# ---------------------------------------------------
# Initialize database and create required tables
# ---------------------------------------------------
def init_db():
    # Connect to SQLite database (creates file if not exists)
    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()


    # -----------------------------
    # Create 'users' table
    # -----------------------------
    # Stores login credentials for users
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,   -- Unique user ID
            username TEXT UNIQUE,                   -- Username (must be unique)
            password TEXT,                          -- Hashed password
            role TEXT DEFAULT 'user'                -- User role: user or admin 
        )
    """)


    # -----------------------------
    # Create 'predictions' table
    # -----------------------------
    # Stores email analysis results
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,   -- Unique record ID
            user_id INTEGER,                        -- Reference to user (optional)
            subject TEXT,                           -- Email subject
            body TEXT,                              -- Email body content
            prediction TEXT,                        -- Model result (phishing / legit)
            probability REAL,                       -- Confidence score
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP  -- Auto timestamp
        )
    """)


    # -----------------------------
    # Create 'attachment_scans' table
    # -----------------------------
    # Stores attachment metadata and risk analysis results
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attachment_scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            -- Links attachment scan to related prediction record
            prediction_id INTEGER,

            -- Microsoft Graph message ID for the email
            message_id TEXT,

            -- Original attachment filename, e.g., invoice.pdf
            filename TEXT,

            -- File extension, e.g., .pdf, .exe, .docm
            extension TEXT,

            -- MIME type returned by Microsoft Graph
            mime_type TEXT,

            -- Attachment size in bytes
            size_bytes INTEGER,

            -- SHA-256 hash of the attachment content
            sha256_hash TEXT,

            -- Risk level: Low, Medium, or High
            risk_level TEXT,

            -- Explanation for the risk level
            risk_reason TEXT,

            -- Optional VirusTotal result
            virustotal_result TEXT DEFAULT 'Not checked',

            -- Auto timestamp for scan record
            scanned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            -- Connect this attachment record to predictions table
            FOREIGN KEY (prediction_id) REFERENCES predictions(id)
        )
    """)


        # -------------------------------
    # Create 'sandbox_scans' table
    # -------------------------------
    # Stores sandbox-based behavioral analysis results for suspicious attachments
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sandbox_scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            -- Links sandbox scan to related attachment scan record
            attachment_scan_id INTEGER,

            -- Sandbox provider information, for example Hybrid Analysis or ANY.RUN
            sandbox_provider TEXT,

            -- Task/report ID returned by the sandbox API
            sandbox_task_id TEXT,

            -- Current sandbox analysis status: submitted, running, completed, failed
            sandbox_status TEXT,

            -- Final sandbox verdict: malicious, suspicious, clean, unknown
            sandbox_verdict TEXT,

            -- Numeric score returned or calculated from sandbox result
            threat_score INTEGER,

            -- Human-readable behavior summary
            behavior_summary TEXT,

            -- Network indicators such as domains, IPs, or URLs contacted
            network_indicators TEXT,

            -- File/process indicators observed during sandbox execution
            file_indicators TEXT,

            -- Link to external sandbox report, if available
            report_url TEXT,

            -- Raw JSON/text response from sandbox API for debugging/audit
            raw_result TEXT,

            -- Timestamp when file was submitted to sandbox
            submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            -- Timestamp when sandbox report was completed/retrieved
            completed_at TIMESTAMP,

            -- Connect this sandbox record to attachment_scans table
            FOREIGN KEY (attachment_scan_id) REFERENCES attachment_scans(id)
        )
    """)


    # -----------------------------
    # Create 'header_scans' table
    # -----------------------------
    # Stores email header spoofing analysis results
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS header_scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            -- Links header scan to related prediction record
            prediction_id INTEGER,

            -- Microsoft Graph message ID for the email
            message_id TEXT,

            -- Visible sender domain from the From header
            from_domain TEXT,

            -- Bounce/envelope sender domain from Return-Path
            return_path_domain TEXT,

            -- Reply-To domain, if present
            reply_to_domain TEXT,

            -- SPF, DKIM, DMARC, and Microsoft composite authentication results
            spf_result TEXT,
            dkim_result TEXT,
            dmarc_result TEXT,
            compauth_result TEXT,

            -- DKIM signing domain from header.d
            dkim_domain TEXT,

            -- Header risk score and label
            risk_score INTEGER,
            risk_level TEXT,

            -- Human-readable explanation
            risk_reason TEXT,

            -- Auto timestamp for scan record
            scanned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            -- Connect this header record to predictions table
            FOREIGN KEY (prediction_id) REFERENCES predictions(id)
        )
    """)

    # Save changes and close connection
    conn.commit()
    conn.close()


# ---------------------------------------------------
# Retrieve all prediction records from database
# ---------------------------------------------------
def get_all_predictions():
    # Connect to database
    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()

    # SQL query to get all predictions
    # Ordered by latest first
    cursor.execute("""
        SELECT 
            predictions.id,
            users.username,
            predictions.subject,
            predictions.body,
            predictions.prediction,
            predictions.probability,
            predictions.timestamp
        FROM predictions
        JOIN users ON predictions.user_id = users.id
        ORDER BY predictions.timestamp DESC
    """)

    # Fetch all rows from query result
    rows = cursor.fetchall()

    # Close database connection
    conn.close()

    # Return data to Flask app
    return rows


# ---------------------------------------------------
# Retrieve prediction records for one specific user
# ---------------------------------------------------
def get_user_predictions(user_id):
    # Connect to database
    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()

    # Get only predictions submitted by this user
    cursor.execute("""
        SELECT 
            predictions.id,
            users.username,
            predictions.subject,
            predictions.body,
            predictions.prediction,
            predictions.probability,
            predictions.timestamp
        FROM predictions
        JOIN users ON predictions.user_id = users.id
        WHERE predictions.user_id = ?
        ORDER BY predictions.timestamp DESC
    """, (user_id,))

    rows = cursor.fetchall()

    conn.close()

    return rows

def save_attachment_scan(
    prediction_id,
    message_id,
    filename,
    extension,
    mime_type,
    size_bytes,
    sha256_hash,
    risk_level,
    risk_reason,
    virustotal_result
):
    """
    Save one attachment analysis result into the attachment_scans table.

    This function stores only attachment metadata and analysis results.
    It does not store the actual attachment file content.
    """

    # Connect to the local SQLite database.
    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()

    # Insert attachment scan result into attachment_scans table.
    cursor.execute("""
        INSERT INTO attachment_scans (
            prediction_id,
            message_id,
            filename,
            extension,
            mime_type,
            size_bytes,
            sha256_hash,
            risk_level,
            risk_reason,
            virustotal_result
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        prediction_id,
        message_id,
        filename,
        extension,
        mime_type,
        size_bytes,
        sha256_hash,
        risk_level,
        risk_reason,
        virustotal_result
    ))

    # Get the ID of the newly inserted attachment scan record.
    # This ID will be used to link sandbox analysis results to this attachment.
    attachment_scan_id = cursor.lastrowid

    # Save changes and close the connection.
    conn.commit()
    conn.close()

    return attachment_scan_id


def save_sandbox_scan(
    attachment_scan_id,
    sandbox_provider,
    sandbox_task_id,
    sandbox_status="submitted",
    sandbox_verdict=None,
    threat_score=None,
    behavior_summary=None,
    network_indicators=None,
    file_indicators=None,
    report_url=None,
    raw_result=None
):
    """
    Save a new sandbox scan record after submitting an attachment to the sandbox.
    Returns the new sandbox scan ID.
    """
    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO sandbox_scans (
            attachment_scan_id,
            sandbox_provider,
            sandbox_task_id,
            sandbox_status,
            sandbox_verdict,
            threat_score,
            behavior_summary,
            network_indicators,
            file_indicators,
            report_url,
            raw_result
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        attachment_scan_id,
        sandbox_provider,
        sandbox_task_id,
        sandbox_status,
        sandbox_verdict,
        threat_score,
        behavior_summary,
        network_indicators,
        file_indicators,
        report_url,
        raw_result
    ))

    sandbox_scan_id = cursor.lastrowid

    conn.commit()
    conn.close()

    return sandbox_scan_id

def get_sandbox_scan_by_attachment(attachment_scan_id):
    """
    Get the latest sandbox scan linked to one attachment scan.
    """
    conn = sqlite3.connect("app.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM sandbox_scans
        WHERE attachment_scan_id = ?
        ORDER BY id DESC
        LIMIT 1
    """, (attachment_scan_id,))

    sandbox_scan = cursor.fetchone()

    conn.close()

    return sandbox_scan

def update_sandbox_scan_result(
    sandbox_scan_id,
    sandbox_status,
    sandbox_verdict=None,
    threat_score=None,
    behavior_summary=None,
    network_indicators=None,
    file_indicators=None,
    report_url=None,
    raw_result=None
):
    """
    Update sandbox scan after retrieving the final sandbox report.
    """
    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE sandbox_scans
        SET
            sandbox_status = ?,
            sandbox_verdict = ?,
            threat_score = ?,
            behavior_summary = ?,
            network_indicators = ?,
            file_indicators = ?,
            report_url = ?,
            raw_result = ?,
            completed_at = CURRENT_TIMESTAMP
        WHERE id = ?
    """, (
        sandbox_status,
        sandbox_verdict,
        threat_score,
        behavior_summary,
        network_indicators,
        file_indicators,
        report_url,
        raw_result,
        sandbox_scan_id
    ))

    conn.commit()
    conn.close()


#----------------------------------------------------
# Save email header spoofing analysis result
# ---------------------------------------------------
def save_header_scan(
    prediction_id,
    message_id,
    from_domain,
    return_path_domain,
    reply_to_domain,
    spf_result,
    dkim_result,
    dmarc_result,
    compauth_result,
    dkim_domain,
    risk_score,
    risk_level,
    risk_reason
):
    """
    Save one email header spoofing analysis result into the header_scans table.
    """

    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO header_scans (
            prediction_id,
            message_id,
            from_domain,
            return_path_domain,
            reply_to_domain,
            spf_result,
            dkim_result,
            dmarc_result,
            compauth_result,
            dkim_domain,
            risk_score,
            risk_level,
            risk_reason
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        prediction_id,
        message_id,
        from_domain,
        return_path_domain,
        reply_to_domain,
        spf_result,
        dkim_result,
        dmarc_result,
        compauth_result,
        dkim_domain,
        risk_score,
        risk_level,
        risk_reason
    ))

    conn.commit()
    conn.close()


# ---------------------------------------------------
# Retrieve email header scan for one prediction
# ---------------------------------------------------
def get_header_scan(prediction_id):
    """
    Retrieve the saved email header spoofing analysis result
    for a specific prediction.
    """

    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            id,
            prediction_id,
            message_id,
            from_domain,
            return_path_domain,
            reply_to_domain,
            spf_result,
            dkim_result,
            dmarc_result,
            compauth_result,
            dkim_domain,
            risk_score,
            risk_level,
            risk_reason,
            scanned_at
        FROM header_scans
        WHERE prediction_id = ?
        ORDER BY scanned_at DESC
        LIMIT 1
    """, (prediction_id,))

    row = cursor.fetchone()

    conn.close()

    return row

# ---------------------------------------------------
# Run database initialization when file is executed
# ---------------------------------------------------
if __name__ == "__main__":
    init_db()