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

# ---------------------------------------------------
# Run database initialization when file is executed
# ---------------------------------------------------
if __name__ == "__main__":
    init_db()