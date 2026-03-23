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
            password TEXT                           -- Password (plain text for PoC)
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
        SELECT id, user_id, subject, body, prediction, probability, timestamp
        FROM predictions
        ORDER BY timestamp DESC
    """)

    # Fetch all rows from query result
    rows = cursor.fetchall()

    # Close database connection
    conn.close()

    # Return data to Flask app
    return rows


# ---------------------------------------------------
# Run database initialization when file is executed
# ---------------------------------------------------
if __name__ == "__main__":
    init_db()