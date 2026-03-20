import sqlite3

def init_db():
    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()

    # Users table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT
    )
    """)

    # Predictions table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        subject TEXT,
        body TEXT,
        prediction TEXT,
        probability REAL
    )
    """)

    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()