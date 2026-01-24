import sqlite3
DB_FILE = "levis_cache.db"

try:
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM rmas")
    count = c.fetchone()[0]
    conn.close()
    print(f"Found {count} RMAs in the database.")
except Exception as e:
    print(f"An error occurred: {e}")
