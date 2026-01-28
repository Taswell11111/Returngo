import os
import sqlite3

# Where the database should be
script_dir = r"C:\Users\Taswell\OneDrive\Documents\GitHub\Returngo"
db_file = os.path.join(script_dir, "rma_cache.db")

print(f"Looking for database at: {db_file}")
print(f"File exists: {os.path.exists(db_file)}")

if os.path.exists(db_file):
    size_kb = os.path.getsize(db_file) / 1024
    print(f"Database size: {size_kb:.1f} KB")
    
    try:
        conn = sqlite3.connect(db_file)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM rmas")
        rma_count = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM sync_log")
        sync_count = c.fetchone()[0]
        conn.close()
        
        print(f"RMAs in database: {rma_count}")
        print(f"Sync logs: {sync_count}")
    except Exception as e:
        print(f"Error reading database: {e}")
else:
    print("Database file not found. Run a sync first to create it.")