import unittest
import sqlite3
from levis_web import init_db, DB_FILE, clear_db

class TestDBInitialization(unittest.TestCase):
    def test_db_initialization(self):
        print("Starting test_db_initialization...")
        
        # 1. Clear the database to ensure a clean state
        print("Clearing the database...")
        clear_db()

        # 2. Initialize the database
        print("Initializing the database...")
        init_db()

        # 3. Query the database to check if the 'rmas' table exists
        print("Querying the database...")
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        try:
            c.execute("SELECT COUNT(*) FROM rmas")
            count = c.fetchone()[0]
            print(f"Found {count} RMAs in the database.")
        except sqlite3.OperationalError as e:
            self.fail(f"The 'rmas' table was not created correctly: {e}")
        finally:
            conn.close()

if __name__ == '__main__':
    unittest.main()
