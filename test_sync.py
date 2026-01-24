import unittest
import sqlite3
from unittest.mock import patch
from levis_web import perform_sync, clear_db, DB_FILE

class TestSync(unittest.TestCase):
    @patch('levis_web.st')
    def test_perform_sync(self, mock_st):
        print("Starting test_perform_sync...")
        
        # 1. Clear the database to ensure a clean state
        print("Clearing the database...")
        clear_db()

        try:
            print("Running perform_sync...")
            perform_sync()
            print("perform_sync completed.")
        except Exception as e:
            self.fail(f"perform_sync raised an unexpected exception: {e}")

        # 2. Query the database to check if data has been saved
        print("Querying the database...")
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM rmas")
        count = c.fetchone()[0]
        conn.close()

        print(f"Found {count} RMAs in the database.")
        self.assertGreater(count, 0, "No RMAs were saved to the database.")

if __name__ == '__main__':
    unittest.main()