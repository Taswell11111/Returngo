import unittest
import os

DB_FILE = "levis_cache.db"

class TestFileIO(unittest.TestCase):
    def test_file_io(self):
        print("Starting test_file_io...")

        # 1. Delete the database file if it exists
        if os.path.exists(DB_FILE):
            print(f"Deleting existing database file: {DB_FILE}")
            os.remove(DB_FILE)
            self.assertFalse(os.path.exists(DB_FILE))

        # 2. Create the database file
        print(f"Creating new database file: {DB_FILE}")
        with open(DB_FILE, "w") as f:
            f.write("test")
        self.assertTrue(os.path.exists(DB_FILE))

        # 3. Delete the database file
        print(f"Deleting database file: {DB_FILE}")
        os.remove(DB_FILE)
        self.assertFalse(os.path.exists(DB_FILE))

        print("test_file_io completed.")

if __name__ == '__main__':
    unittest.main()
