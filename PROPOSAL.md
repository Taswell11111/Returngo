I have created the `sync_test_script.py` file for you. You can run it with `streamlit run sync_test_script.py` to test the sync function in isolation.

Regarding your second question, yes, I can create a formal Python test.

I've examined `test_levis_web.py` and it appears to be a version of your application that uses a local SQLite database, rather than a file for automated tests.

Therefore, I propose creating a new file, `test_sync_logic.py`, to house a dedicated unit test for the `perform_sync` function. This test would use the `pytest` framework and `unittest.mock` to simulate the environment, allowing us to test the logic of `perform_sync` without needing to run a UI or make real network calls.

The test would:
1.  **Simulate** the API by providing predefined example data.
2.  **Simulate** the database to confirm that the function tries to save the data correctly.
3.  **Verify** that the logic inside `perform_sync` works as expected under controlled conditions.

This is a standard practice in software development for creating reliable, automated checks on your code.

Would you like me to proceed with creating this `test_sync_logic.py` file?