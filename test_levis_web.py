import unittest
from unittest.mock import patch, MagicMock, call
from datetime import datetime, timezone

# Since the functions are in levis_web.py, we need to import them.
# If you run this test file from the same directory as levis_web.py, this will work.
import levis_web

class TestLevisWebSync(unittest.TestCase):

    @patch('levis_web.st')
    @patch('levis_web.get_incremental_since')
    @patch('levis_web.fetch_rma_list')
    @patch('levis_web.get_local_ids_for_status')
    @patch('levis_web.delete_rmas')
    @patch('levis_web.should_refresh_detail')
    @patch('levis_web.fetch_rma_detail')
    @patch('levis_web.set_last_sync')
    @patch('levis_web._now_utc')
    def test_perform_sync_incremental_success(
        self,
        mock_now_utc,
        mock_set_last_sync,
        mock_fetch_rma_detail,
        mock_should_refresh,
        mock_delete_rmas,
        mock_get_local_ids,
        mock_fetch_list,
        mock_get_since,
        mock_st
    ):
        """
        Tests the perform_sync function for a successful incremental sync.
        """
        print("\nRunning: test_perform_sync_incremental_success")

        # --- 1. Setup Mocks ---

        # Mock Streamlit UI elements to avoid errors
        mock_st.empty.return_value = MagicMock()
        mock_st.progress.return_value = MagicMock()

        # Mock timings and dates
        mock_now = datetime(2023, 10, 27, 10, 0, 0, tzinfo=timezone.utc)
        mock_now_utc.return_value = mock_now
        mock_get_since.return_value = datetime(2023, 10, 27, 9, 0, 0) # Incremental sync

        # Mock API responses
        mock_fetch_list.return_value = (
            [{"rmaId": "101"}, {"rmaId": "102"}], # summaries
            True, # ok
            None  # err
        )

        # Mock DB interactions
        mock_get_local_ids.return_value = {"101", "103"} # "103" is stale and should be deleted
        
        # For incremental sync, should_refresh_detail is not called, but we mock it for safety
        mock_should_refresh.return_value = True

        # --- 2. Execute Function ---
        
        # We need to wrap this in a try/except because st.rerun() raises an exception
        try:
            levis_web.perform_sync(statuses=['Pending', 'Approved', 'Received'], full=False)
        except Exception as e:
            # We expect a RerunException from st.rerun(), but since st is a mock,
            # it might raise something else if not configured. We pass if it's related to rerun.
            if 'rerun' not in str(e).lower():
                self.fail(f"perform_sync raised an unexpected exception: {e}")

        # --- 3. Assertions ---

        # Check if it correctly identified stale RMAs to delete
        print("Asserting that stale RMAs are deleted...")
        mock_delete_rmas.assert_called_once_with({'103'})

        # Check if it fetches details for the new/updated RMAs from the API list
        print("Asserting that details for new/updated RMAs are fetched...")
        # For incremental sync, `force` should be True
        calls = [call('101', force=True), call('102', force=True)]
        mock_fetch_rma_detail.assert_has_calls(calls, any_order=True)
        self.assertEqual(mock_fetch_rma_detail.call_count, 2)

        # Check if the sync timestamps are updated correctly
        print("Asserting that sync logs are updated...")
        scope_call = call('Pending,Approved,Received', mock_now)
        pending_call = call('Pending', mock_now)
        approved_call = call('Approved', mock_now)
        received_call = call('Received', mock_now)
        mock_set_last_sync.assert_has_calls([scope_call, pending_call, approved_call, received_call], any_order=True)

        # Check for success toast
        print("Asserting that a success toast is shown...")
        self.assertTrue(mock_st.session_state.__setitem__.called)
        mock_st.session_state.__setitem__.assert_any_call("show_toast", True)

        print("Test completed successfully.")

    @patch('levis_web.st')
    @patch('levis_web.get_incremental_since')
    @patch('levis_web.fetch_rma_list')
    @patch('levis_web.get_local_ids_for_status')
    @patch('levis_web.delete_rmas')
    @patch('levis_web.should_refresh_detail')
    @patch('levis_web.fetch_rma_detail')
    @patch('levis_web.set_last_sync')
    @patch('levis_web._now_utc')
    def test_perform_sync_full_sync(
        self,
        mock_now_utc,
        mock_set_last_sync,
        mock_fetch_rma_detail,
        mock_should_refresh,
        mock_delete_rmas,
        mock_get_local_ids,
        mock_fetch_list,
        mock_get_since,
        mock_st
    ):
        """
        Tests the perform_sync function for a successful full sync.
        """
        print("\nRunning: test_perform_sync_full_sync")

        # --- 1. Setup Mocks ---
        mock_st.empty.return_value = MagicMock()
        mock_st.progress.return_value = MagicMock()

        # For a full sync, get_incremental_since returns None
        mock_get_since.return_value = None

        # Mock API to return two RMAs
        mock_fetch_list.return_value = ([{"rmaId": "201"}, {"rmaId": "202"}], True, None)

        # Mock should_refresh_detail to control which details are fetched.
        # Let's say 201 is old and needs a refresh, but 202 is recent.
        mock_should_refresh.side_effect = lambda rma_id: rma_id == "201"

        # --- 2. Execute Function ---
        try:
            levis_web.perform_sync(statuses=['Pending'], full=True)
        except Exception as e:
            if 'rerun' not in str(e).lower():
                self.fail(f"perform_sync raised an unexpected exception: {e}")

        # --- 3. Assertions ---

        # In a full sync, we don't delete stale RMAs based on the API list.
        print("Asserting that stale RMAs are NOT deleted on full sync...")
        mock_delete_rmas.assert_not_called()

        # It should only fetch detail for the one that `should_refresh_detail` returned True for.
        print("Asserting that only expired RMA details are fetched...")
        mock_fetch_rma_detail.assert_called_once_with('201', force=False)

        # Check that `get_local_ids_for_status` is not called during a full sync for tidying.
        mock_get_local_ids.assert_not_called()

        # Check for success toast
        print("Asserting that a success toast is shown...")
        mock_st.session_state.__setitem__.assert_any_call("show_toast", True)

        print("Test completed successfully.")


    @patch('levis_web.st')
    @patch('levis_web.get_incremental_since')
    @patch('levis_web.fetch_rma_list')
    @patch('levis_web.fetch_rma_detail')
    def test_perform_sync_api_list_fails(
        self,
        mock_fetch_detail,
        mock_fetch_list,
        mock_get_since,
        mock_st
    ):
        """
        Tests that the sync stops gracefully if the initial RMA list fetch fails.
        """
        print("\nRunning: test_perform_sync_api_list_fails")

        # --- 1. Setup Mocks ---
        mock_st.empty.return_value = MagicMock()
        mock_st.progress.return_value = MagicMock()

        # Mock a failed API call for the list
        mock_fetch_list.return_value = ([], False, "503 Service Unavailable")

        # --- 2. Execute Function ---
        levis_web.perform_sync(full=True)

        # --- 3. Assertions ---

        # Check that an error message is displayed
        print("Asserting that an error message is shown...")
        mock_st.empty.return_value.error.assert_called_once_with("ReturnGO sync failed: 503 Service Unavailable")

        # Check that it does NOT proceed to fetch details
        print("Asserting that no detail fetch is attempted...")
        mock_fetch_detail.assert_not_called()

        # Check that st.rerun is not called
        print("Asserting that the app does not rerun...")
        mock_st.rerun.assert_not_called()

        print("Test completed successfully.")


if __name__ == '__main__':
    unittest.main()