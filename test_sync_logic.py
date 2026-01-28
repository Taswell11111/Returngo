
import pytest
from unittest.mock import patch, MagicMock, ANY

# As levis_web.py runs database connections on import, we need to mock
# the dependencies BEFORE the module is imported by pytest.
# We create a fixture that will automatically run for every test in this file.
@pytest.fixture(autouse=True)
def mock_global_dependencies():
    """
    This fixture mocks dependencies that are used when the levis_web.py
    module is first imported. This prevents errors during test collection.
    """
    # Mock Streamlit's secrets management
    mock_secrets = MagicMock()
    mock_secrets.connections.postgresql.instance_connection_name = "mock-instance"
    mock_secrets.connections.postgresql.username = "mock-user"
    mock_secrets.connections.postgresql.password = "mock-pass"
    mock_secrets.connections.postgresql.database = "mock-db"
    # Mock access via st.secrets['key'] and st.secrets.get('key')
    mock_secrets.__getitem__.return_value = "mock-api-key"
    mock_secrets.get.return_value = "mock-parcel-ninja-token"

    # Mock the Google Cloud SQL Connector and SQLAlchemy's create_engine
    with patch('streamlit.secrets', mock_secrets), \
         patch('google.cloud.sql.connector.Connector', MagicMock()), \
         patch('sqlalchemy.create_engine', MagicMock()):
        # The 'yield' allows the test session to run with these mocks active.
        # After the session, the patches are automatically removed.
        yield

# Now that the global dependencies are mocked, we can safely import our script
from levis_web import perform_sync

@patch('levis_web.st')
@patch('levis_web.get_incremental_since', return_value=None)
@patch('levis_web.fetch_rma_list')
@patch('levis_web.should_refresh_detail', return_value=True)
@patch('levis_web.fetch_rma_detail')
@patch('levis_web.delete_rmas')
@patch('levis_web.get_local_ids_for_status', return_value=set())
@patch('levis_web.set_last_sync')
@patch('levis_web.load_open_rmas')
def test_perform_sync_full_run(
    mock_load_open_rmas,
    mock_set_last_sync,
    mock_get_local_ids,
    mock_delete_rmas,
    mock_fetch_rma_detail,
    mock_should_refresh,
    mock_fetch_rma_list,
    mock_get_incremental,
    mock_st,
):
    """
    GIVEN: A full sync is triggered.
    WHEN: perform_sync is called.
    THEN: It should fetch the list of RMAs, fetch details for each one,
          and save the sync timestamp.
    """
    # ARRANGE: Set up the return values for our mocked functions
    
    # 1. Mock the API call that gets the list of RMAs
    mock_fetch_rma_list.return_value = (
        [{"rmaId": "RMA001"}, {"rmaId": "RMA002"}],  # A list of 2 RMA summaries
        True,  # Indicates the API call was successful
        None   # No error message
    )
    
    # 2. Mock the API call that gets the detail for a single RMA
    mock_fetch_rma_detail.return_value = {
        "rmaId": "RMA001",
        "rmaSummary": {"status": "Pending", "createdAt": "2023-01-01T12:00:00Z"},
        "shipments": [],
        "comments": []
    }

    # ACT: Run the function we are testing
    perform_sync(full=True, rerun=False)

    # ASSERT: Check if the function behaved as expected

    # Was the initial list of RMAs fetched?
    mock_fetch_rma_list.assert_called_once()
    
    # Was the detail for each RMA fetched? (since should_refresh_detail returns True)
    assert mock_fetch_rma_detail.call_count == 2
    mock_fetch_rma_detail.assert_any_call("RMA001", force=False)
    mock_fetch_rma_detail.assert_any_call("RMA002", force=False)

    # Since this is a full sync, it tries to delete stale RMAs.
    # Was the function to get local IDs called?
    # NOTE: This part of perform_sync is commented out in the latest version.
    # If we restore it, this assertion becomes relevant.
    # mock_get_local_ids.assert_not_called() -> because the code for full=True does not call it
    
    # Was the function to save the sync time called?
    mock_set_last_sync.assert_called() 
    
    # Were Streamlit UI elements for progress updated?
    mock_st.empty.assert_called()
    mock_st.progress.assert_called()
    
    # The `success` method is called on the object returned by `st.empty()`
    status_message_mock = mock_st.empty.return_value
    status_message_mock.success.assert_called_with("✅ Sync Complete!")
    
    # Was the cache clearing function called?
    mock_load_open_rmas.clear.assert_called_once()

"""
Instructions for running this test:

1. Make sure you have pytest installed:
   pip install pytest

2. Run pytest from your terminal in the root of the project:
   pytest test_sync_logic.py

"""
