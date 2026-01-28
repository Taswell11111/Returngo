import streamlit as st
import logging

# It's good practice to have a logger
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Sync Test", layout="centered")
st.title("Isolated Sync Test")

st.write(
    "This app tests the `perform_sync` function from `levis_web.py` in isolation."
)
st.info(
    "Ensure your `.streamlit/secrets.toml` is correctly configured and you have authenticated "
    "with Google Cloud (`gcloud auth application-default login`)."
)

try:
    # We import the functions here, after the initial page load message.
    # This assumes levis_web.py can be imported.
    # The database initialization inside levis_web.py will be triggered on import.
    from levis_web import perform_sync, engine

    if engine is None:
        st.error(
            "Database engine failed to initialize on import. "
            "Check the terminal logs from when you started Streamlit."
        )
        st.stop()
    else:
        st.success("Successfully imported `perform_sync` and `engine` from `levis_web.py`.")
        st.success("Database connection was likely successful during import.")

    # The button to trigger the sync
    if st.button("🔄 Run Sync Test"):
        st.write("---")
        st.subheader("Sync Execution")
        
        # Call the sync function.
        # The function itself will render its own progress indicators (spinners, messages).
        perform_sync()

        st.success("✅ Sync function finished.")
        st.balloons()

except ImportError as e:
    st.error(f"Failed to import from `levis_web.py`. Error: {e}")
    st.error(
        "Please ensure this script is in the same directory as `levis_web.py`."
    )
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
    logger.exception("An error occurred during the test script execution.")

