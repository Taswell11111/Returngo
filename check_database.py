import os
import toml
from sqlalchemy import create_engine, text, exc

def get_db_url_and_connect_args():
    """Reads secrets and constructs the database URL and connection arguments."""
    try:
        # Construct the full path to the secrets.toml file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        secrets_path = os.path.join(script_dir, ".streamlit", "secrets.toml")

        if not os.path.exists(secrets_path):
            raise FileNotFoundError(f"Secrets file not found at: {secrets_path}")

        secrets = toml.load(secrets_path)
        conn = secrets["connections"]["postgresql"]
        
        dialect = conn.get("dialect", "postgresql")
        driver = conn.get("driver")
        user = conn["username"]
        password = conn["password"]
        host = conn["host"]
        port = conn["port"]
        database = conn["database"]
        
        # Construct the database URL, including the driver if specified
        db_url = f"{dialect}{f'+{driver}' if driver else ''}://{user}:{password}@{host}:{port}/{database}"

        connect_args = {}
        if "sslmode" in conn:
             db_url += f"?sslmode={conn['sslmode']}"

        return db_url, connect_args
    except (FileNotFoundError, KeyError) as e:
        print(f"Error reading database configuration: {e}")
        return None, None

def main():
    """Connects to the PostgreSQL database and reports table counts."""
    db_url, connect_args = get_db_url_and_connect_args()

    if not db_url:
        return

    print(f"Connecting to PostgreSQL database at: {db_url.split('@')[-1]}")

    try:
        engine = create_engine(db_url, connect_args=connect_args)
        
        with engine.connect() as connection:
            print("✅ Connection successful!")
            
            # Check for rmas table
            try:
                rma_count_result = connection.execute(text("SELECT COUNT(*) FROM rmas")).scalar_one_or_none()
                print(f"   - Found 'rmas' table with {rma_count_result} rows.")
            except exc.ProgrammingError:
                print("   - 'rmas' table not found.")

            # Check for sync_logs table
            try:
                sync_logs_count_result = connection.execute(text("SELECT COUNT(*) FROM sync_logs")).scalar_one_or_none()
                print(f"   - Found 'sync_logs' table with {sync_logs_count_result} rows.")
            except exc.ProgrammingError:
                print("   - 'sync_logs' table not found.")

    except exc.OperationalError as e:
        print(f"❌ Connection failed: Could not connect to the database.")
        print(f"   Error: {e}")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()