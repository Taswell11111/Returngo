import sys
import toml
import os
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

def get_db_url():
    """Reads secrets and constructs the database URL."""
    try:
        # Construct the full path to the secrets.toml file
        # Assumes the script is run from the root of the project directory
        secrets_path = os.path.join(os.getcwd(), ".streamlit", "secrets.toml")

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

        if "sslmode" in conn:
             db_url += f"?sslmode={conn['sslmode']}"

        return db_url
    except (FileNotFoundError, KeyError) as e:
        print(f"Error reading database configuration: {e}", file=sys.stderr)
        return None

def main():
    """Tests the database connection using settings from the secrets file."""
    db_url = get_db_url()

    if not db_url:
        sys.exit(1)

    print(f"Attempting to connect to PostgreSQL database...")
    print(f"Target: {db_url.split('@')[-1]}") # Print host/db without credentials

    try:
        engine = create_engine(db_url)
        with engine.connect() as connection:
            # A simple query to confirm connectivity
            result = connection.execute(text("SELECT 1")).scalar()
            if result == 1:
                print("✅ Connection successful!")
            else:
                print("⚠️ Connection reported success, but test query failed.", file=sys.stderr)
                sys.exit(1)

    except OperationalError as e:
        print(f"❌ Connection failed: Could not connect to the database.", file=sys.stderr)
        print(f"   Please check your network settings and the credentials in '.streamlit/secrets.toml'.", file=sys.stderr)
        # To avoid printing the full error which might contain sensitive info in some cases,
        # we print a generic message. For debugging, you might want to print the full 'e'.
        # print(f"   Error details: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()