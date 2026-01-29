import sys
import toml
import os
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

def get_db_config():
    """Reads secrets and constructs the database URL and connection arguments."""
    try:
        # Construct the full path to the secrets.toml file
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

        connect_args = {}
        # Handle SSL settings based on the specified driver
        if driver == "pg8000" and conn.get("sslmode") == "require":
            connect_args["ssl_context"] = True
        elif "sslmode" in conn:
            db_url += f"?sslmode={conn['sslmode']}"

        return db_url, connect_args
    except (FileNotFoundError, KeyError) as e:
        print(f"Error reading database configuration: {e}", file=sys.stderr)
        return None, None

def main():
    """Tests the database connection using settings from the secrets file."""
    db_url, connect_args = get_db_config()

    if not db_url:
        sys.exit(1)

    print(f"Attempting to connect to PostgreSQL database...")
    print(f"Target: {db_url.split('@')[-1]}") # Print host/db without credentials

    try:
        engine = create_engine(db_url, connect_args=connect_args)
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