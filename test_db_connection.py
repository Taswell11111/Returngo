import psycopg2
import sys

try:
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="Taswell11!",
        host="34.11.167.191",
        port="5432",
        sslmode="require"
    )
    print("Connection successful!")
    conn.close()
except psycopg2.OperationalError as e:
    print(f"Connection failed: {e}", file=sys.stderr)
    sys.exit(1)
