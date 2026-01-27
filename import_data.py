import pandas as pd
import toml
from sqlalchemy import create_engine, text

def get_db_url():
    secrets = toml.load(r"c:\Users\Taswell\OneDrive\Documents\GitHub\Returngo\.streamlit\secrets.toml")
    conn = secrets["connections"]["postgresql"]
    return f"{conn['dialect']}+{conn['driver']}://{conn['username']}:{conn['password']}@{conn['host']}:{conn['port']}/{conn['database']}?sslmode={conn['sslmode']}"

def create_table(engine):
    create_table_query = text("""
    CREATE TABLE IF NOT EXISTS rmas (
        rma_id TEXT PRIMARY KEY,
        store_url TEXT,
        status TEXT,
        created_at TIMESTAMP WITH TIME ZONE,
        json_data JSONB,
        last_fetched TIMESTAMP WITH TIME ZONE,
        courier_status TEXT,
        courier_last_checked TIMESTAMP WITH TIME ZONE,
        received_first_seen TIMESTAMP WITH TIME ZONE
    );
    """)
    try:
        with engine.connect() as connection:
            connection.execute(create_table_query)
            connection.commit()
        print("Table 'rmas' is ready.")
    except Exception as e:
        print(f"Error creating table: {e}")
        raise

def import_csv_to_db(engine, filepath):
    try:
        df = pd.read_csv(filepath)
        
        for col in ['created_at', 'last_fetched', 'courier_last_checked', 'received_first_seen']:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        
        print(f"Loaded {len(df)} records from CSV.")
        
        with engine.connect() as connection:
            df.to_sql('temp_rmas', connection, if_exists='replace', index=False)
            
            insert_query = text("""
            INSERT INTO rmas (rma_id, store_url, status, created_at, json_data, last_fetched, courier_status, courier_last_checked, received_first_seen)
            SELECT rma_id, store_url, status, created_at, CAST(json_data AS JSONB), last_fetched, courier_status, courier_last_checked, received_first_seen
            FROM temp_rmas
            ON CONFLICT (rma_id) DO UPDATE SET
                store_url = EXCLUDED.store_url,
                status = EXCLUDED.status,
                created_at = EXCLUDED.created_at,
                json_data = EXCLUDED.json_data,
                last_fetched = EXCLUDED.last_fetched,
                courier_status = EXCLUDED.courier_status,
                courier_last_checked = EXCLUDED.courier_last_checked,
                received_first_seen = EXCLUDED.received_first_seen;
            """)
            connection.execute(insert_query)
            connection.commit()
            
        print("Successfully imported data into 'rmas' table.")

    except Exception as e:
        print(f"Error importing data: {e}")
        raise

def main():
    print("Importing Levi's Web DB to PostgreSQL")

    try:
        db_url = get_db_url()
        engine = create_engine(db_url)
        print("Successfully connected to the PostgreSQL database.")
    except Exception as e:
        print(f"Failed to connect to PostgreSQL: {e}")
        return

    create_table(engine)

    csv_path = "download_levis-web_db.csv"
    import_csv_to_db(engine, csv_path)

if __name__ == "__main__":
    main()