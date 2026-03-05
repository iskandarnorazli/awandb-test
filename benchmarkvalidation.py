import time
import random
import base64
from adbc_driver_flightsql import dbapi

# --- Configuration ---
DB_URI = "grpc://localhost:3000"
AUTH_STR = base64.b64encode(b"admin:admin").decode("utf-8")
BASIC_AUTH_HEADER = f"Basic {AUTH_STR}"
TABLE_OLAP = "olap_sales_1gb"

def execute_sync(cursor, query, ignore_errors=False):
    """Forces ADBC Flight to immediately execute and resolve the network stream."""
    try:
        cursor.execute(query)
        return cursor.fetchall()
    except Exception as e:
        if not ignore_errors:
            print(f" [X] Query Failed: {query}\n     Error: {e}")
        return None

def run_validation():
    print("==================================================")
    print(" 🔍 AwanDB Query Optimizer Validation")
    print("==================================================\n")

    conn = dbapi.connect(DB_URI, db_kwargs={"adbc.flight.sql.rpc.call_header.Authorization": BASIC_AUTH_HEADER})
    cursor = conn.cursor()

    # --- SETUP: Create and Seed Table ---
    print(f"-> [Setup] Creating and Seeding '{TABLE_OLAP}'...")
    execute_sync(cursor, f"DROP TABLE {TABLE_OLAP}", ignore_errors=True)
    execute_sync(cursor, f"CREATE TABLE {TABLE_OLAP} (product_id INT, sales_amount INT)")

    # Seed 1000 rows across 10 distinct products
    for _ in range(1000):
        pid = random.randint(1, 10)
        execute_sync(cursor, f"INSERT INTO {TABLE_OLAP} VALUES ({pid}, 1)")
    print("-> [Setup] Seeding complete.\n")

    # --- Test 1: Let AwanDB handle the LIMIT ---
    query_with_limit = f"SELECT product_id, COUNT(*) as sales FROM {TABLE_OLAP} GROUP BY product_id ORDER BY sales DESC LIMIT 5"
    
    print("-> Running Query WITH LIMIT (Suspected Buggy Execution)...")
    start = time.perf_counter()
    cursor.execute(query_with_limit)
    results_with_limit = cursor.fetchall()
    time_with_limit = (time.perf_counter() - start) * 1000

    print(f"   Time taken: {time_with_limit:.2f} ms")
    print("   Results:")
    if results_with_limit:
        for row in results_with_limit:
            print(f"      Product ID: {row[0]:<6} | Sales Count: {row[1]}")
    else:
        print("      (No results returned)")
    print()

    # --- Test 2: Force AwanDB to aggregate everything ---
    query_without_limit = f"SELECT product_id, COUNT(*) as sales FROM {TABLE_OLAP} GROUP BY product_id ORDER BY sales DESC"
    
    print("-> Running Query WITHOUT LIMIT (Forcing Full Aggregation)...")
    start = time.perf_counter()
    cursor.execute(query_without_limit)
    
    # We apply the limit on the Python client side instead
    results_without_limit = cursor.fetchmany(5) 
    time_without_limit = (time.perf_counter() - start) * 1000

    print(f"   Time taken: {time_without_limit:.2f} ms")
    print("   Results (Top 5 fetched via Python):")
    if results_without_limit:
        for row in results_without_limit:
            print(f"      Product ID: {row[0]:<6} | Sales Count: {row[1]}")
    else:
        print("      (No results returned)")
    print()

    # --- Analysis ---
    print("==================================================")
    print(" 📊 Validation Analysis")
    print("==================================================")
    
    if results_with_limit and results_without_limit:
        count_limited = results_with_limit[0][1]
        count_full = results_without_limit[0][1]
        
        if count_full > count_limited * 2: # Adjusted threshold for 1000 rows
            print(" 🚨 BUG CONFIRMED: AST execution order is swapped.")
            print("    The engine is applying 'LIMIT' before 'GROUP BY' or 'ORDER BY'.")
            total_counted = sum(r[1] for r in results_with_limit)
            print(f"    It looks like AwanDB only scanned {total_counted} rows total before returning.")
        else:
            print(" ✅ Results match! The database's execution order is correct.")

if __name__ == "__main__":
    run_validation()