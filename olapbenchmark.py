import time
import base64
import numpy as np
import pyarrow as pa
import pyarrow.flight as flight
from adbc_driver_flightsql import dbapi

# --- Configuration ---
DB_URI = "grpc://localhost:3000"
AUTH_STR = base64.b64encode(b"admin:admin").decode("utf-8")
BASIC_AUTH_HEADER = f"Basic {AUTH_STR}"

TABLE_OLAP = "olap_sales_1gb"

# 32 bytes per row: int32(4) * 8 columns = 32 bytes
# 31,250,000 rows * 32 bytes = 1,000,000,000 bytes (1 GB)
TOTAL_ROWS = 31_250_000  
BATCH_SIZE = 1_000_000

olap_schema = pa.schema([
    ('txn_id', pa.int32()),      
    ('user_id', pa.int32()),   
    ('product_id', pa.int32()),  
    ('store_id', pa.int32()),
    ('category_id', pa.int32()),
    ('price_cents', pa.int32()),  
    ('quantity', pa.int32()),
    ('txn_time', pa.int32()) # Epoch seconds  
])

# --- Fast Data Generation using NumPy ---
def generate_olap_batch(size, start_id):
    """Generates a record batch using NumPy for extreme speed."""
    txn_ids = np.arange(start_id, start_id + size, dtype=np.int32)
    user_ids = np.random.randint(1, 1_000_000, size, dtype=np.int32)
    product_ids = np.random.randint(1, 50_000, size, dtype=np.int32)
    store_ids = np.random.randint(1, 500, size, dtype=np.int32)
    category_ids = np.random.randint(1, 100, size, dtype=np.int32)
    prices = np.random.randint(100, 100_000, size, dtype=np.int32) 
    quantities = np.random.randint(1, 20, size, dtype=np.int32)
    
    # Base timestamp around current time (seconds), spread over the last year
    base_time = int(time.time())
    time_offsets = np.random.randint(0, 31_536_000, size, dtype=np.int32)
    timestamps = base_time - time_offsets

    batch = pa.RecordBatch.from_arrays([
        pa.array(txn_ids),
        pa.array(user_ids),
        pa.array(product_ids),
        pa.array(store_ids),
        pa.array(category_ids),
        pa.array(prices),
        pa.array(quantities),
        pa.array(timestamps)
    ], schema=olap_schema)
    return batch

# --- Synchronous Execution Wrapper with AwanDB Parser ---
def execute_sync(cursor, query, ignore_errors=False):
    """Executes queries and patches AwanDB's stringified GROUP BY results."""
    try:
        cursor.execute(query)
        results = cursor.fetchall()
        
        if not results:
            return []
            
        # Workaround: Detect if AwanDB returned a single giant string blob
        if len(results) == 1 and len(results[0]) == 1 and isinstance(results[0][0], str) and '|' in results[0][0]:
            parsed_results = []
            for line in results[0][0].strip().split('\n'):
                if '|' in line:
                    # Split multiple columns safely
                    parsed_results.append(tuple(p.strip() for p in line.split('|')))
            return parsed_results
            
        return results
    except Exception as e:
        if not ignore_errors:
            print(f" [X] Query Failed: {query}\n     Error: {e}")
        return None

# --- Benchmarking Utilities ---
def record_metrics(name, latencies, query_str, results_table):
    if not latencies:
        print(f" [!] {name} Failed to collect metrics.")
        return
        
    latencies.sort()
    count = len(latencies)
    p50 = latencies[int(count * 0.50)]
    p90 = latencies[int(count * 0.90)]
    p99 = latencies[int(count * 0.99)]
    p99_9 = latencies[int(count * 0.999) if count > 1000 else -1]
    avg = sum(latencies) / count
    max_val = latencies[-1]
    
    results_table.append({
        'name': name,
        'sql': query_str,
        'iters': count,
        'avg': avg,
        'p50': p50,
        'p90': p90,
        'p99': p99,
        'p99_9': p99_9,
        'max': max_val
    })
    print(f"   [+] Finished: {name} ({count} iterations)")

def benchmark_query(cursor, name, query, results_table, iterations=30):
    # Warmup
    for _ in range(3):
        execute_sync(cursor, query, ignore_errors=True)

    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        execute_sync(cursor, query)
        latencies.append((time.perf_counter() - start) * 1000) 
        
    record_metrics(name, latencies, query, results_table)

# --- Execution ---
def run_benchmarks():
    print("==================================================")
    print(" 🧊 AwanDB 1GB OLAP Benchmark Suite")
    print("==================================================\n")

    conn = dbapi.connect(DB_URI, db_kwargs={"adbc.flight.sql.rpc.call_header.Authorization": BASIC_AUTH_HEADER})
    cursor = conn.cursor()
    results_table = [] 

    # ---------------------------------------------------------
    # SETUP & 1GB DATA SEEDING
    # ---------------------------------------------------------
    print(f"-> [Phase 1] Generating and Seeding ~1 GB of Data ({TOTAL_ROWS:,} rows)...")
    
    execute_sync(cursor, f"DROP TABLE {TABLE_OLAP}", ignore_errors=True)

    create_sql = f"""
    CREATE TABLE {TABLE_OLAP} (
        txn_id INT, user_id INT, product_id INT, store_id INT, 
        category_id INT, price_cents INT, quantity INT, txn_time INT
    )
    """
    execute_sync(cursor, create_sql)

    client = flight.FlightClient(DB_URI)
    options = flight.FlightCallOptions(headers=[(b"authorization", BASIC_AUTH_HEADER.encode("utf-8"))])
    descriptor = flight.FlightDescriptor.for_path(TABLE_OLAP)
    writer, _ = client.do_put(descriptor, olap_schema, options=options)

    start_time = time.perf_counter()
    
    # Stream in chunks to manage memory
    for i in range(TOTAL_ROWS // BATCH_SIZE):
        batch = generate_olap_batch(BATCH_SIZE, start_id=i * BATCH_SIZE)
        writer.write_batch(batch)
        if (i + 1) % 5 == 0:
            print(f"     ... Inserted {(i + 1) * BATCH_SIZE:,} rows")
            
    writer.close()
    ingest_time = time.perf_counter() - start_time
    bytes_per_sec = (TOTAL_ROWS * 32) / ingest_time
    mb_per_sec = bytes_per_sec / (1024 * 1024)
    
    print(f"   [+] Ingestion Complete: {TOTAL_ROWS:,} rows in {ingest_time:.2f}s")
    print(f"   [+] Throughput: {TOTAL_ROWS / ingest_time:,.0f} rows/sec ({mb_per_sec:.2f} MB/s)\n")

    # ---------------------------------------------------------
    # OLAP BENCHMARKS
    # ---------------------------------------------------------
    print("-> [Phase 2] Running OLAP Workload Tail-Latencies...\n")

    benchmark_query(cursor, "Q1: Full Column Sum (31M rows)", 
                    f"SELECT SUM(price_cents) FROM {TABLE_OLAP}", results_table, iterations=50)

    benchmark_query(cursor, "Q2: High-Selectivity Filter", 
                    f"SELECT COUNT(*) FROM {TABLE_OLAP} WHERE category_id = 42", results_table, iterations=50)

    benchmark_query(cursor, "Q3: 1D Group-By Aggregation", 
                    f"SELECT store_id, SUM(quantity) FROM {TABLE_OLAP} GROUP BY store_id", results_table, iterations=30)

    benchmark_query(cursor, "Q4: Multi-Dim Group-By & Order", 
                    f"SELECT category_id, store_id, SUM(price_cents) FROM {TABLE_OLAP} GROUP BY category_id, store_id ORDER BY 3 DESC LIMIT 10", results_table, iterations=30)

    benchmark_query(cursor, "Q5: Top-K Heavy Hitters", 
                    f"SELECT product_id, SUM(quantity) as sales FROM {TABLE_OLAP} GROUP BY product_id ORDER BY sales DESC LIMIT 5", results_table, iterations=30)


    # ---------------------------------------------------------
    # FINAL METRICS
    # ---------------------------------------------------------
    print("\n\n" + "="*145)
    print(" 🏆 AWANDB 1GB OLAP METRICS 🏆".center(145))
    print("="*145)
    
    header = f"| {'Test Name':<32} | {'Iter':<4} | {'Avg(ms)':<8} | {'p50(ms)':<8} | {'p90(ms)':<8} | {'p99(ms)':<8} | {'p99.9(ms)':<9} | {'Max(ms)':<8} |"
    print(header)
    print("-" * len(header))
    
    for r in results_table:
        print(f"| {r['name']:<32} | {r['iters']:<4} | {r['avg']:>8.2f} | {r['p50']:>8.2f} | {r['p90']:>8.2f} | {r['p99']:>8.2f} | {r['p99_9']:>9.2f} | {r['max']:>8.2f} |")
        print(f"|   -> SQL: {r['sql']:<130} |")
        print("-" * len(header))
        
    print("="*145 + "\n")

    # ---------------------------------------------------------
    # TEARDOWN & CLEANUP
    # ---------------------------------------------------------
    print("-> [Phase 3] Cleaning up resources...")
    execute_sync(cursor, f"DROP TABLE {TABLE_OLAP}", ignore_errors=True)
    conn.close()
    print("   [+] Cleanup complete. Benchmark finished!\n")

if __name__ == "__main__":
    run_benchmarks()