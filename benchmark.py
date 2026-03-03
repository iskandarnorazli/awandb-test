import time
import random
import base64
import threading
import pyarrow as pa
import pyarrow.flight as flight
from adbc_driver_flightsql import dbapi

# --- Configuration ---
DB_URI = "grpc://localhost:3000"
AUTH_STR = base64.b64encode(b"admin:admin").decode("utf-8")
BASIC_AUTH_HEADER = f"Basic {AUTH_STR}"

# Tables
TABLE_TICKS = "hft_ticks"
TABLE_SYMBOLS = "hft_symbols"
TABLE_GRAPH = "social_graph"
TABLE_AI = "ai_docs"
TABLE_SEQ = "sequential_test"

hft_schema = pa.schema([
    ('tick_id', pa.int32()),      
    ('symbol_id', pa.int32()),   
    ('price_cents', pa.int32()),  
    ('volume', pa.int32()),
    ('time_offset', pa.int32())   
])

def generate_tick_batch(size, start_id):
    tick_ids = [start_id + i for i in range(size)]
    symbols = [random.randint(1, 5) for _ in range(size)]
    prices = [random.randint(10000, 50000) for _ in range(size)] 
    volumes = [random.randint(1, 10) for _ in range(size)]
    timestamps = [int(time.time() % 100000) + i for i in range(size)]

    batch = pa.RecordBatch.from_arrays([
        pa.array(tick_ids, type=pa.int32()),
        pa.array(symbols, type=pa.int32()),
        pa.array(prices, type=pa.int32()),
        pa.array(volumes, type=pa.int32()),
        pa.array(timestamps, type=pa.int32())
    ], schema=hft_schema)
    return batch

# --- Synchronous Execution Wrapper ---
def execute_sync(cursor, query, ignore_errors=False):
    """Forces ADBC Flight to immediately execute and resolve the network stream."""
    try:
        cursor.execute(query)
        return cursor.fetchall()
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
    p99_9 = latencies[int(count * 0.999)]
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

def benchmark_query(cursor, name, query, results_table, iterations=100):
    for _ in range(5):
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
    print(" 🚀 AwanDB High-Performance Benchmark Suite")
    print("==================================================\n")

    conn = dbapi.connect(DB_URI, db_kwargs={"adbc.flight.sql.rpc.call_header.Authorization": BASIC_AUTH_HEADER})
    cursor = conn.cursor()
    results_table = [] # Collects all metrics for the final printout

    # ---------------------------------------------------------
    # SETUP & DATA SEEDING
    # ---------------------------------------------------------
    print("-> [Phase 1] Seeding Database (This may take a moment)...")
    
    for t in [TABLE_TICKS, TABLE_SYMBOLS, TABLE_GRAPH, TABLE_AI, TABLE_SEQ]:
        execute_sync(cursor, f"DROP TABLE {t}", ignore_errors=True)

    for q in [
        f"CREATE TABLE {TABLE_TICKS} (tick_id INT, symbol_id INT, price_cents INT, volume INT, time_offset INT)",
        f"CREATE TABLE {TABLE_SYMBOLS} (symbol_id INT, symbol_name STRING, is_active INT)",
        f"CREATE TABLE {TABLE_GRAPH} (id INT, src_id INT, dst_id INT)",
        f"CREATE TABLE {TABLE_AI} (id INT, embedding VECTOR)",
        f"CREATE TABLE {TABLE_SEQ} (id INT, val INT)" 
    ]:
        execute_sync(cursor, q)

    for sid in range(1, 6):
        execute_sync(cursor, f"INSERT INTO {TABLE_SYMBOLS} VALUES ({sid}, 'SYM_{sid}', 1)")

    for i in range(1000):
        src = random.randint(0, 999)
        dst = random.randint(0, 999)
        execute_sync(cursor, f"INSERT INTO {TABLE_GRAPH} VALUES ({i}, {src}, {dst})")

    for i in range(1000):
        vec = [round(random.random(), 3) for _ in range(10)]
        vec_str = "[" + ",".join(map(str, vec)) + "]"
        execute_sync(cursor, f"INSERT INTO {TABLE_AI} VALUES ({i}, '{vec_str}')")
    
    execute_sync(cursor, f"INSERT INTO {TABLE_AI} VALUES (99999, '[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]')")

    client = flight.FlightClient(DB_URI)
    options = flight.FlightCallOptions(headers=[(b"authorization", BASIC_AUTH_HEADER.encode("utf-8"))])
    descriptor = flight.FlightDescriptor.for_path(TABLE_TICKS)
    writer, _ = client.do_put(descriptor, hft_schema, options=options)

    start_time = time.perf_counter()
    for i in range(100):
        batch = generate_tick_batch(10000, start_id=i * 10000)
        writer.write_batch(batch)
    writer.close()
    
    ingest_time = time.perf_counter() - start_time
    print(f"   [+] Streaming Ingestion Throughput: {1_000_000 / ingest_time:,.0f} rows/sec\n")

    # ---------------------------------------------------------
    # BENCHMARKS
    # ---------------------------------------------------------
    print("-> [Phase 2] Running Tail-Latency Benchmarks...\n")

    latencies_insert = []
    for i in range(1000):
        start = time.perf_counter()
        execute_sync(cursor, f"INSERT INTO {TABLE_SEQ} VALUES ({i}, {random.randint(1, 100)})")
        latencies_insert.append((time.perf_counter() - start) * 1000)
    record_metrics("Sequential SQL INSERTs", latencies_insert, f"INSERT INTO {TABLE_SEQ} VALUES (<id>, <random_val>)", results_table)

    latencies_read = []
    for _ in range(1000):
        target_id = random.randint(0, 999)
        start = time.perf_counter()
        execute_sync(cursor, f"SELECT * FROM {TABLE_SEQ} WHERE id = {target_id}")
        latencies_read.append((time.perf_counter() - start) * 1000)
    record_metrics("Sequential SQL SELECTs", latencies_read, f"SELECT * FROM {TABLE_SEQ} WHERE id = <random_id>", results_table)

    benchmark_query(cursor, "Full-Table Aggregation (1M rows)", 
                    f"SELECT SUM(volume) FROM {TABLE_TICKS}", results_table, iterations=100)

    benchmark_query(cursor, "Predicate Filter (1M rows)", 
                    f"SELECT COUNT(*) FROM {TABLE_TICKS} WHERE price_cents > 40000", results_table, iterations=100)

    benchmark_query(cursor, "Native Hash-Join (1M x Dim)", 
                    f"SELECT * FROM {TABLE_TICKS} JOIN {TABLE_SYMBOLS} ON {TABLE_TICKS}.symbol_id = {TABLE_SYMBOLS}.symbol_id", results_table, iterations=50)

    benchmark_query(cursor, "Graph CSR Native BFS", 
                    f"SELECT BFS_DISTANCE(0) FROM {TABLE_GRAPH}", results_table, iterations=200)

    benchmark_query(cursor, "AI Vector Cosine Similarity", 
                f"SELECT * FROM {TABLE_AI} WHERE VECTOR_SEARCH(embedding, '[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]', 0.9) = 1", 
                results_table, iterations=100)

    latencies = []
    for _ in range(500):
        target_id = random.randint(0, 999_999)
        start = time.perf_counter()
        execute_sync(cursor, f"SELECT * FROM {TABLE_TICKS} WHERE tick_id = {target_id}")
        latencies.append((time.perf_counter() - start) * 1000)
    record_metrics("Point Lookups (Random ID on 1M rows)", latencies, f"SELECT * FROM {TABLE_TICKS} WHERE tick_id = <random_id>", results_table)

    # ---------------------------------------------------------
    # HTAP CONCURRENCY BENCHMARK
    # ---------------------------------------------------------
    print("\n-> [Phase 3] HTAP Concurrency Tail-Latencies (Read under Write Load)...\n")

    write_flag = {"running": True}

    def background_writer():
        bg_conn = dbapi.connect(DB_URI, db_kwargs={"adbc.flight.sql.rpc.call_header.Authorization": BASIC_AUTH_HEADER})
        bg_cursor = bg_conn.cursor()
        i = 0
        while write_flag["running"]:
            tick_id = 9_000_000 + i
            execute_sync(bg_cursor, f"INSERT INTO {TABLE_TICKS} VALUES ({tick_id}, 1, 999, 5, {i})", ignore_errors=True)
            i += 1
            time.sleep(0.001) 
        bg_conn.close()

    bg_thread = threading.Thread(target=background_writer)
    bg_thread.start()

    benchmark_query(cursor, "HTAP: Aggregation (Active Writes)", 
                    f"SELECT SUM(volume) FROM {TABLE_TICKS}", results_table, iterations=100)
    
    latencies = []
    for _ in range(500):
        target_id = random.randint(0, 999_999)
        start = time.perf_counter()
        execute_sync(cursor, f"SELECT * FROM {TABLE_TICKS} WHERE tick_id = {target_id}")
        latencies.append((time.perf_counter() - start) * 1000)
    record_metrics("HTAP: Point Lookups (Active Writes)", latencies, f"SELECT * FROM {TABLE_TICKS} WHERE tick_id = <random_id>", results_table)

    write_flag["running"] = False
    bg_thread.join()
    conn.close()

    # ---------------------------------------------------------
    # FINAL CONSOLIDATED SCREENSHOT TABLE
    # ---------------------------------------------------------
    print("\n\n" + "="*140)
    print(" 🏆 AWANDB FINAL BENCHMARK METRICS 🏆".center(140))
    print("="*140)
    
    # Header
    header = f"| {'Test Name':<38} | {'Iter':<5} | {'Avg (ms)':<8} | {'p50 (ms)':<8} | {'p90 (ms)':<8} | {'p99 (ms)':<8} | {'p99.9(ms)':<9} | {'Max (ms)':<8} |"
    print(header)
    print("-" * len(header))
    
    # Rows
    for r in results_table:
        print(f"| {r['name']:<38} | {r['iters']:<5} | {r['avg']:>8.2f} | {r['p50']:>8.2f} | {r['p90']:>8.2f} | {r['p99']:>8.2f} | {r['p99_9']:>9.2f} | {r['max']:>8.2f} |")
        print(f"|   -> SQL: {r['sql']:<124} |")
        print("-" * len(header))
        
    print("="*140 + "\n")

if __name__ == "__main__":
    run_benchmarks()