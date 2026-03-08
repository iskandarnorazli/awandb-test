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

TABLE_BENCH = "phase3_telemetry"

# Test matrix: Scale up to 10M rows to watch the AVX throughput
DATA_SIZES = [10_000, 100_000, 1_000_000, 10_000_000]
BATCH_SIZE = 1_000_000

# Schema matching our Phase 3 tests
bench_schema = pa.schema([
    ('id', pa.int32()),      
    ('status', pa.int32()),   # 0 or 1
    ('price', pa.int32()),
    ('category', pa.int32())
])

def generate_bench_batch(size, start_id=0, deterministic=False):
    ids = np.arange(start_id, start_id + size, dtype=np.int32)
    if deterministic:
        # Deterministic generation for Validation Suite
        status = ids % 2
        price = (ids % 1000) * 10
        category = (ids % 100) + 1
    else:
        # Random generation for Benchmark Suite
        status = np.random.randint(0, 2, size, dtype=np.int32)
        price = np.random.randint(10, 5000, size, dtype=np.int32)
        category = np.random.randint(1, 100, size, dtype=np.int32)
        
    return pa.RecordBatch.from_arrays([
        pa.array(ids), 
        pa.array(status), 
        pa.array(price), 
        pa.array(category)
    ], schema=bench_schema)

def execute_sync(cursor, query, ignore_errors=False):
    try:
        cursor.execute(query)
        return cursor.fetchall()
    except Exception as e:
        if not ignore_errors:
            print(f" [X] Query Failed: {query}\n     Error: {e}")
        return None

def measure_latency(cursor, query, iters):
    # Warmup
    for _ in range(2):
        execute_sync(cursor, query, ignore_errors=True)

    latencies = []
    for _ in range(iters):
        start = time.perf_counter()
        execute_sync(cursor, query)
        latencies.append((time.perf_counter() - start) * 1000) # ms
        
    return {
        'p50': np.percentile(latencies, 50),
        'p90': np.percentile(latencies, 90),
        'p99': np.percentile(latencies, 99)
    }

# ==========================================
# PHASE 3 VALIDATION SUITE
# ==========================================
def run_validation(cursor, client, options):
    print("--- [ PHASE 3 SYSTEM VALIDATION ] ---")
    val_size = 10_000
    
    execute_sync(cursor, f"DROP TABLE {TABLE_BENCH}", ignore_errors=True)
    execute_sync(cursor, f"CREATE TABLE {TABLE_BENCH} (id INT, status INT, price INT, category INT)")

    writer, _ = client.do_put(flight.FlightDescriptor.for_path(TABLE_BENCH), bench_schema, options=options)
    writer.write_batch(generate_bench_batch(val_size, deterministic=True))
    writer.close()

    def check(name, query, expected_substring):
        rows = execute_sync(cursor, query)
        output = rows[0][0] if rows and rows[0] else ""
        if expected_substring in str(output):
            print(f"  ✅ {name} Passed")
        else:
            print(f"  ❌ {name} Failed! Expected '{expected_substring}' but got:\n{output}")
            raise SystemExit("Validation Failed. Aborting Benchmark.")

    print(" Validating Native Math & Parsers...")
    
    # 1. Composite AVX Filtering (Status = 1 and Price >= 5000)
    # Price is 0-9990 repeating every 1000 rows. Half have status=1.
    check("Composite AVX Pushdown", 
          f"SELECT COUNT(*) FROM {TABLE_BENCH} WHERE status = 1 AND price >= 5000", 
          "2500")
    
    # 2. Recursive IN Subquery (Nested Execution)
    check("Recursive Subquery (IN)", 
          f"SELECT COUNT(*) FROM {TABLE_BENCH} WHERE category IN (SELECT category FROM {TABLE_BENCH} WHERE price = 9990)", 
          "100")

    # 3. AVX Math UPDATE (In-place execution)
    # Give everyone with status 1 a +1000 price boost.
    execute_sync(cursor, f"UPDATE {TABLE_BENCH} SET price = price + 1000 WHERE status = 1")
    check("AVX Math Update Application", 
          f"SELECT COUNT(*) FROM {TABLE_BENCH} WHERE status = 1 AND price >= 6000", 
          "2500")

    print(" 🚀 Phase 3 Native Engines validated!\n")

# ==========================================
# PHASE 3 BENCHMARK MATRIX
# ==========================================
def run_matrix():
    print("=====================================================")
    print(" 📈 AwanDB Phase 3 AVX Throughput Benchmark")
    print("=====================================================\n")

    conn = dbapi.connect(DB_URI, db_kwargs={"adbc.flight.sql.rpc.call_header.Authorization": BASIC_AUTH_HEADER})
    cursor = conn.cursor()
    client = flight.FlightClient(DB_URI)
    options = flight.FlightCallOptions(headers=[(b"authorization", BASIC_AUTH_HEADER.encode("utf-8"))])

    run_validation(cursor, client, options)

    metrics = {
        "Composite AVX Filter (AND/OR)": [],
        "Recursive Subquery (IN SELECT)": [],
        "AVX SIMD Math Update (In-Place)": []
    }

    for size in DATA_SIZES:
        iters = 10 if size >= 10_000_000 else 50
        print(f"-> Measuring {size:,} rows (Iters: {iters})...")
        
        execute_sync(cursor, f"DROP TABLE {TABLE_BENCH}", ignore_errors=True)
        execute_sync(cursor, f"CREATE TABLE {TABLE_BENCH} (id INT, status INT, price INT, category INT)")
        
        writer, _ = client.do_put(flight.FlightDescriptor.for_path(TABLE_BENCH), bench_schema, options=options)
        
        chunks = max(1, size // BATCH_SIZE)
        remainder = size % BATCH_SIZE
        for i in range(chunks):
            chunk_size = BATCH_SIZE if (i < chunks - 1 or remainder == 0) else BATCH_SIZE
            writer.write_batch(generate_bench_batch(chunk_size, i * BATCH_SIZE))
        if remainder > 0 and chunks > 0:
            writer.write_batch(generate_bench_batch(remainder, chunks * BATCH_SIZE))
        writer.close() 

        # --- RUN THE QUERIES ---
        
        # 1. Composite RPN Evaluation
        metrics["Composite AVX Filter (AND/OR)"].append(
            measure_latency(cursor, f"SELECT COUNT(*) FROM {TABLE_BENCH} WHERE status = 1 AND (price > 1500 OR category < 50)", iters))

        # 2. Recursive Subquery
        metrics["Recursive Subquery (IN SELECT)"].append(
            measure_latency(cursor, f"SELECT COUNT(*) FROM {TABLE_BENCH} WHERE category IN (SELECT category FROM {TABLE_BENCH} WHERE price > 4900)", iters))

        # 3. AVX Math Mutator (Using +1 to avoid integer overflow over 50 iterations)
        metrics["AVX SIMD Math Update (In-Place)"].append(
            measure_latency(cursor, f"UPDATE {TABLE_BENCH} SET price = price + 1 WHERE status = 1", iters))

    conn.close()

    # ---------------------------------------------------------
    # CALCULATE EXECUTION EQUATIONS
    # ---------------------------------------------------------
    print("\n=====================================================")
    print(" 🧮 BIG-O EXECUTION EQUATIONS (E2E Latency)")
    print("=====================================================")

    x_rows = np.array(DATA_SIZES)
    size_labels = [f"{s//1000}K" if s < 1_000_000 else f"{s/1_000_000:g}M" for s in DATA_SIZES]

    for q_type, y_latencies in metrics.items():
        p50s = np.array([m['p50'] for m in y_latencies])
        
        slope, intercept = np.polyfit(x_rows, p50s, 1)
        ns_per_row = slope * 1_000_000  
        base_ms = intercept             
        
        print(f" 🔹 {q_type.upper()}")
        print(f"    p50 Latencies : " + " -> ".join([f"{p:>7.2f}ms ({l})" for p, l in zip(p50s, size_labels)]))
        print(f"    Equation(p50) : Time(N) = ({ns_per_row:.2f} ns * N) + {base_ms:.2f} ms")
        print()

if __name__ == "__main__":
    run_matrix()