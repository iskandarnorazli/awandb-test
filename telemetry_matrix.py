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

TABLE_MATRIX = "telemetry_matrix"
TABLE_DIM = "telemetry_dim"

# The test matrix: Scaling through L1/L2, L3, and Main RAM boundaries
DATA_SIZES = [10_000, 100_000, 1_000_000, 10_000_000, 31_250_000]
BATCH_SIZE = 1_000_000 # Max rows per Arrow Flight batch to prevent Client OOM

# Expanded Schema for Multi-Col, GroupBy, and Join testing
fact_schema = pa.schema([
    ('id', pa.int32()),      
    ('group_id', pa.int32()),   
    ('val1', pa.int32()),
    ('val2', pa.int32())
])

dim_schema = pa.schema([
    ('group_id', pa.int32()),
    ('multiplier', pa.int32())
])

def generate_fact_batch(size, start_id=0):
    ids = np.arange(start_id, start_id + size, dtype=np.int32)
    groups = np.random.randint(1, 100, size, dtype=np.int32) # 100 Distinct Groups
    val1 = np.random.randint(0, 1000, size, dtype=np.int32)
    val2 = np.random.randint(0, 1000, size, dtype=np.int32)
    return pa.RecordBatch.from_arrays([
        pa.array(ids), 
        pa.array(groups), 
        pa.array(val1), 
        pa.array(val2)
    ], schema=fact_schema)

def generate_dim_batch():
    groups = np.arange(1, 101, dtype=np.int32) # 1 to 100
    multipliers = np.random.randint(1, 10, 100, dtype=np.int32)
    return pa.RecordBatch.from_arrays([pa.array(groups), pa.array(multipliers)], schema=dim_schema)

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

def run_matrix():
    print("=====================================================")
    print(" 📈 AwanDB Dynamic Telemetry & Cache Boundary Matrix")
    print("=====================================================\n")

    conn = dbapi.connect(DB_URI, db_kwargs={"adbc.flight.sql.rpc.call_header.Authorization": BASIC_AUTH_HEADER})
    cursor = conn.cursor()
    
    # Expanded Metric Tracking
    metrics = {
        "Arrow DoPut Ingestion": [],
        "Clean Point Lookup (O(1) PK)": [],
        "Clean Range Scan (1D Predicate)": [],
        "Clean Range Scan (Multi-Col AND)": [],
        "Clean Aggregation (Full Scan)": [],
        "Clean Group By (Hash Map Build)": [],
        "Clean Hash Join (Fact x Dim)": [],
        "Dirty Point Lookup (O(1) PK)": [],
        "Dirty Range Scan (1D Predicate)": [],
        "Dirty Aggregation (Full Scan)": [],
        "Dirty Hash Join (Fact x Dim)": []
    }

    client = flight.FlightClient(DB_URI)
    options = flight.FlightCallOptions(headers=[(b"authorization", BASIC_AUTH_HEADER.encode("utf-8"))])

    for size in DATA_SIZES:
        # Dynamic Iteration Scaling to save time on massive datasets
        if size >= 31_250_000:
            iters = 5
        elif size >= 10_000_000:
            iters = 10
        else:
            iters = 50

        print(f"-> [Seeding & Ingestion Test] Measuring {size:,} rows (Iters: {iters})...")
        
        # --- 0. Seed Dimension Table for Joins ---
        execute_sync(cursor, f"DROP TABLE {TABLE_DIM}", ignore_errors=True)
        execute_sync(cursor, f"CREATE TABLE {TABLE_DIM} (group_id INT, multiplier INT)")
        dim_descriptor = flight.FlightDescriptor.for_path(TABLE_DIM)
        dim_writer, _ = client.do_put(dim_descriptor, dim_schema, options=options)
        dim_writer.write_batch(generate_dim_batch())
        dim_writer.close()

        # --- 1. Measure Raw DoPut Ingestion Latency (Fact Table) ---
        ingest_latencies = []
        ingest_iters = 5 if size <= 1_000_000 else 2
        
        for _ in range(ingest_iters):
            execute_sync(cursor, f"DROP TABLE {TABLE_MATRIX}", ignore_errors=True)
            execute_sync(cursor, f"CREATE TABLE {TABLE_MATRIX} (id INT, group_id INT, val1 INT, val2 INT)")
            
            descriptor = flight.FlightDescriptor.for_path(TABLE_MATRIX)
            writer, _ = client.do_put(descriptor, fact_schema, options=options)
            
            start = time.perf_counter()
            chunks = max(1, size // BATCH_SIZE)
            remainder = size % BATCH_SIZE
            
            for i in range(chunks):
                chunk_size = BATCH_SIZE if (i < chunks - 1 or remainder == 0) else BATCH_SIZE
                writer.write_batch(generate_fact_batch(chunk_size, i * BATCH_SIZE))
            
            if remainder > 0 and chunks > 0:
                writer.write_batch(generate_fact_batch(remainder, chunks * BATCH_SIZE))
            elif chunks == 0:
                writer.write_batch(generate_fact_batch(size, 0))
                
            writer.close() 
            ingest_latencies.append((time.perf_counter() - start) * 1000)
            
        metrics["Arrow DoPut Ingestion"].append({
            'p50': np.percentile(ingest_latencies, 50),
            'p90': np.percentile(ingest_latencies, 90),
            'p99': np.percentile(ingest_latencies, 99)
        })

        # --- 2. CLEAN STATE: Run Query Matrix ---
        print(f"   Running Clean topology matrix for {size:,} rows...")
        
        target_id = np.random.randint(0, size)
        metrics["Clean Point Lookup (O(1) PK)"].append(
            measure_latency(cursor, f"SELECT * FROM {TABLE_MATRIX} WHERE id = {target_id}", iters))

        metrics["Clean Range Scan (1D Predicate)"].append(
            measure_latency(cursor, f"SELECT COUNT(*) FROM {TABLE_MATRIX} WHERE val1 > 500", iters))

        metrics["Clean Range Scan (Multi-Col AND)"].append(
            measure_latency(cursor, f"SELECT COUNT(*) FROM {TABLE_MATRIX} WHERE val1 > 500 AND val2 < 500", iters))

        metrics["Clean Aggregation (Full Scan)"].append(
            measure_latency(cursor, f"SELECT SUM(val1) FROM {TABLE_MATRIX}", iters))

        metrics["Clean Group By (Hash Map Build)"].append(
            measure_latency(cursor, f"SELECT group_id, SUM(val1) FROM {TABLE_MATRIX} GROUP BY group_id", iters))

        metrics["Clean Hash Join (Fact x Dim)"].append(
            measure_latency(cursor, f"SELECT * FROM {TABLE_MATRIX} JOIN {TABLE_DIM} ON {TABLE_MATRIX}.group_id = {TABLE_DIM}.group_id", iters))

        # --- 3. INJECT DIRTY STATE (Tombstones & RAM Deltas) ---
        print("   -> Injecting Tombstones and Delta Updates (Dirtying Table)...")
        # Delete first 10% of rows (Creates Disk Tombstones)
        del_limit = size // 10
        execute_sync(cursor, f"DELETE FROM {TABLE_MATRIX} WHERE id < {del_limit}")
        
        # Update last 10% of rows (Moves Disk records to RAM Delta Buffer)
        upd_limit = size - (size // 10)
        execute_sync(cursor, f"UPDATE {TABLE_MATRIX} SET val1 = 8888 WHERE id > {upd_limit}")

        # --- 4. DIRTY STATE: Run Query Matrix ---
        print(f"   Running Dirty topology matrix for {size:,} rows...")
        
        dirty_target_id = np.random.randint(0, size) # Random ID, might be deleted, might be updated
        metrics["Dirty Point Lookup (O(1) PK)"].append(
            measure_latency(cursor, f"SELECT * FROM {TABLE_MATRIX} WHERE id = {dirty_target_id}", iters))

        metrics["Dirty Range Scan (1D Predicate)"].append(
            measure_latency(cursor, f"SELECT COUNT(*) FROM {TABLE_MATRIX} WHERE val1 > 500", iters))

        metrics["Dirty Aggregation (Full Scan)"].append(
            measure_latency(cursor, f"SELECT SUM(val1) FROM {TABLE_MATRIX}", iters))

        metrics["Dirty Hash Join (Fact x Dim)"].append(
            measure_latency(cursor, f"SELECT * FROM {TABLE_MATRIX} JOIN {TABLE_DIM} ON {TABLE_MATRIX}.group_id = {TABLE_DIM}.group_id", iters))


    conn.close()

    # ---------------------------------------------------------
    # CALCULATE EXECUTION EQUATIONS (Linear Regression)
    # ---------------------------------------------------------
    print("\n=====================================================")
    print(" 🧮 BIG-O EXECUTION EQUATIONS (E2E Latency)")
    print("=====================================================")
    print(" Equation Format: Total_Time = (Cost_Per_Row * N) + Base_Overhead\n")

    x_rows = np.array(DATA_SIZES)
    size_labels = [f"{s//1000}K" if s < 1_000_000 else f"{s/1_000_000:g}M" for s in DATA_SIZES]

    for q_type, y_latencies in metrics.items():
        p50s = np.array([m['p50'] for m in y_latencies])
        p90s = np.array([m['p90'] for m in y_latencies])
        p99s = np.array([m['p99'] for m in y_latencies])
        
        slope, intercept = np.polyfit(x_rows, p50s, 1)
        
        ns_per_row = slope * 1_000_000  
        base_ms = intercept             
        
        print(f" 🔹 {q_type.upper()}")
        print(f"    p50 Latencies : " + " -> ".join([f"{p:>7.2f}ms ({l})" for p, l in zip(p50s, size_labels)]))
        print(f"    p90 Latencies : " + " -> ".join([f"{p:>7.2f}ms ({l})" for p, l in zip(p90s, size_labels)]))
        print(f"    p99 Latencies : " + " -> ".join([f"{p:>7.2f}ms ({l})" for p, l in zip(p99s, size_labels)]))
        
        if slope < 0.000001: 
            print(f"    Complexity    : O(1) Constant Time")
            print(f"    Equation(p50) : Time(N) = {base_ms:.2f} ms")
        else:
            print(f"    Complexity    : O(N) Linear Scan")
            print(f"    Equation(p50) : Time(N) = ({ns_per_row:.2f} ns * N) + {base_ms:.2f} ms")
        print()

if __name__ == "__main__":
    run_matrix()