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

# Lowered slightly to 2M to prevent Python String generators from OOMing locally
ROWS = 2_000_000
BATCH_SIZE = 500_000
VECTOR_DIM = 3

TABLE_ZC = "benchmark_zerocopy"
TABLE_NORM = "benchmark_normal"

# 1. Zero-Copy Schema (Ints + Strings + Vectors -> Triggers C++ Memory Blasting)
zc_schema = pa.schema([
    ('id', pa.int32()),      
    ('val1', pa.int32()),   
    ('val2', pa.int32()),
    ('text_col', pa.string()),
    ('embed', pa.list_(pa.float32()))
])

# 2. Normal Schema (Includes an Int64 column which forces the JVM fallback loop)
norm_schema = pa.schema([
    ('id', pa.int32()),      
    ('val1', pa.int32()),   
    ('val2', pa.int32()),
    ('text_col', pa.string()),
    ('embed', pa.list_(pa.float32())),
    ('dummy_fallback', pa.int64()) 
])

def generate_zc_batch(size, start_id):
    ids = np.arange(start_id, start_id + size, dtype=np.int32)
    val1 = np.random.randint(0, 1000, size, dtype=np.int32)
    val2 = np.random.randint(0, 1000, size, dtype=np.int32)
    
    # Generate Strings
    str_data = np.array([f"word_{i}" for i in ids], dtype=object)

    # Fast AI Vector Generation using Arrow Flat Buffers (Bypasses Python lists)
    flat_data = np.random.rand(size * VECTOR_DIM).astype(np.float32)
    offsets = np.arange(0, (size + 1) * VECTOR_DIM, VECTOR_DIM, dtype=np.int32)
    embed_array = pa.ListArray.from_arrays(pa.array(offsets), pa.array(flat_data))

    return pa.RecordBatch.from_arrays(
        [pa.array(ids), pa.array(val1), pa.array(val2), pa.array(str_data), embed_array], 
        schema=zc_schema
    )

def generate_norm_batch(size, start_id):
    batch = generate_zc_batch(size, start_id)
    arrays = batch.columns
    # Add the Int64 poison pill to force JVM loops
    arrays.append(pa.array(np.zeros(size, dtype=np.int64)))
    return pa.RecordBatch.from_arrays(arrays, schema=norm_schema)

# --- Execution Wrappers ---
def execute_ddl(cursor, query):
    try:
        cursor.execute(query)
        cursor.fetchall() 
    except Exception as e:
        print(f" [!] DDL Execution Ignored/Failed: {query} -> {e}")

def execute_normal(cursor, query):
    """Legacy Mode: Receives a giant string blob and parses it in Python"""
    cursor.execute(query)
    results = cursor.fetchall()
    if len(results) == 1 and len(results[0]) == 1 and isinstance(results[0][0], str) and '|' in results[0][0]:
        parsed = []
        for line in results[0][0].strip().split('\n'):
            if '|' in line:
                parsed.append(tuple(p.strip() for p in line.split('|')))
        return parsed
    return results

def execute_zerocopy(cursor, query):
    """Zero-Copy Mode: Receives C++ pointers directly as a PyArrow DataFrame"""
    cursor.execute(query)
    return cursor.fetch_arrow_table()

def measure(name, func, *args):
    start = time.perf_counter()
    func(*args)
    return (time.perf_counter() - start) * 1000

# ==========================================
# RUN A/B BENCHMARK
# ==========================================
def run_comparison():
    print("==================================================")
    print(" 🏎️  AwanDB: Normal Mode vs. Zero-Copy Mode")
    print(f"    Payload: {ROWS:,} Rows (Mixed Types)")
    print("==================================================\n")

    client = flight.FlightClient(DB_URI)
    options = flight.FlightCallOptions(headers=[(b"authorization", BASIC_AUTH_HEADER.encode("utf-8"))])
    
    conn_norm = dbapi.connect(DB_URI, db_kwargs={"adbc.flight.sql.rpc.call_header.Authorization": BASIC_AUTH_HEADER})
    cursor_norm = conn_norm.cursor()

    conn_zc = dbapi.connect(DB_URI, db_kwargs={
        "adbc.flight.sql.rpc.call_header.Authorization": BASIC_AUTH_HEADER,
        "adbc.flight.sql.rpc.call_header.x-awan-format": "arrow"
    })
    cursor_zc = conn_zc.cursor()

    execute_ddl(cursor_zc, f"DROP TABLE {TABLE_ZC}")
    execute_ddl(cursor_zc, f"CREATE TABLE {TABLE_ZC} (id INT, val1 INT, val2 INT, text_col STRING, embed VECTOR)")
    
    execute_ddl(cursor_norm, f"DROP TABLE {TABLE_NORM}")
    execute_ddl(cursor_norm, f"CREATE TABLE {TABLE_NORM} (id INT, val1 INT, val2 INT, text_col STRING, embed VECTOR, dummy_fallback INT)")

    # ---------------------------------------------------------
    # PHASE 1: INGRESS
    # ---------------------------------------------------------
    print("--- [ PHASE 1: INGRESS / DATA LOADING ] ---")
    
    start = time.perf_counter()
    writer_norm, _ = client.do_put(flight.FlightDescriptor.for_path(TABLE_NORM), norm_schema, options=options)
    for i in range(ROWS // BATCH_SIZE):
        writer_norm.write_batch(generate_norm_batch(BATCH_SIZE, i * BATCH_SIZE))
    writer_norm.close()
    time_norm_write = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    writer_zc, _ = client.do_put(flight.FlightDescriptor.for_path(TABLE_ZC), zc_schema, options=options)
    for i in range(ROWS // BATCH_SIZE):
        writer_zc.write_batch(generate_zc_batch(BATCH_SIZE, i * BATCH_SIZE))
    writer_zc.close()
    time_zc_write = (time.perf_counter() - start) * 1000

    print(f" 🐢 Normal Mode (JVM Loop) : {time_norm_write:,.0f} ms")
    print(f" 🚀 Zero-Copy (C++ Memcpy) : {time_zc_write:,.0f} ms")
    print(f"    -> Zero-Copy is {time_norm_write / time_zc_write:.1f}x Faster!\n")

    # ---------------------------------------------------------
    # PHASE 2: EGRESS
    # ---------------------------------------------------------
    print("--- [ PHASE 2: EGRESS / QUERYING ] ---")

    queries = [
        ("Full Scan (Ints & Strings) (100k Rows)", "SELECT val1, text_col FROM {t} LIMIT 100000"),
        ("AI Vector Scan (10k Embeddings)", "SELECT embed FROM {t} LIMIT 10000"),
        ("Pure Aggregation", "SELECT SUM(val1) FROM {t}"),
        ("Group By Aggregation", "SELECT val1, SUM(val2) FROM {t} GROUP BY val1 LIMIT 100")
    ]

    for name, q_template in queries:
        execute_normal(cursor_norm, q_template.format(t=TABLE_NORM))
        execute_zerocopy(cursor_zc, q_template.format(t=TABLE_ZC))

        t_norm = measure(name, execute_normal, cursor_norm, q_template.format(t=TABLE_NORM))
        t_zc = measure(name, execute_zerocopy, cursor_zc, q_template.format(t=TABLE_ZC))

        print(f" 🔹 {name}")
        print(f"    🐢 Normal (String Parse) : {t_norm:,.0f} ms")
        print(f"    🚀 Zero-Copy (Arrow API) : {t_zc:,.0f} ms")
        print(f"       -> Zero-Copy is {t_norm / t_zc:.1f}x Faster!\n")

    conn_norm.close()
    conn_zc.close()

if __name__ == "__main__":
    run_comparison()