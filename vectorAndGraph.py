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

TABLE_GRAPH = "telemetry_graph"
TABLE_VECTOR = "telemetry_vectors"

# Test matrix. 
# Graph can scale higher because it uses binary DoPut (IntVectors).
# Vector sizes are kept smaller because we must ingest them via SQL strings.
GRAPH_SIZES = [10_000, 100_000, 1_000_000]
VECTOR_SIZES = [1_000, 5_000, 10_000] 
BATCH_SIZE = 1_000_000

graph_schema = pa.schema([
    ('src_id', pa.int32()),      
    ('dst_id', pa.int32())   
])

def generate_graph_batch(size):
    # Create an interconnected graph by limiting the node space to size/10
    node_space = max(1, size // 10)
    src = np.random.randint(0, node_space, size, dtype=np.int32)
    dst = np.random.randint(0, node_space, size, dtype=np.int32)
    return pa.RecordBatch.from_arrays([pa.array(src), pa.array(dst)], schema=graph_schema)

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

def print_equations(title, metrics, sizes):
    print(f"\n=====================================================")
    print(f" 🧮 BIG-O EXECUTION EQUATIONS: {title}")
    print(f"=====================================================")
    x_rows = np.array(sizes)
    size_labels = [f"{s//1000}K" if s < 1_000_000 else f"{s/1_000_000:g}M" for s in sizes]

    for q_type, y_latencies in metrics.items():
        p50s = np.array([m['p50'] for m in y_latencies])
        slope, intercept = np.polyfit(x_rows, p50s, 1)
        ns_per_row = slope * 1_000_000  
        base_ms = intercept             
        
        print(f" 🔹 {q_type.upper()}")
        print(f"    p50 Latencies : " + " -> ".join([f"{p:>7.2f}ms ({l})" for p, l in zip(p50s, size_labels)]))
        if slope < 0.000001: 
            print(f"    Complexity    : O(1) Constant Time")
            print(f"    Equation(p50) : Time(N) = {base_ms:.2f} ms\n")
        else:
            print(f"    Complexity    : O(N) Linear / Graph Traversal")
            print(f"    Equation(p50) : Time(N) = ({ns_per_row:.2f} ns * N) + {base_ms:.2f} ms\n")

def run_matrix():
    print("=====================================================")
    print(" 🚀 AwanDB Graph & Vector Benchmark Matrix")
    print("=====================================================\n")

    conn = dbapi.connect(DB_URI, db_kwargs={"adbc.flight.sql.rpc.call_header.Authorization": BASIC_AUTH_HEADER})
    cursor = conn.cursor()
    client = flight.FlightClient(DB_URI)
    options = flight.FlightCallOptions(headers=[(b"authorization", BASIC_AUTH_HEADER.encode("utf-8"))])

    graph_metrics = {"BFS Full Graph Traversal (Node 0)": []}
    vector_metrics = {"AVX Vector Search (Threshold 0.5)": []}

    # ==========================================
    # 1. GRAPH BFS TESTING (Using Arrow DoPut)
    # ==========================================
    print("--- [ PHASE 1: GRAPH BFS ENGINE ] ---")
    for size in GRAPH_SIZES:
        iters = 20 if size < 1_000_000 else 5
        print(f" -> Seeding Graph Table with {size:,} edges...")
        
        execute_sync(cursor, f"DROP TABLE {TABLE_GRAPH}", ignore_errors=True)
        execute_sync(cursor, f"CREATE TABLE {TABLE_GRAPH} (src_id INT, dst_id INT)")
        
        # Ingest via DoPut
        descriptor = flight.FlightDescriptor.for_path(TABLE_GRAPH)
        writer, _ = client.do_put(descriptor, graph_schema, options=options)
        writer.write_batch(generate_graph_batch(size))
        writer.close()

        print(f"    Running BFS Distances from Node 0...")
        graph_metrics["BFS Full Graph Traversal (Node 0)"].append(
            measure_latency(cursor, f"SELECT BFS_DISTANCE(0) FROM {TABLE_GRAPH}", iters)
        )

    # ==========================================
    # 2. VECTOR SEARCH TESTING (Using SQL Inserts)
    # ==========================================
    print("\n--- [ PHASE 2: VECTOR SEARCH ENGINE ] ---")
    for size in VECTOR_SIZES:
        iters = 10
        print(f" -> Seeding Vector Table with {size:,} vectors (via SQL)...")
        
        execute_sync(cursor, f"DROP TABLE {TABLE_VECTOR}", ignore_errors=True)
        # The parser explicitly looks for "VECTOR" in the data type string
        execute_sync(cursor, f"CREATE TABLE {TABLE_VECTOR} (id INT, embedding VECTOR)")
        
        # Insert vectors row-by-row since DoPut doesn't support strings/vectors yet
        for i in range(size):
            vec = np.random.rand(3).tolist() 
            vec_str = "[" + ", ".join(f"{v:.3f}" for v in vec) + "]"
            execute_sync(cursor, f"INSERT INTO {TABLE_VECTOR} (id, embedding) VALUES ({i}, '{vec_str}')")

        print(f"    Running AVX Vector Similarity Search...")
        # The parser expects: VECTOR_SEARCH(col, target_vec, threshold) = 1
        target = "[0.5, 0.5, 0.5]"
        vector_metrics["AVX Vector Search (Threshold 0.5)"].append(
            measure_latency(cursor, f"SELECT * FROM {TABLE_VECTOR} WHERE VECTOR_SEARCH(embedding, '{target}', 0.5) = 1", iters)
        )

    conn.close()

    # --- Print Results ---
    print_equations("GRAPH ENGINE", graph_metrics, GRAPH_SIZES)
    print_equations("VECTOR ENGINE", vector_metrics, VECTOR_SIZES)

if __name__ == "__main__":
    run_matrix()