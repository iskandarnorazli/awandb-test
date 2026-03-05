import time
import base64
import numpy as np
import pyarrow as pa
import pyarrow.flight as flight
from adbc_driver_flightsql import dbapi
from collections import deque, defaultdict

# --- Configuration ---
DB_URI = "grpc://localhost:3000"
AUTH_STR = base64.b64encode(b"admin:admin").decode("utf-8")
BASIC_AUTH_HEADER = f"Basic {AUTH_STR}"

TABLE_GRAPH = "telemetry_graph"
TABLE_VECTOR = "telemetry_vectors"

GRAPH_SIZES = [10_000, 100_000, 1_000_000]
VECTOR_SIZES = [10_000, 100_000, 1_000_000]

graph_schema = pa.schema([
    ('src_id', pa.int32()),      
    ('dst_id', pa.int32())   
])

def generate_graph_batch(size):
    node_space = max(1, size // 10)
    src = np.random.randint(0, node_space, size, dtype=np.int32)
    dst = np.random.randint(0, node_space, size, dtype=np.int32)
    return pa.RecordBatch.from_arrays([pa.array(src), pa.array(dst)], schema=graph_schema), src, dst

def execute_sync(cursor, query, ignore_errors=False):
    try:
        cursor.execute(query)
        return cursor.fetchall()
    except Exception as e:
        if not ignore_errors:
            print(f" [X] Query Failed: {query}\n     Error: {e}")
        return []

# ==========================================
# VALIDATION LOGIC (GROUND TRUTH)
# ==========================================
def validate_graph(cursor, src_arr, dst_arr):
    print(f"    [Validation] Calculating exact BFS distances in Python...")
    
    # 1. Build Python Ground Truth
    graph = defaultdict(list)
    for s, d in zip(src_arr, dst_arr):
        graph[s].append(d)
        
    expected_distances = {}
    queue = deque([(0, 0)])
    while queue:
        node, dist = queue.popleft()
        if node not in expected_distances:
            expected_distances[node] = dist
            for neighbor in graph[node]:
                if neighbor not in expected_distances:
                    queue.append((neighbor, dist + 1))
                    
    # 2. Get AwanDB Results [UPDATED: Added src_id and dst_id]
    rows = execute_sync(cursor, f"SELECT BFS_DISTANCE(0, src_id, dst_id) FROM {TABLE_GRAPH}")
    
    if not rows or not rows[0]:
        print(f"    ❌ VALIDATION FAILED: AwanDB returned empty results!")
        return

    awandb_distances = {}
    
    # Extract the giant string from the first row and column
    raw_output = rows[0][0] 
    
    # Parse the "Node | Distance" string block
    for line in raw_output.strip().split('\n'):
        if not line.strip(): 
            continue
        parts = line.split('|')
        if len(parts) == 2:
            try:
                node = int(parts[0].strip())
                dist = int(parts[1].strip())
                awandb_distances[node] = dist
            except ValueError:
                pass # Skip headers or unparseable lines
                
    # 3. Compare
    if len(awandb_distances) == 0 and len(expected_distances) > 0:
        print(f"    ❌ VALIDATION FAILED: Failed to parse AwanDB string output!")
        return
        
    mismatches = 0
    for node, expected_dist in expected_distances.items():
        if awandb_distances.get(node) != expected_dist:
            mismatches += 1
            
    if mismatches == 0 and len(awandb_distances) == len(expected_distances):
        print(f"    ✅ VALIDATION PASSED: AwanDB Graph Engine matches Python perfectly ({len(expected_distances)} connected nodes).")
    else:
        print(f"    ❌ VALIDATION FAILED: {mismatches} nodes have wrong distances or missing paths.")

def validate_vector(cursor, embeddings_np, target_list, threshold):
    print(f"    [Validation] Calculating exact AVX SIMD math in NumPy...")
    
    # 1. Build Python Ground Truth (Cosine Similarity)
    target_np = np.array(target_list, dtype=np.float32)
    dot_products = np.dot(embeddings_np, target_np)
    norms_emb = np.linalg.norm(embeddings_np, axis=1)
    norm_target = np.linalg.norm(target_np)
    
    norms_emb[norms_emb == 0] = 1e-10 # Safe division
    similarities = dot_products / (norms_emb * norm_target)
    
    # Epsilon tolerance for Float32 differences between C++ and Python
    valid_indices = np.where(similarities >= (threshold - 1e-5))[0]
    
    # [UPDATED: Rank and truncate to exactly match AwanDB's limit of 100]
    sorted_indices = valid_indices[np.argsort(-similarities[valid_indices])]
    expected_ids = set(sorted_indices[:100])
    
    # 2. Get AwanDB Results
    target_str = "[" + ", ".join(f"{v:.3f}" for v in target_list) + "]"
    query = f"SELECT * FROM {TABLE_VECTOR} WHERE VECTOR_SEARCH(embedding, '{target_str}', {threshold}) = 1"
    rows = execute_sync(cursor, query)
    
    # [FIX] Parse the string output instead of casting the entire block into a Set
    awandb_ids = set()
    if rows and rows[0] and rows[0][0]:
        raw_output = rows[0][0]
        for line in raw_output.strip().split('\n'):
            line = line.strip()
            if not line or line.startswith("Found Rows"):
                continue
            parts = line.split('|')
            if len(parts) >= 1:
                try:
                    node_id = int(parts[0].strip())
                    awandb_ids.add(node_id)
                except ValueError:
                    pass
    
    # 3. Compare
    missing = expected_ids - awandb_ids
    extra = awandb_ids - expected_ids
    
    if not missing and not extra:
        print(f"    ✅ VALIDATION PASSED: C++ Engine matched NumPy perfectly ({len(awandb_ids)} vectors found).")
    else:
        print(f"    ❌ VALIDATION FAILED!")
        print(f"       Missing from AwanDB: {len(missing)} rows")
        print(f"       Extra in AwanDB: {len(extra)} rows")

# ==========================================
# BENCHMARK RUNNER
# ==========================================
def measure_latency(cursor, query, iters):
    for _ in range(2):
        execute_sync(cursor, query, ignore_errors=True)

    latencies = []
    for _ in range(iters):
        start = time.perf_counter()
        execute_sync(cursor, query)
        latencies.append((time.perf_counter() - start) * 1000)
        
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
    print(" 🚀 AwanDB Graph & Vector Benchmark Matrix (Validated)")
    print("=====================================================\n")

    conn = dbapi.connect(DB_URI, db_kwargs={"adbc.flight.sql.rpc.call_header.Authorization": BASIC_AUTH_HEADER})
    cursor = conn.cursor()
    client = flight.FlightClient(DB_URI)
    options = flight.FlightCallOptions(headers=[(b"authorization", BASIC_AUTH_HEADER.encode("utf-8"))])

    graph_metrics = {"BFS Full Graph Traversal (Node 0)": []}
    vector_metrics = {
        "AVX Vector Search (Fixed, Thresh 0.5)": [],
        "AVX Vector Search (Fixed, Thresh 0.8)": [],
        "AVX Vector Search (Random, Thresh 0.8)": []
    }

    # ==========================================
    # 1. GRAPH BFS TESTING
    # ==========================================
    print("--- [ PHASE 1: GRAPH BFS ENGINE ] ---")
    for size in GRAPH_SIZES:
        iters = 20 if size < 1_000_000 else 5
        print(f"\n -> Seeding Graph Table with {size:,} edges...")
        
        execute_sync(cursor, f"DROP TABLE {TABLE_GRAPH}", ignore_errors=True)
        execute_sync(cursor, f"CREATE TABLE {TABLE_GRAPH} (src_id INT, dst_id INT)")
        
        batch, src_arr, dst_arr = generate_graph_batch(size)
        descriptor = flight.FlightDescriptor.for_path(TABLE_GRAPH)
        writer, _ = client.do_put(descriptor, graph_schema, options=options)
        writer.write_batch(batch)
        writer.close()

        # RUN VALIDATION BEFORE BENCHMARK
        validate_graph(cursor, src_arr, dst_arr)

        print(f"    Running latency loops...")
        # [UPDATED: Added src_id and dst_id]
        graph_metrics["BFS Full Graph Traversal (Node 0)"].append(
            measure_latency(cursor, f"SELECT BFS_DISTANCE(0, src_id, dst_id) FROM {TABLE_GRAPH}", iters)
        )

    # ==========================================
    # 2. VECTOR SEARCH TESTING
    # ==========================================
    print("\n--- [ PHASE 2: VECTOR SEARCH ENGINE ] ---")
    
    vector_schema = pa.schema([
        ('id', pa.int32()),
        ('embedding', pa.list_(pa.float32()))
    ])

    for size in VECTOR_SIZES:
        iters = 50
        print(f"\n -> Seeding Vector Table with {size:,} vectors...")
        
        execute_sync(cursor, f"DROP TABLE {TABLE_VECTOR}", ignore_errors=True)
        execute_sync(cursor, f"CREATE TABLE {TABLE_VECTOR} (id INT, embedding VECTOR)")
        
        ids = np.arange(size, dtype=np.int32)
        embeddings_np = np.random.rand(size, 3).astype(np.float32)
        
        batch = pa.RecordBatch.from_arrays([
            pa.array(ids),
            pa.array(embeddings_np.tolist(), type=pa.list_(pa.float32()))
        ], schema=vector_schema)
        
        descriptor = flight.FlightDescriptor.for_path(TABLE_VECTOR)
        writer, _ = client.do_put(descriptor, vector_schema, options=options)
        writer.write_batch(batch)
        writer.close()

        print(f"    [System] Waiting 3s for AwanDB background daemon to compact vectors to C++ Native RAM...")
        time.sleep(3)

        target_fixed = [0.5, 0.5, 0.5]
        target_str = "[0.5, 0.5, 0.5]"
        
        # VALIDATE VECTORS
        validate_vector(cursor, embeddings_np, target_fixed, 0.8)

        print(f"    Running latency loops (Fixed 0.5)...")
        vector_metrics["AVX Vector Search (Fixed, Thresh 0.5)"].append(
            measure_latency(cursor, f"SELECT * FROM {TABLE_VECTOR} WHERE VECTOR_SEARCH(embedding, '{target_str}', 0.5) = 1", iters)
        )

        print(f"    Running latency loops (Fixed 0.8)...")
        vector_metrics["AVX Vector Search (Fixed, Thresh 0.8)"].append(
            measure_latency(cursor, f"SELECT * FROM {TABLE_VECTOR} WHERE VECTOR_SEARCH(embedding, '{target_str}', 0.8) = 1", iters)
        )

        print(f"    Running latency loops (Random 0.8)...")
        random_vec = np.random.rand(3).tolist()
        random_str = "[" + ", ".join(f"{v:.3f}" for v in random_vec) + "]"
        vector_metrics["AVX Vector Search (Random, Thresh 0.8)"].append(
            measure_latency(cursor, f"SELECT * FROM {TABLE_VECTOR} WHERE VECTOR_SEARCH(embedding, '{random_str}', 0.8) = 1", iters)
        )

    conn.close()

    print_equations("GRAPH ENGINE", graph_metrics, GRAPH_SIZES)
    print_equations("VECTOR ENGINE", vector_metrics, VECTOR_SIZES)

if __name__ == "__main__":
    run_matrix()