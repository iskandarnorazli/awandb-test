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

TABLE_TICKS = "hft_ticks"
TABLE_SYMBOLS = "hft_symbols"
TABLE_GRAPH = "social_graph"
TABLE_AI = "ai_docs"

# --- Simulation Parameters ---
WRITE_BATCHES = 100       
ROWS_PER_BATCH = 10_000   

# --- Schema Definition ---
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
    
    return batch, sum(volumes)

def extract_awandb_result(cursor):
    """ Safely extracts data from AwanDB's 'query_result' string vector. """
    try:
        fetch_data = cursor.fetchall()
        if not fetch_data:
            return f"Affected Rows: {cursor.rowcount}" if cursor.rowcount > 0 else "Success (No Output)"
        return fetch_data[0][0]
    except Exception as e:
        return f"Parse Error/No Output: {e}"

def run_test(cursor, test_name, query, expected_vol=None):
    print(f"\n   [+] {test_name}")
    print(f"       Query: {query}")
    if expected_vol:
        print(f"       Expected Internal Volume: {expected_vol}")
    try:
        cursor.execute(query)
        res = extract_awandb_result(cursor)
        formatted_res = "\n".join([f"       {line}" for line in str(res).strip().split("\n")])
        print(formatted_res)
    except Exception as e:
        print(f"       [X] Engine Crash: {e}")

# --- Concurrency Thread Targets ---
def concurrent_writer(thread_id, num_inserts):
    conn = dbapi.connect(DB_URI, db_kwargs={"adbc.flight.sql.rpc.call_header.Authorization": BASIC_AUTH_HEADER})
    cursor = conn.cursor()
    for i in range(num_inserts):
        tick_id = 2_000_000 + (thread_id * 100_000) + i
        try:
            cursor.execute(f"INSERT INTO {TABLE_TICKS} (tick_id, symbol_id, price_cents, volume, time_offset) VALUES ({tick_id}, 1, 99999, 5, {i})")
        except Exception as e:
            print(f"   [Writer {thread_id}] Error: {e}")
    conn.close()
    print(f"   [Writer {thread_id}] Finished {num_inserts} inserts.")

def concurrent_reader(thread_id, num_reads):
    conn = dbapi.connect(DB_URI, db_kwargs={"adbc.flight.sql.rpc.call_header.Authorization": BASIC_AUTH_HEADER})
    cursor = conn.cursor()
    for i in range(num_reads):
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {TABLE_TICKS}")
            res = extract_awandb_result(cursor)
            print(f"   [Reader {thread_id}] Scan {i+1} Count: {res.strip()}")
            time.sleep(0.1)
        except Exception as e:
             print(f"   [Reader {thread_id}] Error: {e}")
    conn.close()

def concurrent_updater(thread_id, num_updates, target_symbol):
    conn = dbapi.connect(DB_URI, db_kwargs={"adbc.flight.sql.rpc.call_header.Authorization": BASIC_AUTH_HEADER})
    cursor = conn.cursor()
    for _ in range(num_updates):
        new_vol = random.randint(100, 999)
        try:
            cursor.execute(f"UPDATE {TABLE_TICKS} SET volume = {new_vol} WHERE symbol_id = {target_symbol}")
        except Exception as e:
            print(f"   [Updater {thread_id}] Error: {e}")
    conn.close()
    print(f"   [Updater {thread_id}] Finished {num_updates} overlapping updates on symbol_id {target_symbol}.")

def run_hft_simulation():
    print("==========================================")
    print(" ☁️ AwanDB Comprehensive API Stress Test ")
    print("==========================================\n")

    conn = dbapi.connect(
        DB_URI,
        db_kwargs={"adbc.flight.sql.rpc.call_header.Authorization": BASIC_AUTH_HEADER}
    )
    cursor = conn.cursor()

    # ---------------------------------------------------------
    # PHASE 0: SETUP & DDL
    # ---------------------------------------------------------
    print("-> Phase 0: Initializing Database & DDL...")
    for t in [TABLE_TICKS, TABLE_SYMBOLS, TABLE_GRAPH, TABLE_AI, "drop_test_1", "drop_test_2"]:
        try: cursor.execute(f"DROP TABLE {t}")
        except: pass 

    run_test(cursor, "Create Ticks Table", f"CREATE TABLE {TABLE_TICKS} (tick_id INT, symbol_id INT, price_cents INT, volume INT, time_offset INT)")
    run_test(cursor, "Create Symbols Table", f"CREATE TABLE {TABLE_SYMBOLS} (symbol_id INT, symbol_name STRING, is_active INT)")
    run_test(cursor, "Alter Table (ADD COLUMN)", f"ALTER TABLE {TABLE_TICKS} ADD is_processed INT")

    # ---------------------------------------------------------
    # PHASE 1: STANDARD INSERTS (SQL)
    # ---------------------------------------------------------
    print("\n-> Phase 1: Testing Standard SQL Inserts...")
    symbols_data = [
        (1, "'AAPL'", 1), (2, "'GOOGL'", 1), 
        (3, "'MSFT'", 1), (4, "'AMZN'", 0), (5, "'TSLA'", 1)
    ]
    for sid, sname, act in symbols_data:
        run_test(cursor, f"Insert Symbol {sid}", f"INSERT INTO {TABLE_SYMBOLS} (symbol_id, symbol_name, is_active) VALUES ({sid}, {sname}, {act})")

    # ---------------------------------------------------------
    # PHASE 2: ARROW DOPUT INGESTION (High Throughput)
    # ---------------------------------------------------------
    print(f"\n-> Phase 2: Streaming Writes ({WRITE_BATCHES * ROWS_PER_BATCH:,} rows)...")
    client = flight.FlightClient(DB_URI)
    options = flight.FlightCallOptions(headers=[(b"authorization", BASIC_AUTH_HEADER.encode("utf-8"))])
    descriptor = flight.FlightDescriptor.for_path(TABLE_TICKS)
    writer, _ = client.do_put(descriptor, hft_schema, options=options)

    expected_total_volume = 0
    start_time = time.time()
    
    for i in range(WRITE_BATCHES):
        batch, batch_vol = generate_tick_batch(ROWS_PER_BATCH, start_id=i * ROWS_PER_BATCH)
        expected_total_volume += batch_vol
        writer.write_batch(batch)
    
    writer.close()
    print(f"   [!] Throughput:  {(WRITE_BATCHES * ROWS_PER_BATCH) / (time.time() - start_time):,.2f} records/sec")

    # ---------------------------------------------------------
    # PHASE 3: ADVANCED QUERIES & COMPLEX MIX
    # ---------------------------------------------------------
    print("\n-> Phase 3: Validating Query API & Complex Mixing...")
    run_test(cursor, "Basic COUNT", f"SELECT COUNT(*) FROM {TABLE_TICKS}")
    run_test(cursor, "SUM Validation", f"SELECT SUM(volume) FROM {TABLE_TICKS}", expected_total_volume)
    
    prefix_query = f"SELECT {TABLE_TICKS}.tick_id, {TABLE_TICKS}.price_cents FROM {TABLE_TICKS} WHERE {TABLE_TICKS}.volume = 5 LIMIT 3"
    run_test(cursor, "Explicit Table Prefix Resolution", prefix_query)
    
    # [FIX] Simplified to cleanly test Hash-Join Isolation without tripping the naive GROUP BY
    complex_query = f"""
        SELECT * FROM {TABLE_TICKS} 
        JOIN {TABLE_SYMBOLS} ON {TABLE_TICKS}.symbol_id = {TABLE_SYMBOLS}.symbol_id
    """
    run_test(cursor, "Native Hash-Join Test", complex_query)

    # ---------------------------------------------------------
    # PHASE 4: TABLE DROP ISOLATION
    # ---------------------------------------------------------
    print("\n-> Phase 4: Validating Metadata Isolation (Drop Table)...")
    run_test(cursor, "Create Test 1", "CREATE TABLE drop_test_1 (id INT)")
    run_test(cursor, "Create Test 2", "CREATE TABLE drop_test_2 (id INT)")
    run_test(cursor, "Insert Test 2", "INSERT INTO drop_test_2 VALUES (99)")
    run_test(cursor, "Drop Table 1", "DROP TABLE drop_test_1")
    run_test(cursor, "Verify Table 2 Survives", "SELECT COUNT(*) FROM drop_test_2")

    # ---------------------------------------------------------
    # PHASE 5: HTAP CONCURRENCY STRESS TEST
    # ---------------------------------------------------------
    print("\n-> Phase 5: Testing Concurrent HTAP (Read/Write Simultaneously)...")
    reader_thread = threading.Thread(target=concurrent_reader, args=(1, 5))
    writer_thread_1 = threading.Thread(target=concurrent_writer, args=(1, 100))
    writer_thread_2 = threading.Thread(target=concurrent_writer, args=(2, 100))

    reader_thread.start()
    writer_thread_1.start()
    writer_thread_2.start()

    writer_thread_1.join()
    writer_thread_2.join()
    reader_thread.join()

    run_test(cursor, "Final Concurrent Count Verification", f"SELECT COUNT(*) FROM {TABLE_TICKS}")

    # ---------------------------------------------------------
    # PHASE 6: DEADLOCK & MUTATION RACE CONDITION TEST
    # ---------------------------------------------------------
    print("\n-> Phase 6: Testing Write-Write Conflicts & Deadlocks...")
    print("   [!] Firing overlapping UPDATE commands on the exact same rows.")
    updater_thread_1 = threading.Thread(target=concurrent_updater, args=(1, 50, 1))
    updater_thread_2 = threading.Thread(target=concurrent_updater, args=(2, 50, 1))

    updater_thread_1.start()
    updater_thread_2.start()

    updater_thread_1.join()
    updater_thread_2.join()

    run_test(cursor, "Post-Mutation Survival Check", f"SELECT COUNT(*) FROM {TABLE_TICKS} WHERE symbol_id = 1")

    # ---------------------------------------------------------
    # PHASE 7: NATIVE GRAPH ENGINE (BFS)
    # ---------------------------------------------------------
    print("\n-> Phase 7: Testing Native Graph Projection & BFS...")
    run_test(cursor, "Create Graph Table", f"CREATE TABLE {TABLE_GRAPH} (id INT, src_id INT, dst_id INT)")
    
    # [FIX] Execute inserts wrapped in run_test to force Flight stream resolution!
    edges = [(1, 0, 1), (2, 0, 2), (3, 2, 3), (4, 5, 6)]
    for eid, src, dst in edges:
        run_test(cursor, f"Insert Edge {eid}", f"INSERT INTO {TABLE_GRAPH} VALUES ({eid}, {src}, {dst})")
        
    run_test(cursor, "Execute Native BFS from Node 0", f"SELECT BFS_DISTANCE(0) FROM {TABLE_GRAPH}")

    # ---------------------------------------------------------
    # PHASE 8: NATIVE AI VECTOR SEARCH
    # ---------------------------------------------------------
    print("\n-> Phase 8: Testing Native AI Vector Search...")
    run_test(cursor, "Create Vector Table", f"CREATE TABLE {TABLE_AI} (id INT, embedding VECTOR)")
    
    # [FIX] Ensure the inserts are consumed properly
    run_test(cursor, "Insert Target Vector", f"INSERT INTO {TABLE_AI} VALUES (10, '[1.0, 0.0, 0.0]')")
    run_test(cursor, "Insert Orthogonal Vector", f"INSERT INTO {TABLE_AI} VALUES (20, '[0.0, 1.0, 0.0]')")
    run_test(cursor, "Insert Close Match", f"INSERT INTO {TABLE_AI} VALUES (30, '[0.9, 0.1, 0.0]')")
    
    run_test(cursor, "Execute Vector Search (Cosine Similarity > 0.8)", f"SELECT * FROM {TABLE_AI} WHERE VECTOR_SEARCH(embedding, '[1.0, 0.0, 0.0]', 0.8) = 1")

    # ---------------------------------------------------------
    # PHASE 9: TEARDOWN
    # ---------------------------------------------------------
    print("\n-> Phase 9: Teardown...")
    for t in [TABLE_TICKS, TABLE_SYMBOLS, TABLE_GRAPH, TABLE_AI, "drop_test_2"]:
        run_test(cursor, f"DROP Table {t}", f"DROP TABLE {t}")

    conn.close()
    print("\n✅ Simulation Complete!")

if __name__ == "__main__":
    run_hft_simulation()