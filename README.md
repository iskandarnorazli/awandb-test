# AwanDB Python Client & Benchmark Suite

This repository contains the Python client scripts, performance benchmarks, and dynamic telemetry matrices for testing **AwanDB** via Apache Arrow Flight SQL.

## Prerequisites

Before running any scripts, you need to have the AwanDB native executable ready.

1. **Download the Executable:** Download the latest AwanDB release from the main repository.
2. **Setup the Bin Folder:** Extract the downloaded release and copy the `bin` folder directly into the root of this Python repository. Your folder structure should look like this:
```text
├── bin/
│   ├── awandb-server (Linux/macOS)
│   └── awandb-server.exe (Windows)
├── data/
├── benchmark.py
├── telemetry_matrix.py
└── README.md

```



## Installation

Install the required Python libraries for the Flight SQL client and data generation:

```bash
pip install pyarrow adbc-driver-flightsql numpy

```

## How to Run

Running the tests requires a **Two-Terminal** setup: one terminal to run the AwanDB server, and a second terminal to execute the Python scripts.

### Terminal 1: Start the AwanDB Server

** IMPORTANT WARNING REGARDING THE `-debug` FLAG:**

* You must **ONLY** use the `--debug` (or `-debug`) flag when running the `telemetry_matrix.py` script.
* **Do NOT** use the `-debug` flag when running standard benchmarks (`benchmark.py`, `olapbenchmark.py`, etc.). The profiler stdout will pollute the standard tests and may cause them to break or report inaccurate latency.

**For Standard Benchmarks (NO Debug Flag):**

```bash
# On Linux / macOS
./bin/awandb-server --port 3000 --data-dir ./data/default

# On Windows
.\bin\awandb-server.exe --port 3000 --data-dir ./data/default

```

**For Telemetry Matrix ONLY (WITH Debug Flag):**

```bash
# On Linux / macOS
./bin/awandb-server --port 3000 --data-dir ./data/default --debug

# On Windows
.\bin\awandb-server.exe --port 3000 --data-dir ./data/default --debug

```

### Terminal 2: Run the Python Scripts

Once the server is operational and listening on port 3000, open a new terminal and run your desired test:

**Run the Dynamic Telemetry Matrix:**
*(Requires the server to be running with `--debug`)*

```bash
python telemetry_matrix.py

```

This script will scale datasets from 10K up to 31.25M rows across Clean and Dirty (tombstoned) states, calculate p50/p90/p99 percentiles, and output the dynamic Big-O execution equations for the native engine.

**Run Standard Benchmarks:**
*(Requires the server to be running WITHOUT `--debug`)*

```bash
python benchmark.py
python olapbenchmark.py
python train_agent.py

```

## Available Tests

* **`telemetry_matrix.py`**: Deep-dive execution profiler. Tests O(1) PK lookups, 1D/Multi-Col range scans, Hash Aggregations, and Hash Joins across L1/L2/L3 cache boundaries.
* **`olapbenchmark.py`**: Heavy analytical workloads simulating 1GB+ OLAP queries.
* **`benchmark.py`**: Mixed HTAP workloads (High-Frequency Trading ticks, active writes while querying).
* **`train_agent.py`**: Specialized tests for AI Vector Search and Graph BFS capabilities.
* **`benchmarkvalidate.py`**: Test to validate the result for LIMIT function to not skip or cheat by returning once found the limit result instead of resolving the full query, then, return with limited values/result.