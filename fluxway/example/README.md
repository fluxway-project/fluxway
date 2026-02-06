# Fluxway Example

This example demonstrates a comprehensive usage of the Fluxway framework with the following features:

## Features

1. **CPU-only resources**: All SPMDs use CPU only (no GPU)
2. **Data transmission simulation**: All forward/backward operations simulate data transmission only, no actual computation
3. **Rank constraints**: Single SPMD total_rank <= 6
4. **Reader Node with Backpressure**: 
   - New `add_reader()` API for creating data reading nodes
   - Active data reading from JSON files
   - Backpressure handling to prevent downstream overload
5. **Transmission combinations**:
   - Sparse -> Sparse (eager mode, Preprocessing)
   - Sparse -> Sparse (Embedding)
   - Sparse -> Dense (Model)
6. **Eager mode execution**: Preprocessing uses eager_trigger for immediate data processing
7. **Bidirectional edges**: Forward and backward data flow
8. **Complete logging**: All operations include timestamps for performance monitoring
9. **Dual-stream transmission**: Both tensor data (data plane) and state data (control plane) are transmitted

## Graph Structure

```
Reader (Reader node, 2 workers)
    |
    | (reads JSON files, backpressure-aware)
    v
Preprocessing (4 ranks, preprocessing stage, eager mode)
    |
    v
Embedding (4 ranks, embedding stage)
    |
    v
Model (4 ranks, model computation stage)
    |
    <--- (backward edge to Embedding)
```

## Node Functions

- **Reader**: Reads data from JSON files (outputs sparse)
- **Preprocessing**: Preprocesses raw data (eager mode, sparse -> sparse)
- **Embedding**: Performs embedding operations (sparse -> sparse)
- **Model**: Performs model computation (sparse -> dense)

## New Reader Node API

The example now uses the new `add_reader()` API instead of the old `DataReaderSPMD`:

```python
builder.add_reader(
    name="Reader",
    src_dir="/tmp/fluxway_example_data",
    data_type="json",
    batch_size=100,
    file_pattern="*.json",
    worker_count=2,
    resource={"nums_cpu": 1, "nums_gpu": 0, "memory": 10},
    description="Data reader node for reading JSON files",
)
```

### Reader Node Features

- **Active Data Reading**: Unlike SPMD nodes that are data-driven, Reader nodes actively read data from files
- **File Partitioning**: Files are automatically partitioned across reader workers
- **Backpressure Support**: Reader pauses when downstream SPMDs are busy
- **Multiple Data Formats**: Currently supports JSON (more formats can be added)

## Forward Edges

- Reader -> Preprocessing (eager mode)
- Preprocessing -> Embedding
- Embedding -> Model

## Backward Edges

- Model -> Embedding
- (No backward edge from Embedding to Preprocessing - eager node doesn't need backward)
- (No backward edge to Reader - it's a source node)

## Usage

### Prerequisites

- Python 3.8+
- Ray installed: `pip install ray`
- NumPy installed: `pip install numpy`

### Running the Example

You can run the example in several ways:

**Option 1: As a module (recommended)**
```bash
# From the project root directory
python -m fluxway.example.main
```

**Option 2: Direct execution**
```bash
# From the project root directory
python fluxway/example/main.py
```

**Option 3: From Python code**
```python
from fluxway.example.main import main
main()
```

**Note**: The script automatically creates example JSON data files in `/tmp/fluxway_example_data/` for testing.

### Monitoring

The example includes comprehensive logging with timestamps. You can also monitor the execution through:

1. **Console logs**: All operations are logged with timestamps
2. **Ray Dashboard**: Access at `http://localhost:8265` (default Ray dashboard port)
   - View actor states
   - Monitor resource usage
   - Check task execution times
   - View performance metrics

### Logging Format

All logs include:
- Timestamp (with microseconds)
- Log level
- Module name
- Operation details (batch_id, tensor shapes, durations)

Example log output:
```
2024-01-01 12:00:00.123456 [INFO] [fluxway.example.main] Reader node will read data from files automatically...
2024-01-01 12:00:00.123456 [INFO] [fluxway.example.main] Backpressure handling is enabled between Reader and SPMD nodes...
```

## File Structure

```
fluxway/example/
├── __init__.py          # Package initialization
├── data_types.py        # Data type definitions (with state and tensor fields)
├── spmd_modules.py      # SPMD module implementations
├── main.py              # Main entry point and graph execution
└── README.md            # This file
```

## Data Source Implementation

The Reader node uses the new datasource framework:

```
BaseSource (Abstract base class)
    |
    v
FileBaseSource (File-based reading)
    |
    v
JsonDatasource (JSON file reading)
```

Key features:
- File discovery with glob patterns
- Load-balanced file partitioning across workers
- Batch reading with configurable batch size
- In-memory size estimation for backpressure

## Backpressure Mechanism

The backpressure system works at multiple levels:

1. **Worker Level**: Each worker's mailbox reports its fullness status
2. **Driver Level**: Aggregates worker status, considers SPMD busy if >50% workers are full
3. **Controller Level**: Monitors all SPMDs and signals Reader to pause/resume
4. **Reader Level**: Pauses file reading when signaled, resumes when downstream catches up

```
Reader Worker -> Reader Driver -> Controller -> SPMD Driver -> SPMD Worker
                 (pause/resume)   (monitors)    (reports)      (mailbox status)
```

## Performance Considerations

- Reader nodes actively push data, enabling better throughput
- Backpressure prevents memory overflow in downstream workers
- File partitioning ensures balanced load across reader workers
- Logging includes duration measurements for performance analysis
- Ray dashboard provides detailed metrics for resource usage and execution times

## Notes

- This is a simulation example: no actual computation is performed
- All tensors are NumPy arrays (not PyTorch/JAX tensors) for simplicity
- The example uses generated JSON data files for testing
- Backpressure threshold is configurable (default: 50% worker fullness)
