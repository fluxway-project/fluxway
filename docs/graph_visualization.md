# Computation graph visualization

[Chinese edition](graph_visualization_zh.md)

Fluxway exports a compiled graph for Google Model Explorer. Export does not run
the graph, open a browser, or start a server. It writes a single
`model_explorer.json` file; this file is for inspection, not job reconstruction.

## Install and export a complete example

Install the optional viewer dependency:

```bash
python -m pip install "fluxway[visualization]"
```

Run this example from the repository root. The output directory
`tmp/graph-example` must not already exist. It builds one tensor dependency and
one ordering-only dependency, then checks both exported views. No Ray cluster
or GPU is required.

```python
import json
from pathlib import Path

import numpy as np
import fluxway as fw


class Arithmetic:
    def produce(self) -> dict[str, np.ndarray]:
        return {"values": np.array([1, 2, 3])}

    def consume(self, values: np.ndarray) -> None:
        if values.sum() != 6:
            raise ValueError("unexpected input")

    def finish(self) -> None:
        return None


replicated = {"type": "static", "axis": []}
ArithmeticSPMD = fw.bind_spmd(
    Arithmetic,
    resource_profile=fw.ResourceProfile(base={"CPU": 1}),
    mesh_axis_names=("data",),
    mesh_axis_shapes=(1,),
    produce={"output_sharding": {"values": replicated}},
    consume={"input_sharding": {"values": replicated}},
    finish={},
)
builder = fw.Builder()
builder.add_node("source", ArithmeticSPMD)
builder.add_node("target", ArithmeticSPMD)
builder.add_func_edge(
    "source",
    "produce",
    "target",
    "consume",
    tensor_map={"values": "values"},
)
builder.add_func_edge(
    "target",
    "consume",
    "target",
    "finish",
    local_only=True,
)
graph = builder.build("arithmetic")
bundle = fw.visualize_graph(graph, Path("tmp/graph-example"))

projection = json.loads((bundle / "model_explorer.json").read_text())
assert [view["id"] for view in projection["graphs"]] == [
    "Dependencies",
    "Potential overlap",
]
assert len(projection["graphs"][0]["nodes"]) == 3
assert [path.name for path in bundle.iterdir()] == ["model_explorer.json"]
print(bundle.resolve())
```

`visualize_graph` reads the graph's descriptions and its existing
`CompiledGraph`: operators, ports, dependencies, transfer codes, logical tensor
IDs, callable identities, resource declarations, and `AsyncPolicy`. It does not
reconstruct these facts from the JSON. Runtime transport still uses native
Python objects.

An existing directory raises `FileExistsError`. To replace a previous Fluxway
export, explicitly pass `overwrite=True`. Fluxway first validates that directory
as an export bundle, then replaces it atomically. If the directory swap fails,
the previous export is restored and the error propagates.

## View locally or through SSH

`view_graph(bundle)` starts the Model Explorer server and blocks until stopped.
For a headless machine, use the function below with an existing bundle directory.
It keeps the server on loopback and does not open a browser.

```python
from pathlib import Path

from fluxway import view_graph


def serve_graph(bundle_directory: str | Path) -> None:
    view_graph(
        bundle_directory,
        host="127.0.0.1",
        port=8080,
        open_browser=False,
    )
```

Call `serve_graph("tmp/graph-example")` on the machine that holds the bundle.
On your own machine, pass your SSH destination as the argument to this function:

```bash
forward_graph() {
    ssh -N -L 8080:127.0.0.1:8080 "$1"
}
```

Keep that terminal open and visit `http://127.0.0.1:8080`. Both ends must have
port 8080 available. To view the export without keeping the remote server
running, use the following function. Its arguments are the SSH destination,
absolute remote bundle directory, and local JSON output path. The local machine
must have Model Explorer installed.

```bash
copy_and_view_graph() {
    scp "$1:$2/model_explorer.json" "$3" &&
        model-explorer "$3"
}
```

## Read the two views

| View | What it shows |
| --- | --- |
| `Dependencies` | One edge per compiled dependency, with all tensor transfers listed in its metadata |
| `Potential overlap` | Same-step causal layout, grouped by dependency rank and schedule lane |

Both views come from the same compiled graph. An ordering-only dependency has
zero tensor transfers; runtime transport represents it with a header-only frame.
In the example, `produce` to `consume` carries `values`, while `consume` to
`finish` only establishes order.

In `Potential overlap`, faint native edges show same-step causal pairs. Teal
`#007B83` overlays show nonzero-step phase constraints from `AsyncPolicy`.
Cross-step constraints do not alter the same-step dependency rank. The
`Dependencies` view has no overlays.

Box colors describe structural roles: green for a source, blue for a transform,
amber for a sink, purple for a dependency-cycle member, and gray for an isolated
operator. Model Explorer does not support dashed or dotted edge strokes.

## Inspect policies and correlate traces

SPMDs are namespaces and operators are nodes. Select a node to inspect its mesh,
resource range, function, ports, dependency rank, and schedule policy. Model
Explorer may collapse a namespace with one operator; enable its option to retain
single-child layers when you need every SPMD visible.

| Fields | Purpose |
| --- | --- |
| `graph_id`, `graph_generation`, `spmd_id`, `op_id`, `function_id` | Identify the compiled graph and callable |
| `input_signature_id`, `output_signature_id`, `dependency_code` | Identify ports and dependencies |
| `async_mode`, `async_depth`, `async_commit`, `async_max_staleness` | Describe concurrency limits |
| `state_access`, `state_resource` | Describe state serialization |
| `schedule_kind`, `schedule_base_priority`, `schedule_phase`, `schedule_lane` | Describe scheduling choices |
| `schedule_phase_order`, `schedule_phase_dependencies`, `schedule_phase_priorities` | Describe phase relations |
| `schedule_lane_order`, `schedule_lane_priorities` | Describe lane ordering and priority |

These fields come from compiled `AsyncPolicy`; visualization does not maintain a
second schedule definition. Workers and exports use the shared
`compile_phase_schedule` semantics. Search the same graph, SPMD, operator, and
function IDs in [Perfetto traces](user_profiler.md) to compare rules with measured
execution.

Static views express possible overlap. See [operator execution
sequences](worker_process_operators.md) for calls, waits, and shutdown, and [profiling
sequences](user_profiler.md) for capture and file export. `visualize_graph()` writes a
static file synchronously; `view_graph()` blocks while serving it. Neither starts graph
execution.

## Interpret overlap and scale

Fluxway combines graph dependencies with same-step phase relations, collapses
causal cycles, and computes the longest dependency rank on the resulting acyclic
graph. Equal-rank operators form a group; lanes appear inside that group and
SPMDs inside each lane. Cross-step relations remain overlays.

A group means its members may become ready together. Resource contention,
conditions, step numbers, and backpressure can still serialize them. Actual
queueing, transport, and device timing must be read from a trace.

The component and rank algorithms are iterative and support exports with more
than 10,000 operators. Browser layout has its own cost; a dense graph can still
take much longer to display than to export.
