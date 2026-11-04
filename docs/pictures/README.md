# Visualization images

These images show Google Model Explorer and Perfetto with the documentation
example exported by `scripts/docs_visualization_examples.py`.

| Image | What to inspect |
| --- | --- |
| `graph_dependencies.png` | A feature-processing branch, tensor dependencies, and ordering-only training dependencies |
| `graph_overlap.png` | Dependency ranks, schedule lanes, and the cross-step phase constraint |
| `profiler_timeline.png` | Aligned rank tracks for calls, transport, mailbox waits, and scheduling |
| `profiler_scope.png` | A custom lookup scope and its readable operator, step, rank, and application attributes |

The graph is compiled and exported by Fluxway. The trace contains deterministic
illustrative timings, written by Fluxway's `PerfettoTraceWriter` using symbols
from that same compiled graph. It is not a recorded training job or a benchmark.
This synthetic timeline contains worker events only; a real job trace exported
by the controller also includes control-plane events.
The longer lookup on embedding rank 1 at step 1 illustrates how to follow an
upstream delay into a downstream wait. Durations do not establish GPU execution
time or a performance improvement.

To reproduce the inputs, install the `visualization` and `profiler` extras and
run from the repository root:

```bash
PYTHONPATH=. python scripts/docs_visualization_examples.py tmp/docs-visualization
```

The destination must not already exist. Open the resulting `graph` bundle with
`fluxway.view_graph()` and select `Dependencies` or `Potential overlap`. Expand
the namespaces and enable the cross-step overlay for the overlap image. Open
`trace.pftrace` in [Perfetto](https://ui.perfetto.dev/), expand or pin the relevant
rank tracks, and select `user.embedding_lookup` for its attributes.

The Perfetto screenshots use custom workspaces to collect the original leaf
tracks under compact SPMD/rank groups. The leaf labels retain their subsystem
and purpose. This changes the viewer layout without changing event names,
timings, or attributes.
The scope screenshot stops after `operator.function`; static port and parameter
details are available in the compiled graph rather than the compact trace.

Titles and short reading cues sit outside the viewer screenshots. Empty canvas
space is cropped for legibility; the graph edges and trace events are not
redrawn. In Model Explorer, disable overlays for the dependency image; for the
overlap image, enable `Cross-step phase` and disable the selected-node-only edge
filter, then select `trainer.forward` to display the cross-step constraint.
The overlap image focuses on dependency ranks 1 through 4; the dependency image
shows the complete pipeline.

The screenshots use Model Explorer 0.1.34 and Perfetto v59.0. The images are
shared by the English and Chinese guides; the surrounding captions explain
the selected details in each language.
