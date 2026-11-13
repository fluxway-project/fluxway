from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from decimal import Decimal
import json
from pathlib import Path
from typing import Any

import numpy as np

import fluxway as fw
from fluxway.common.operator_policy import AsyncPolicy, SchedulePolicy
from fluxway.profiler.graph_symbols import GraphTraceSymbols
from fluxway.profiler.perfetto_writer import PerfettoTraceWriter
from fluxway.profiler.record import (
    TraceArena,
    TraceEventKind,
    TraceMetadata,
    stable_u64,
)
from fluxway.profiler.recorder import (
    DATA_PLANE_RECEIVE_TRANSPORT_FRAME,
    MAILBOX_WAIT_STEP_READY,
    USER_CALLABLE_EXECUTE,
    USER_CUSTOM_PAYLOAD_FIELDS,
    WORKER_BLOCKED_TASK,
    TraceEventSpec,
)


REPLICATED = {"type": "static", "axis": []}
PHASES = ("forward", "backward", "update")


class Reader:
    def read(self) -> dict[str, np.ndarray]:
        return {"batch": np.ones((8, 4))}


class Features:
    def encode(self, batch: np.ndarray) -> dict[str, np.ndarray]:
        return {"dense": batch}


class Embedding:
    def lookup(self, batch: np.ndarray) -> dict[str, np.ndarray]:
        return {"sparse": batch}


class Trainer:
    def forward(self, dense: np.ndarray, sparse: np.ndarray) -> None:
        if dense.shape != sparse.shape:
            raise ValueError("feature shapes must match")

    def backward(self) -> None:
        return None

    def update(self) -> None:
        return None


class Metrics:
    def publish(self) -> None:
        return None


def phase_policy(phase: str) -> AsyncPolicy:
    return AsyncPolicy.pipelined(
        depth=2,
        schedule=SchedulePolicy.phase_graph(
            phase=phase,
            phase_order=PHASES,
            dependencies=(
                ("forward", "backward", 0),
                ("backward", "update", 0),
                ("update", "forward", 1),
            ),
            lane="optimizer" if phase == "update" else "compute",
            lane_order=("compute", "optimizer"),
        ),
    )


def build_graph() -> fw.Graph:
    builder = fw.Builder()
    declarations: tuple[tuple[str, type, dict[str, Any]], ...] = (
        ("reader", Reader, {"read": {"output_sharding": {"batch": REPLICATED}}}),
        (
            "features",
            Features,
            {
                "encode": {
                    "input_sharding": {"batch": REPLICATED},
                    "output_sharding": {"dense": REPLICATED},
                }
            },
        ),
        (
            "embedding",
            Embedding,
            {
                "lookup": {
                    "input_sharding": {"batch": REPLICATED},
                    "output_sharding": {"sparse": REPLICATED},
                }
            },
        ),
        (
            "trainer",
            Trainer,
            {
                "forward": {
                    "input_sharding": {"dense": REPLICATED, "sparse": REPLICATED},
                    "async_policy": phase_policy("forward"),
                },
                "backward": {"async_policy": phase_policy("backward")},
                "update": {"async_policy": phase_policy("update")},
            },
        ),
        ("metrics", Metrics, {"publish": {}}),
    )
    for name, cls, operators in declarations:
        bound = fw.bind_spmd(
            cls,
            resource_profile=fw.ResourceProfile(base={"CPU": Decimal(1)}),
            mesh_axis_names=("data",),
            mesh_axis_shapes=(2 if name in ("embedding", "trainer") else 1,),
            **operators,
        )
        builder.add_node(name, bound)
    for target, op in (("features", "encode"), ("embedding", "lookup")):
        builder.add_func_edge(
            "reader", "read", target, op, tensor_map={"batch": "batch"}
        )
    builder.add_func_edge(
        "features", "encode", "trainer", "forward", tensor_map={"dense": "dense"}
    )
    builder.add_func_edge(
        "embedding", "lookup", "trainer", "forward", tensor_map={"sparse": "sparse"}
    )
    builder.add_func_edge("trainer", "forward", "trainer", "backward", local_only=True)
    builder.add_func_edge("trainer", "backward", "trainer", "update", local_only=True)
    builder.add_func_edge("trainer", "update", "metrics", "publish")
    return builder.build("profile-example")


@dataclass
class ExampleTrace:
    arena: TraceArena = field(
        default_factory=lambda: TraceArena(buffer_bytes=1024 * 1024)
    )
    names: dict[int, str] = field(default_factory=dict)
    fields: dict[int, tuple[str, ...]] = field(default_factory=dict)

    def intern(self, value: str) -> int:
        ident = stable_u64("documentation-example", value)
        self.names[ident] = value
        return ident

    def add(
        self,
        owner: str,
        spec: TraceEventSpec,
        start_ms: float,
        duration_ms: float,
        **payload: str | int,
    ) -> None:
        if set(payload) != set(spec.payload_fields):
            raise ValueError(f"payload does not match {spec.name}: {payload}")
        name_id = self.intern(spec.name)
        self.fields[name_id] = spec.payload_fields
        self.arena.append(
            kind=TraceEventKind.SLICE_BEGIN,
            ts_ns=1_000_000_000 + round(start_ms * 1_000_000),
            dur_ns=round(duration_ms * 1_000_000),
            track_id=self.intern(f"{owner}/{spec.track}"),
            name_id=name_id,
            payload=tuple(
                self.intern(value) if isinstance(value, str) else value
                for value in (payload[key] for key in spec.payload_fields)
            ),
        )


def write_trace(graph: fw.Graph, output: Path) -> None:
    plan = graph.require_compiled_graph()
    trace = ExampleTrace()

    def call(
        spmd: str, rank: int, op: str, step: int, start: float, duration: float
    ) -> None:
        trace.add(
            f"worker/{spmd}/rank:{rank}",
            USER_CALLABLE_EXECUTE,
            start,
            duration,
            graph_id=plan.graph_id,
            spmd_id=spmd,
            op_id=op,
            step=step,
            execution_mode="worker_process",
        )

    custom = TraceEventSpec(
        "user.embedding_lookup",
        "compute/user_custom/worker_process",
        USER_CUSTOM_PAYLOAD_FIELDS,
        "step",
    )
    for step in range(4):
        start = step * 60.0
        call("reader", 0, "read", step, start, 12)
        trace.add(
            "worker/reader/rank:0",
            DATA_PLANE_RECEIVE_TRANSPORT_FRAME,
            start + 12,
            6,
            remote_rank=0,
            step=step,
        )
        for rank in range(2):
            lookup_duration = 28 if rank == 1 and step == 1 else 12
            call("embedding", rank, "lookup", step, start + 18, lookup_duration + 4)
            trace.add(
                f"worker/embedding/rank:{rank}",
                custom,
                start + 20,
                lookup_duration,
                op_id="lookup",
                step=step,
                rank=rank,
                execution_mode="worker_process",
                attrs_id=json.dumps(
                    {"table": "item", "count": 8192, "example": "illustrative timing"}
                ),
            )
            ready = start + 34
            forward = start + 50 if lookup_duration == 28 else ready
            trace.add(
                f"worker/trainer/rank:{rank}",
                MAILBOX_WAIT_STEP_READY,
                start + 18,
                forward - start - 20,
                op_id="forward",
                step=step,
            )
            trace.add(
                f"worker/trainer/rank:{rank}",
                WORKER_BLOCKED_TASK,
                forward - 2,
                2,
                op_id="forward",
                step=step,
            )
            call("trainer", rank, "forward", step, forward, 12)
            call("trainer", rank, "backward", step, forward + 12, 14)
            call("trainer", rank, "update", step, forward + 26, 4)

    PerfettoTraceWriter().write(
        trace.arena.snapshot(),
        output,
        metadata=TraceMetadata(trace.names, trace.fields),
        graph_symbols=GraphTraceSymbols.from_compiled_graph(plan),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=False)
    graph = build_graph()
    fw.visualize_graph(graph, args.directory / "graph")
    write_trace(graph, args.directory / "trace.pftrace")
    print(args.directory.resolve())


if __name__ == "__main__":
    main()
