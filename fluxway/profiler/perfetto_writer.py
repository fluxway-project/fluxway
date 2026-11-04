from __future__ import annotations

import heapq
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fluxway.common.compiled_graph import OperatorKey
from fluxway.profiler.errors import GraphTraceSymbolConflict, MissingGraphTraceSymbol
from fluxway.profiler.graph_symbols import (
    GraphTraceSymbols,
    OperatorTraceSymbol,
)
from fluxway.profiler.hierarchy import TraceTrackHierarchy
from fluxway.profiler.record import (
    MAX_PAYLOAD_FIELDS,
    PAYLOAD_START_COLUMN,
    TraceEventKind,
    TraceMetadata,
    stable_u64,
)


FLUXWAY_TRACE_PACKET_SEQUENCE_ID = 1

INTERNED_PAYLOAD_FIELDS = frozenset(
    {
        "checkpoint_id",
        "control_names_id",
        "error_id",
        "event_id",
        "fence_id",
        "from_state_id",
        "hook_id",
        "execution_mode",
        "graph_id",
        "method_id",
        "op_id",
        "spmd_id",
        "state_id",
        "static_names_id",
        "status_id",
        "target_spmd_id",
        "tensor_names_id",
        "to_state_id",
    }
)

type GraphAnnotations = tuple[tuple[str, str], ...]

_OPTIONAL_OPERATOR_ID_EVENTS = frozenset({"execution.phase"})


@dataclass(frozen=True, slots=True)
class _GraphAnnotationPlan:
    graph_index: int | None
    op_index: int | None
    spmd_index: int | None
    transfer_index: int | None
    logical_tuuid_index: int | None
    track_spmd_id: str | None


@dataclass(frozen=True, slots=True)
class _ResolvedTransferAnnotations:
    logical_tuuid: int
    annotations: GraphAnnotations


def _allocate_slice_lanes(records: np.ndarray) -> dict[int, int]:
    intervals_by_track: dict[int, list[tuple[int, int, int]]] = {}
    for row_index, row in enumerate(records):
        if int(row[0]) != int(TraceEventKind.SLICE_BEGIN):
            continue
        start = int(row[1])
        duration = int(row[2])
        if duration < 0:
            raise ValueError(f"trace slice duration must be non-negative: {duration}")
        track_uuid = int(row[3])
        intervals_by_track.setdefault(track_uuid, []).append(
            (start, start + duration, row_index)
        )

    assignments: dict[int, int] = {}
    for intervals in intervals_by_track.values():
        active: list[tuple[int, int]] = []
        available: list[int] = []
        lane_count = 0
        for start, end, row_index in sorted(intervals):
            while active and active[0][0] <= start:
                _, lane = heapq.heappop(active)
                heapq.heappush(available, lane)
            if available:
                lane = heapq.heappop(available)
            else:
                lane = lane_count
                lane_count += 1
            assignments[row_index] = lane
            heapq.heappush(active, (end, lane))
    return assignments


class PerfettoTraceWriter:
    def __init__(self) -> None:
        try:
            # pyrefly: ignore [missing-import]
            from perfetto.protos.perfetto.trace import perfetto_trace_pb2 as proto

            # pyrefly: ignore [missing-import]
            from perfetto.trace_builder.proto_builder import StreamingTraceProtoBuilder
        except ImportError as exc:
            raise ImportError(
                "PerfettoTraceWriter requires the optional dependency `perfetto`."
            ) from exc
        self._proto = proto
        self._builder_cls = StreamingTraceProtoBuilder

    def write(
        self,
        records: np.ndarray,
        path: str | Path,
        *,
        metadata: TraceMetadata,
        graph_symbols: GraphTraceSymbols,
    ) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        slice_lanes = _allocate_slice_lanes(records)
        graph_annotation_resolver = _GraphAnnotationResolver(
            metadata=metadata,
            graph_symbols=graph_symbols,
        )
        with output.open("wb") as f:
            builder = self._builder_cls(f)
            emitted_tracks: set[int] = set()
            emitted_descriptors: set[int] = set()
            for row_index, row in enumerate(records):
                self._write_record(
                    builder,
                    row,
                    metadata,
                    emitted_tracks,
                    emitted_descriptors,
                    graph_annotation_resolver,
                    slice_lane=slice_lanes.get(row_index),
                )

    def _write_record(
        self,
        builder: Any,
        row: np.ndarray,
        metadata: TraceMetadata,
        emitted_tracks: set[int],
        emitted_descriptors: set[int],
        graph_annotation_resolver: _GraphAnnotationResolver,
        *,
        slice_lane: int | None,
    ) -> None:
        proto = self._proto
        kind = int(row[0])
        track_uuid = int(row[3])
        name_id = int(row[4])
        if track_uuid not in emitted_tracks:
            self._write_hierarchical_track_descriptor(
                builder,
                track_uuid=track_uuid,
                track_name=metadata.name_for(track_uuid),
                emitted_descriptors=emitted_descriptors,
            )
            emitted_tracks.add(track_uuid)
        event_track_uuid = track_uuid
        if slice_lane is not None:
            event_track_uuid = stable_u64(
                "fluxway-profiler", "slice-lane", track_uuid, slice_lane
            )
            if event_track_uuid not in emitted_descriptors:
                self._write_track_descriptor(
                    builder,
                    track_uuid=event_track_uuid,
                    name=f"lane:{slice_lane}",
                    parent_uuid=track_uuid,
                    sibling_order_rank=slice_lane,
                )
                emitted_descriptors.add(event_track_uuid)
        payload_fields = metadata.payload_fields_for(name_id)
        payload_values = row[
            PAYLOAD_START_COLUMN : PAYLOAD_START_COLUMN + len(payload_fields)
        ]
        graph_annotations = graph_annotation_resolver.annotations(
            track_uuid=track_uuid,
            name_id=name_id,
            payload_values=payload_values,
        )
        if kind == int(TraceEventKind.SLICE_BEGIN):
            self._write_track_event(
                builder,
                ts_ns=int(row[1]),
                track_uuid=event_track_uuid,
                name=metadata.name_for(name_id),
                event_type=proto.TrackEvent.TYPE_SLICE_BEGIN,
                flow_id=int(row[5]),
                payload_fields=payload_fields,
                payload_values=payload_values,
                metadata=metadata,
                graph_annotations=graph_annotations,
            )
            self._write_track_event(
                builder,
                ts_ns=int(row[1]) + int(row[2]),
                track_uuid=event_track_uuid,
                name=None,
                event_type=proto.TrackEvent.TYPE_SLICE_END,
                flow_id=0,
                payload_fields=(),
                payload_values=payload_values[:0],
                metadata=metadata,
                graph_annotations=(),
            )
            return
        if kind == int(TraceEventKind.FLOW_END):
            self._write_track_event(
                builder,
                ts_ns=int(row[1]),
                track_uuid=track_uuid,
                name=metadata.name_for(name_id),
                event_type=proto.TrackEvent.TYPE_INSTANT,
                terminating_flow_id=int(row[5]),
                payload_fields=payload_fields,
                payload_values=payload_values,
                metadata=metadata,
                graph_annotations=graph_annotations,
            )
            return
        if kind == int(TraceEventKind.COUNTER):
            self._write_track_event(
                builder,
                ts_ns=int(row[1]),
                track_uuid=track_uuid,
                name=metadata.name_for(name_id),
                event_type=proto.TrackEvent.TYPE_COUNTER,
                counter_value=int(row[PAYLOAD_START_COLUMN]),
                payload_fields=payload_fields,
                payload_values=payload_values,
                metadata=metadata,
                graph_annotations=graph_annotations,
            )
            return
        self._write_track_event(
            builder,
            ts_ns=int(row[1]),
            track_uuid=track_uuid,
            name=metadata.name_for(name_id),
            event_type=proto.TrackEvent.TYPE_INSTANT,
            flow_id=int(row[5]),
            payload_fields=payload_fields,
            payload_values=payload_values,
            metadata=metadata,
            graph_annotations=graph_annotations,
        )

    def _write_track_descriptor(
        self,
        builder: Any,
        *,
        track_uuid: int,
        name: str,
        parent_uuid: int = 0,
        sibling_order_rank: int = 0,
    ) -> None:
        packet = builder.create_packet()
        packet.trusted_packet_sequence_id = FLUXWAY_TRACE_PACKET_SEQUENCE_ID
        packet.track_descriptor.uuid = track_uuid
        packet.track_descriptor.parent_uuid = parent_uuid
        packet.track_descriptor.name = name
        packet.track_descriptor.sibling_order_rank = sibling_order_rank
        builder.write_packet(packet)

    def _write_hierarchical_track_descriptor(
        self,
        builder: Any,
        *,
        track_uuid: int,
        track_name: str,
        emitted_descriptors: set[int],
    ) -> None:
        layout = TraceTrackHierarchy.from_track_name(track_name).to_layout(
            leaf_uuid=track_uuid
        )
        for node in layout.nodes:
            if node.uuid in emitted_descriptors:
                continue
            self._write_track_descriptor(
                builder,
                track_uuid=node.uuid,
                name=node.name,
                parent_uuid=node.parent_uuid,
                sibling_order_rank=node.sibling_order_rank,
            )
            emitted_descriptors.add(node.uuid)

    def _write_track_event(
        self,
        builder: Any,
        *,
        ts_ns: int,
        track_uuid: int,
        name: str | None,
        event_type: int,
        flow_id: int = 0,
        terminating_flow_id: int = 0,
        counter_value: int | None = None,
        payload_fields: tuple[str, ...],
        payload_values: np.ndarray,
        metadata: TraceMetadata,
        graph_annotations: GraphAnnotations,
    ) -> None:
        if len(payload_fields) > MAX_PAYLOAD_FIELDS:
            raise ValueError(
                f"trace payload has {len(payload_fields)} fields; max={MAX_PAYLOAD_FIELDS}"
            )
        packet = builder.create_packet()
        packet.trusted_packet_sequence_id = FLUXWAY_TRACE_PACKET_SEQUENCE_ID
        packet.timestamp = ts_ns
        event = packet.track_event
        event.track_uuid = track_uuid
        if name is not None:
            event.name = name
        event.type = event_type
        if counter_value is not None:
            event.counter_value = counter_value
        if flow_id:
            event.flow_ids.append(flow_id)
        if terminating_flow_id:
            event.terminating_flow_ids.append(terminating_flow_id)
        for name, value in zip(payload_fields, payload_values, strict=True):
            payload_value = int(value)
            if name == "attrs_id":
                self._write_user_attrs(
                    event.debug_annotations,
                    metadata.name_for(payload_value),
                )
                continue
            annotation = event.debug_annotations.add()
            annotation.name = name
            if name in INTERNED_PAYLOAD_FIELDS:
                annotation.string_value = metadata.name_for(payload_value)
            else:
                annotation.int_value = payload_value
        for annotation_name, annotation_value in graph_annotations:
            annotation = event.debug_annotations.add()
            annotation.name = annotation_name
            annotation.string_value = annotation_value
        builder.write_packet(packet)

    @staticmethod
    def _write_user_attrs(debug_annotations: Any, attrs_json: str) -> None:
        attrs = json.loads(attrs_json)
        if not isinstance(attrs, dict):
            raise ValueError("profiler attrs_id metadata must decode to a JSON object.")
        for key, value in attrs.items():
            if not isinstance(key, str):
                raise ValueError("profiler attrs JSON object keys must be strings.")
            annotation = debug_annotations.add()
            annotation.name = key
            if value is None:
                annotation.string_value = "null"
            elif isinstance(value, bool):
                annotation.bool_value = value
            elif isinstance(value, int):
                annotation.int_value = value
            elif isinstance(value, float):
                annotation.double_value = value
            elif isinstance(value, str):
                annotation.string_value = value
            else:
                raise ValueError(
                    "profiler attrs JSON values must be str, int, float, bool, or null."
                )


class _GraphAnnotationResolver:
    def __init__(
        self,
        *,
        metadata: TraceMetadata,
        graph_symbols: GraphTraceSymbols,
    ) -> None:
        self._metadata = metadata
        self._graph_symbols = graph_symbols
        self._plans: dict[tuple[int, int], _GraphAnnotationPlan] = {}
        self._track_spmd_ids: dict[int, str | None] = {}
        self._names: dict[int, str] = {}
        self._verified_graph_idents: set[int] = set()
        self._operator_annotations_by_key: dict[OperatorKey, GraphAnnotations] = {}
        self._operator_annotations_by_identity: dict[
            tuple[str, int], GraphAnnotations
        ] = {}
        self._transfer_annotations: dict[int, _ResolvedTransferAnnotations] = {}
        self._tensor_annotations: dict[int, GraphAnnotations] = {}
        self._event_annotations: dict[
            tuple[
                int,
                int,
                int | None,
                int | None,
                int | None,
                int | None,
                int | None,
            ],
            GraphAnnotations,
        ] = {}

    def annotations(
        self,
        *,
        track_uuid: int,
        name_id: int,
        payload_values: np.ndarray,
    ) -> GraphAnnotations:
        plan = self._plan(track_uuid=track_uuid, name_id=name_id)
        graph_ident = _payload_value(payload_values, plan.graph_index)
        op_ident = _payload_value(payload_values, plan.op_index)
        spmd_ident = _payload_value(payload_values, plan.spmd_index)
        transfer_code = _payload_value(payload_values, plan.transfer_index)
        logical_tuuid = _payload_value(payload_values, plan.logical_tuuid_index)
        event_key = (
            track_uuid,
            name_id,
            graph_ident,
            op_ident,
            spmd_ident,
            transfer_code,
            logical_tuuid,
        )
        cached = self._event_annotations.get(event_key)
        if cached is not None:
            return cached

        if graph_ident is not None:
            self._verify_graph_identity(graph_ident)

        annotations: list[tuple[str, str]] = []
        if graph_ident is not None:
            annotations.append(
                ("graph_generation", str(self._graph_symbols.generation))
            )
        if op_ident is not None and not self._allows_unresolved_operator_identity(
            name_id=name_id,
            op_ident=op_ident,
            plan=plan,
        ):
            annotations.extend(
                self._resolve_operator_annotations(
                    plan=plan,
                    op_ident=op_ident,
                    spmd_ident=spmd_ident,
                )
            )

        transfer = None
        if transfer_code is not None:
            transfer = self._resolve_transfer_annotations(transfer_code)
            annotations.extend(transfer.annotations)

        if logical_tuuid is not None:
            if transfer is not None and transfer.logical_tuuid != logical_tuuid:
                raise GraphTraceSymbolConflict(
                    symbol_kind="trace_event",
                    identity=transfer_code,
                    detail=(
                        f"payload logical TUUID {logical_tuuid} differs from "
                        f"transfer logical TUUID {transfer.logical_tuuid}"
                    ),
                )
            annotations.extend(self._resolve_tensor_annotations(logical_tuuid))
        resolved = tuple(annotations)
        self._event_annotations[event_key] = resolved
        return resolved

    def _allows_unresolved_operator_identity(
        self,
        *,
        name_id: int,
        op_ident: int,
        plan: _GraphAnnotationPlan,
    ) -> bool:
        event_name = self._metadata.name_for(name_id)
        if event_name not in _OPTIONAL_OPERATOR_ID_EVENTS:
            return False
        if self._name(op_ident) == "":
            return True
        return plan.track_spmd_id is None and plan.spmd_index is None

    def _plan(self, *, track_uuid: int, name_id: int) -> _GraphAnnotationPlan:
        key = (track_uuid, name_id)
        cached = self._plans.get(key)
        if cached is not None:
            return cached
        payload_fields = self._metadata.payload_fields_for(name_id)
        field_indices = {field: index for index, field in enumerate(payload_fields)}
        op_index = field_indices.get("op_id")
        spmd_index = field_indices.get("spmd_id")
        plan = _GraphAnnotationPlan(
            graph_index=field_indices.get("graph_id"),
            op_index=op_index,
            spmd_index=spmd_index,
            transfer_index=field_indices.get("transfer_code"),
            logical_tuuid_index=field_indices.get("logical_tuuid"),
            track_spmd_id=(
                self._track_spmd_id(track_uuid) if op_index is not None else None
            ),
        )
        self._plans[key] = plan
        return plan

    def _verify_graph_identity(self, graph_ident: int) -> None:
        if graph_ident in self._verified_graph_idents:
            return
        graph_id = self._name(graph_ident)
        if graph_id != self._graph_symbols.graph_id:
            raise GraphTraceSymbolConflict(
                symbol_kind="trace_graph",
                identity=graph_id,
                detail=(
                    f"trace graph differs from symbol graph "
                    f"{self._graph_symbols.graph_id!r}"
                ),
            )
        self._verified_graph_idents.add(graph_ident)

    def _track_spmd_id(self, track_uuid: int) -> str | None:
        if track_uuid in self._track_spmd_ids:
            return self._track_spmd_ids[track_uuid]
        hierarchy = TraceTrackHierarchy.from_track_name(
            self._metadata.name_for(track_uuid)
        )
        spmd_id = (
            hierarchy.groups[1]
            if len(hierarchy.groups) >= 2 and hierarchy.groups[0] == "SPMDs"
            else None
        )
        self._track_spmd_ids[track_uuid] = spmd_id
        return spmd_id

    def _resolve_operator_annotations(
        self,
        *,
        plan: _GraphAnnotationPlan,
        op_ident: int,
        spmd_ident: int | None,
    ) -> GraphAnnotations:
        if spmd_ident is not None:
            spmd_id = self._name(spmd_ident)
            if plan.track_spmd_id is not None and spmd_id != plan.track_spmd_id:
                raise GraphTraceSymbolConflict(
                    symbol_kind="trace_operator_spmd",
                    identity=(plan.track_spmd_id, spmd_id),
                    detail="payload SPMD differs from profiler track SPMD",
                )
        else:
            spmd_id = plan.track_spmd_id
        if spmd_id is None:
            raise MissingGraphTraceSymbol(
                symbol_kind="operator_spmd",
                identity=op_ident,
            )
        identity = (spmd_id, op_ident)
        cached = self._operator_annotations_by_identity.get(identity)
        if cached is not None:
            return cached
        operator = self._graph_symbols.operator_by_id(
            spmd_id,
            self._name(op_ident),
        )
        annotations = self._operator_annotations_by_key.get(operator.key)
        if annotations is None:
            annotations = _operator_annotations(operator)
            self._operator_annotations_by_key[operator.key] = annotations
        self._operator_annotations_by_identity[identity] = annotations
        return annotations

    def _resolve_transfer_annotations(
        self,
        transfer_code: int,
    ) -> _ResolvedTransferAnnotations:
        cached = self._transfer_annotations.get(transfer_code)
        if cached is not None:
            return cached
        transfer = self._graph_symbols.transfer(transfer_code)
        source = self._graph_symbols.operator(transfer.source_operator)
        target = self._graph_symbols.operator(transfer.target_operator)
        resolved = _ResolvedTransferAnnotations(
            logical_tuuid=transfer.logical_tuuid,
            annotations=(
                ("transfer.source.operator", source.qualified_id),
                ("transfer.source.function", source.function_id),
                ("transfer.source.port", transfer.source_port),
                ("transfer.target.operator", target.qualified_id),
                ("transfer.target.function", target.function_id),
                ("transfer.target.port", transfer.target_port),
            ),
        )
        self._transfer_annotations[transfer_code] = resolved
        return resolved

    def _resolve_tensor_annotations(self, logical_tuuid: int) -> GraphAnnotations:
        cached = self._tensor_annotations.get(logical_tuuid)
        if cached is not None:
            return cached
        tensor = self._graph_symbols.tensor(logical_tuuid)
        resolved = (("logical_tensor.producer", tensor.producer),)
        self._tensor_annotations[logical_tuuid] = resolved
        return resolved

    def _name(self, ident: int) -> str:
        cached = self._names.get(ident)
        if cached is not None:
            return cached
        resolved = self._metadata.name_for(ident)
        self._names[ident] = resolved
        return resolved


def _payload_value(payload_values: np.ndarray, index: int | None) -> int | None:
    return None if index is None else int(payload_values[index])


def _operator_annotations(
    operator: OperatorTraceSymbol,
) -> GraphAnnotations:
    return (("operator.function", operator.function_id),)
