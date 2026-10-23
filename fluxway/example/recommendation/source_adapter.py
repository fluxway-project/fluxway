"""Canonical recommendation metadata planning and source activation."""

from __future__ import annotations

import json

from fluxway.common.ingestion import (
    LayoutUnitDescriptor,
    SourceBindingIdentity,
    ingestion_worker_id,
)
from fluxway.common.ingestion.source_scan import (
    OpaqueSourceCursor,
    SourceScanBudget,
    SourceScanResult,
    SourceScanStatus,
)
from fluxway.common.metadata import metadata_digest, pack_metadata
from fluxway.common.types.configs import SPMDConfig
from fluxway.control_plane.components import DriverComponentContext
from fluxway.control_plane.ingestion.activation import (
    IngestionLayoutActivationComponent,
)
from fluxway.control_plane.ingestion.adapter import (
    IngestionSourceAdapter,
    SourceScanBinding,
)
from fluxway.control_plane.ingestion.compiler import (
    RankDemand,
    ReaderGroupDemand,
    ReaderTopologyDescriptor,
    TailLayoutMode,
)


class RecommendationSourcePlanner:
    """Canonical planner for the ordered recommendation sample stream."""

    def __init__(self, *, source_identity: str, revision_id: int) -> None:
        if not source_identity or source_identity.strip() != source_identity:
            raise ValueError("recommendation source identity must be canonical")
        if (
            not isinstance(revision_id, int)
            or isinstance(revision_id, bool)
            or revision_id < 0
        ):
            raise ValueError("recommendation revision_id must be non-negative")
        self.source_identity = source_identity
        self.revision_id = revision_id

    def plan_units(self, start: int, end: int) -> tuple[LayoutUnitDescriptor, ...]:
        if (
            not isinstance(start, int)
            or not isinstance(end, int)
            or start < 0
            or end < start
        ):
            raise ValueError("recommendation planning interval is invalid")
        revision = f"recommendation:{self.revision_id}"
        if start == end:
            return ()
        return (
            LayoutUnitDescriptor(
                unit_id=f"{self.source_identity}:{self.revision_id}:{start}:{end}",
                revision_identity=revision,
                cardinality=end - start,
                cardinality_metric="samples",
                split_policy="SplittableSpan",
                locality_key="recommendation",
                physical_identity=f"{self.source_identity}:{self.revision_id}",
                logical_start=start,
                opaque_token=pack_metadata({"revision_id": self.revision_id}),
            ),
        )


class RecommendationSourceAdapter(IngestionSourceAdapter):
    """Bounded scan result adapter backed by the canonical planner."""

    def __init__(
        self,
        *,
        source_identity: str = "recommendation",
        revision_id: int = 0,
    ) -> None:
        self._planner = RecommendationSourcePlanner(
            source_identity=source_identity, revision_id=revision_id
        )

    @property
    def source_identity(self) -> str:
        return self._planner.source_identity

    def close(self) -> None:
        """The deterministic planner has no external resources."""

    @property
    def revision_id(self) -> int:
        return self._planner.revision_id

    @staticmethod
    def cursor(*, start: int, end: int, finite: bool) -> OpaqueSourceCursor:
        if (
            not isinstance(start, int)
            or not isinstance(end, int)
            or start < 0
            or end < start
            or not isinstance(finite, bool)
        ):
            raise ValueError("recommendation cursor fields are invalid")
        payload = json.dumps(
            {"end": end, "start": start, "finite": finite},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return OpaqueSourceCursor.from_payload(payload)

    @staticmethod
    def _decode_cursor(cursor: OpaqueSourceCursor) -> tuple[int, int, bool]:
        try:
            payload = json.loads(cursor.payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("recommendation cursor payload is invalid") from error
        if not isinstance(payload, dict):
            raise TypeError("recommendation cursor payload must be an object")
        start = payload["start"]
        end = payload["end"]
        finite = payload["finite"]
        if type(start) is not int or type(end) is not int or start < 0 or end < start:
            raise ValueError("recommendation cursor interval is invalid")
        if not isinstance(finite, bool):
            raise TypeError("recommendation finite flag must be boolean")
        return start, end, finite

    def advance_cursor(
        self, cursor: OpaqueSourceCursor, records: int
    ) -> OpaqueSourceCursor:
        start, end, finite = self._decode_cursor(cursor)
        if type(records) is not int or not 0 <= records <= end - start:
            raise ValueError("consumed records exceed recommendation interval")
        return self.cursor(start=start + records, end=end, finite=finite)

    def scan_next(
        self, cursor: OpaqueSourceCursor, budget: SourceScanBudget
    ) -> SourceScanResult:
        if not isinstance(cursor, OpaqueSourceCursor):
            raise TypeError("recommendation cursor must be opaque")
        if not isinstance(budget, SourceScanBudget):
            raise TypeError("recommendation scan result budget must be typed")
        start, end, finite = self._decode_cursor(cursor)
        scan_result_end = min(end, start + budget.max_records)
        units = self._planner.plan_units(start, scan_result_end)
        if len(units) > budget.max_units:
            raise ValueError("recommendation scan result exceeds unit budget")
        next_cursor = self.cursor(start=scan_result_end, end=end, finite=finite)
        status = SourceScanStatus.from_scan(
            records=scan_result_end - start,
            exhausted=scan_result_end == end,
            finite=finite,
        )
        return SourceScanResult(
            source_identity=self._planner.source_identity,
            view_identity=f"{self._planner.source_identity}:{self._planner.revision_id}",
            schema_digest="recommendation-schema",
            lineage_digest=f"recommendation-revision:{self._planner.revision_id}",
            units=units,
            unit_cursors=tuple(
                self.cursor(start=unit.logical_start, end=end, finite=finite)
                for unit in units
            ),
            records=sum(unit.cardinality for unit in units),
            next_cursor=next_cursor,
            status=status,
        )


class RecommendationSourceDescriptor:
    requires_source_control_runtime = True

    def _build_reader_topology(
        self,
        *,
        spmd_id: str,
        operator_name: str,
        config: SPMDConfig,
    ) -> ReaderTopologyDescriptor:
        worker_count = config.num_workers
        global_batch_size = config.static_args.get("global_batch_size", 24)
        for value in (worker_count, global_batch_size):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(
                    "recommendation ingestion plan dimensions must be positive integers"
                )
        cycle = (
            config.require_graph_view().operator(operator_name).execution_layout.cycle
        )
        active_count = len(cycle[0])
        if (
            any(len(ranks) != active_count for ranks in cycle)
            or global_batch_size % active_count
        ):
            raise ValueError(
                "recommendation global batch must divide evenly over active readers"
            )
        return ReaderTopologyDescriptor(
            global_cardinality=global_batch_size,
            groups=tuple(
                ReaderGroupDemand(
                    ingestion_worker_id(spmd_id, operator_name, rank),
                    (
                        RankDemand(
                            ingestion_worker_id(spmd_id, operator_name, rank),
                            global_batch_size // active_count,
                        ),
                    ),
                )
                for rank in range(worker_count)
            ),
            execution_cycle=cycle,
        )

    def _build_source_scan_adapter(
        self,
        *,
        spmd_id: str,
        operator_name: str,
        config: SPMDConfig,
    ) -> RecommendationSourceAdapter:
        self._build_reader_topology(
            spmd_id=spmd_id, operator_name=operator_name, config=config
        )
        self._dataset_dimensions(config)
        revision = int(
            metadata_digest(
                {
                    key: value
                    for key, value in config.static_args.items()
                    if key not in {"global_batch_size", "audit_dir"}
                }
            )[:16],
            16,
        )
        return RecommendationSourceAdapter(
            revision_id=revision,
        )

    @staticmethod
    def _dataset_dimensions(config: SPMDConfig) -> tuple[int, int]:
        num_samples = config.static_args.get("num_samples", 1000)
        num_epochs = config.static_args.get("num_epochs", 1)
        if (
            not isinstance(num_samples, int)
            or not isinstance(num_epochs, int)
            or num_samples <= 0
            or num_epochs <= 0
        ):
            raise ValueError("recommendation dataset dimensions must be positive")
        return num_samples, num_epochs

    def build_source_scan_binding(
        self,
        *,
        spmd_id: str,
        operator_name: str,
        config: SPMDConfig,
    ) -> SourceScanBinding:
        adapter = self._build_source_scan_adapter(
            spmd_id=spmd_id, operator_name=operator_name, config=config
        )
        num_samples, num_epochs = self._dataset_dimensions(config)
        initial_end = num_samples * num_epochs

        def initial_cursor_factory() -> OpaqueSourceCursor:
            return adapter.cursor(start=0, end=initial_end, finite=True)

        return SourceScanBinding(
            adapter=adapter,
            identity=SourceBindingIdentity(
                configuration_identity=metadata_digest(
                    {
                        "spmd_id": spmd_id,
                        "operator_name": operator_name,
                        "static_args": dict(config.static_args),
                    }
                ),
                source_identity=adapter.source_identity,
                mode_identity="finite",
                native_identity=f"recommendation-revision:{adapter.revision_id}",
            ),
            initial_cursor_factory=initial_cursor_factory,
            topology=self._build_reader_topology(
                spmd_id=spmd_id, operator_name=operator_name, config=config
            ),
            tail_layout_mode=self._build_source_scan_tail_layout(
                spmd_id=spmd_id, operator_name=operator_name, config=config
            ),
        )

    def _build_source_scan_tail_layout(
        self,
        *,
        spmd_id: str,
        operator_name: str,
        config: SPMDConfig,
    ) -> TailLayoutMode:
        del spmd_id, operator_name, config
        return TailLayoutMode.DISCARD_REMAINDER

    def build_component(
        self, context: DriverComponentContext
    ) -> IngestionLayoutActivationComponent:
        return IngestionLayoutActivationComponent(context)


__all__ = [
    "RecommendationSourceAdapter",
    "RecommendationSourceDescriptor",
    "RecommendationSourcePlanner",
]
