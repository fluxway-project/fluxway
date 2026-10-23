"""Ordered source observations and canonical checkpoint planning."""

from __future__ import annotations

import asyncio
import functools
import multiprocessing
import threading
from collections.abc import Awaitable, Callable, Coroutine, Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from typing import Any, TypeVar

from fluxway.common.async_close import AsyncCloseOnce
from fluxway.common.async_operations import AsyncOperationSupervisor
from fluxway.common.ingestion import (
    LayoutUnitDescriptor,
    SourceLayoutActivation,
    WorkerLayoutDescriptor,
)
from fluxway.common.ingestion.segment import _id
from fluxway.common.ingestion.source_scan import (
    OpaqueSourceCursor,
    SourceBindingIdentity,
    SourceScanBudget,
    SourceScanIdentity,
    SourceScanResult,
    SourceScanStatus,
)
from fluxway.common.io_lane import IOLane
from fluxway.common.metadata import metadata_digest
from fluxway.common.types import CheckpointCut
from fluxway.common.types.checkpoint import CheckpointCommitPermit
from fluxway.control_plane.checkpoint_transaction import (
    CheckpointTransactionFingerprint,
    PreparedCheckpointTransaction,
    PreparedStateStore,
)
from fluxway.control_plane.components import ComponentState
from fluxway.control_plane.ingestion.adapter import (
    IngestionSourceAdapter,
    SourceScanBinding,
)
from fluxway.control_plane.ingestion.compiler import (
    AlgebraicLayoutCompiler,
    AssignmentPolicy,
    CompiledIngestionLayout,
    LayoutCompileRequest,
    ReaderTopologyDescriptor,
    TailLayoutMode,
    compile_layout_request,
)
from fluxway.control_plane.logical_work_ledger import (
    LogicalWorkEpoch,
    LogicalWorkLedger,
)

_Result = TypeVar("_Result")


@dataclass(frozen=True, slots=True)
class _SourceCapture:
    cursor: OpaqueSourceCursor
    anchor: OpaqueSourceCursor
    initial_cursor: OpaqueSourceCursor
    identity: SourceScanIdentity
    tail_mode: TailLayoutMode
    record_end: int
    observations: tuple[tuple[int, SourceScanResult], ...]
    advance: Callable[[OpaqueSourceCursor, int], OpaqueSourceCursor] | None


class SourcePlanningBudgetExceeded(ValueError):
    def __init__(
        self,
        *,
        budget_name: str,
        observed: int,
        limit: int,
        unit_count: int,
        row_count: int,
        window_start_step: int,
        window_end_step: int,
        complete_steps: int,
        reader_encoded_bytes: tuple[int, ...],
        reader_cardinalities: tuple[int, ...],
        source_digest: str,
    ) -> None:
        self.budget_name = budget_name
        self.observed = observed
        self.limit = limit
        self.unit_count = unit_count
        self.row_count = row_count
        self.window_start_step = window_start_step
        self.window_end_step = window_end_step
        self.complete_steps = complete_steps
        self.reader_encoded_bytes = reader_encoded_bytes
        self.reader_cardinalities = reader_cardinalities
        self.reader_count = len(reader_encoded_bytes)
        self.source_digest = source_digest
        super().__init__(
            "source planning budget exceeded: "
            f"budget={budget_name} observed={observed} limit={limit} "
            f"units={unit_count} rows={row_count} "
            f"window=[{window_start_step},{window_end_step}) "
            f"steps={complete_steps} "
            f"reader_count={self.reader_count} "
            f"reader_bytes={reader_encoded_bytes} "
            f"reader_cardinalities={reader_cardinalities} "
            f"source_digest={source_digest}"
        )


@dataclass(frozen=True, slots=True)
class _PlanningBudgetFailure:
    budget_name: str
    observed: int
    limit: int
    request: LayoutCompileRequest
    complete_steps: int
    reader_encoded_bytes: tuple[int, ...] = ()

    def to_error(self, *, start_position: int) -> SourcePlanningBudgetExceeded:
        return SourcePlanningBudgetExceeded(
            budget_name=self.budget_name,
            observed=self.observed,
            limit=self.limit,
            unit_count=len(self.request.units),
            row_count=self.request.cardinality,
            window_start_step=start_position,
            window_end_step=start_position + self.complete_steps,
            complete_steps=self.complete_steps,
            reader_encoded_bytes=self.reader_encoded_bytes,
            reader_cardinalities=tuple(
                group.cardinality for group in self.request.topology.groups
            ),
            source_digest=_source_request_digest(self.request),
        )


def _source_request_digest(request: LayoutCompileRequest) -> str:
    return metadata_digest(
        {
            "source_identity": request.source_identity,
            "view_identity": request.view_identity,
            "schema_digest": request.schema_digest,
            "lineage_digest": request.lineage_digest,
        }
    )


def _split_units_at_cardinality(
    units: tuple[LayoutUnitDescriptor, ...], prefix_cardinality: int
) -> tuple[tuple[LayoutUnitDescriptor, ...], tuple[LayoutUnitDescriptor, ...]]:
    if type(prefix_cardinality) is not int or prefix_cardinality < 0:
        raise ValueError("source planning prefix cardinality must be non-negative")
    if prefix_cardinality == 0:
        return (), units
    remaining = prefix_cardinality
    prefix: list[LayoutUnitDescriptor] = []
    suffix: list[LayoutUnitDescriptor] = []
    for unit in units:
        if remaining >= unit.cardinality:
            prefix.append(unit)
            remaining -= unit.cardinality
        elif remaining:
            prefix.append(unit.slice(0, remaining))
            suffix.append(unit.slice(remaining, unit.cardinality - remaining))
            remaining = 0
        else:
            suffix.append(unit)
    if remaining:
        raise ValueError("source planning target exceeds available cardinality")
    return tuple(prefix), tuple(suffix)


class IngestionLayoutPlanningComponent:
    def __init__(
        self, component_id: str, *, topology: ReaderTopologyDescriptor | None = None
    ) -> None:
        self.component_id = _id(component_id, "layout planning component id")
        self._topology = topology
        self._windows: dict[int, CompiledIngestionLayout] = {}
        self._pending_units: tuple[LayoutUnitDescriptor, ...] = ()
        self._source_request: LayoutCompileRequest | None = None
        self._capture: _SourceCapture | None = None
        self._source_adapter: IngestionSourceAdapter | None = None
        self._source_binding: SourceScanBinding | None = None
        self._source_binding_lock = asyncio.Lock()
        self._initial_cursor_factory: (
            Callable[
                [], OpaqueSourceCursor | None | Awaitable[OpaqueSourceCursor | None]
            ]
            | None
        ) = None
        self._binding_identity: SourceBindingIdentity | None = None
        self._source_budget: SourceScanBudget | None = None
        self._source_tail_layout: TailLayoutMode | None = None
        self._source_assignment_policy: AssignmentPolicy | None = None
        self._resume_step = 0
        self._durable_step = 0
        self._accept_lock = threading.Lock()
        self._window_lock = asyncio.Lock()
        self._publication_lock = asyncio.Lock()
        self._adapter_lane = IOLane(
            f"fluxway-source-adapter-{self.component_id}", capacity=1
        )
        self._lifecycle_lane = IOLane(
            f"fluxway-source-lifecycle-{self.component_id}", capacity=1
        )
        self._compile_executor: ProcessPoolExecutor | None = None
        self._close_once = AsyncCloseOnce(self._finish_close)
        self._operations = AsyncOperationSupervisor()
        self._checkpoint_transactions = PreparedStateStore[dict[str, Any]]()

    @property
    def layout_digest(self) -> str | None:
        return (
            None
            if not self._windows
            else self._windows[max(self._windows)].layout_digest
        )

    @property
    def windows(self) -> tuple[CompiledIngestionLayout, ...]:
        return tuple(self._windows.values())

    def set_reader_topology(self, topology: ReaderTopologyDescriptor) -> None:
        if not isinstance(topology, ReaderTopologyDescriptor):
            raise TypeError("source planning requires ReaderTopologyDescriptor")
        if self._windows:
            raise RuntimeError(
                "reader topology must be set before compiling source windows"
            )
        self._topology = topology

    async def open_source_binding(
        self, factory: Callable[..., Any], **kwargs: Any
    ) -> SourceScanBinding:
        """Own construction and cleanup from the first successful binding."""

        async def open_once() -> SourceScanBinding:
            async with self._source_binding_lock:
                if self._source_binding is None:
                    if self._source_adapter is not None:
                        raise RuntimeError("source adapter is already bound directly")
                    binding = await self._adapter_lane.call(factory, **kwargs)
                    if not isinstance(binding, SourceScanBinding):
                        raise TypeError(
                            "source descriptor must return SourceScanBinding"
                        )
                    self._source_binding = binding
                    self._source_adapter = binding.adapter
                return self._source_binding

        return await self._await_operation(open_once)

    def bind_source_scan(
        self,
        adapter: IngestionSourceAdapter,
        *,
        initial_cursor_factory: Callable[
            [], OpaqueSourceCursor | None | Awaitable[OpaqueSourceCursor | None]
        ],
        budget: SourceScanBudget,
        tail_layout_mode: TailLayoutMode | str,
        assignment_policy: AssignmentPolicy,
    ) -> None:
        if not isinstance(adapter, IngestionSourceAdapter):
            raise TypeError("source planning requires an ingestion source adapter")
        if not callable(initial_cursor_factory):
            raise TypeError("source planning requires an initial cursor factory")
        if not isinstance(budget, SourceScanBudget):
            raise TypeError("source planning requires SourceScanBudget")
        resolved_tail = TailLayoutMode.from_value(tail_layout_mode)
        resolved_policy = AssignmentPolicy.from_value(assignment_policy)
        with self._accept_lock:
            if self._source_adapter is not None and self._source_adapter is not adapter:
                raise ValueError("source scan adapter changed during planner lifetime")
            if (
                self._initial_cursor_factory is not None
                and self._initial_cursor_factory != initial_cursor_factory
            ):
                raise ValueError(
                    "source initial cursor factory changed during planner lifetime"
                )
            if self._source_budget is not None and self._source_budget != budget:
                raise ValueError("source scan budget changed during planner lifetime")
            if (
                self._source_tail_layout is not None
                and self._source_tail_layout is not resolved_tail
            ):
                raise ValueError("source tail policy changed during planner lifetime")
            if (
                self._source_assignment_policy is not None
                and self._source_assignment_policy != resolved_policy
            ):
                raise ValueError(
                    "source assignment policy changed during planner lifetime"
                )
            self._source_adapter = adapter
            self._initial_cursor_factory = initial_cursor_factory
            self._source_budget = budget
            self._source_tail_layout = resolved_tail
            self._source_assignment_policy = resolved_policy

    def bind_source_identity(self, identity: SourceBindingIdentity) -> None:
        if not isinstance(identity, SourceBindingIdentity):
            raise TypeError("source planning requires SourceBindingIdentity")
        with self._accept_lock:
            if (
                self._binding_identity is not None
                and self._binding_identity != identity
            ):
                raise ValueError(
                    "source binding identity changed during planner lifetime"
                )
            self._binding_identity = identity

    async def on_state(self, state: ComponentState) -> None:
        if state is None:
            raise TypeError("layout planning component state is required")

    def _compile_pool(self) -> ProcessPoolExecutor:
        if self._compile_executor is None:
            self._compile_executor = ProcessPoolExecutor(
                max_workers=1,
                mp_context=multiprocessing.get_context("spawn"),
            )
        return self._compile_executor

    async def _await_operation(
        self, operation: Callable[[], Coroutine[Any, Any, _Result]]
    ) -> _Result:
        return await self._operations.run(operation)

    async def close(self) -> None:
        self._operations.stop_accepting()
        await self._close_once.close()

    async def _finish_close(self) -> None:
        failures = list(await self._operations.drain())
        compile_executor = self._compile_executor
        self._compile_executor = None
        closing = []
        if self._source_adapter is not None:
            closing.append(self._adapter_lane.call(self._source_adapter.close))
        if compile_executor is not None:
            closing.append(
                self._lifecycle_lane.call(
                    compile_executor.shutdown,
                    wait=True,
                    cancel_futures=True,
                )
            )
        results = await asyncio.gather(*closing, return_exceptions=True)
        results.extend(
            await asyncio.gather(
                self._adapter_lane.close(),
                self._lifecycle_lane.close(),
                return_exceptions=True,
            )
        )
        failures.extend(
            result for result in results if isinstance(result, BaseException)
        )
        if failures:
            raise BaseExceptionGroup("source planning close failed", failures)

    async def _compile_prepared(
        self, request: LayoutCompileRequest, *, epoch: int, start_position: int
    ) -> CompiledIngestionLayout:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._compile_pool(),
            functools.partial(compile_layout_request, request, epoch, start_position),
        )

    def accept(
        self, request: LayoutCompileRequest, *, epoch: int = 0, start_step: int = 0
    ) -> CompiledIngestionLayout:
        with self._accept_lock:
            prior = self._windows.get(epoch)
            if prior is not None:
                if (
                    prior.request != request
                    or prior.work_epoch.start_step != start_step
                ):
                    raise ValueError(
                        "layout planning component received conflicting request"
                    )
                return prior
            previous = self._windows[max(self._windows)] if self._windows else None
            expected_epoch = 0 if previous is None else previous.work_epoch.epoch + 1
            expected_start = (
                self._resume_step if previous is None else previous.work_epoch.end_step
            )
            if epoch != expected_epoch or start_step != expected_start:
                raise ValueError("layout planning epochs and steps must be contiguous")
            if previous is not None:
                if previous.work_epoch.terminal:
                    raise ValueError(
                        "layout planning cannot append after terminal window"
                    )
                if (
                    previous.request.source_identity,
                    previous.request.view_identity,
                    previous.request.schema_digest,
                    previous.request.lineage_digest,
                    previous.request.topology.global_cardinality,
                ) != (
                    request.source_identity,
                    request.view_identity,
                    request.schema_digest,
                    request.lineage_digest,
                    request.topology.global_cardinality,
                ):
                    raise ValueError(
                        "layout planning source, schema or logical batch changed"
                    )
            compiled = AlgebraicLayoutCompiler().compile(
                request, epoch=epoch, start_step=start_step
            )
            self._windows[epoch] = compiled
            self._topology = request.topology
            return compiled

    def logical_work_epoch(self, *, epoch: int, start_step: int) -> LogicalWorkEpoch:
        plan = self._windows[epoch].work_epoch
        if plan.start_step != start_step:
            raise ValueError("logical work start step differs from planned window")
        return plan

    def _validate_prepared(
        self,
        request: LayoutCompileRequest,
        compiled: CompiledIngestionLayout,
        *,
        epoch: int,
        start_position: int,
    ) -> None:
        with self._accept_lock:
            prior = self._windows.get(epoch)
            if prior is not None:
                if prior != compiled or prior.request != request:
                    raise ValueError(
                        "layout publication conflicts with existing window"
                    )
                return
            previous = self._windows[max(self._windows)] if self._windows else None
            expected_epoch = 0 if previous is None else previous.work_epoch.epoch + 1
            expected_start = (
                self._resume_step if previous is None else previous.work_epoch.end_step
            )
            if epoch != expected_epoch or start_position != expected_start:
                raise ValueError("layout planning epochs and steps must be contiguous")
            if previous is not None:
                if previous.work_epoch.terminal:
                    raise ValueError(
                        "layout planning cannot append after terminal window"
                    )
                if (
                    previous.request.source_identity,
                    previous.request.view_identity,
                    previous.request.schema_digest,
                    previous.request.lineage_digest,
                    previous.request.topology.global_cardinality,
                ) != (
                    request.source_identity,
                    request.view_identity,
                    request.schema_digest,
                    request.lineage_digest,
                    request.topology.global_cardinality,
                ):
                    raise ValueError(
                        "layout planning source, schema or logical batch changed"
                    )
            if (
                compiled.request != request
                or compiled.work_epoch.start_step != start_position
            ):
                raise ValueError("compiled layout does not match publication request")

    def _install_prepared(
        self,
        request: LayoutCompileRequest,
        compiled: CompiledIngestionLayout,
        *,
        pending_units: tuple[LayoutUnitDescriptor, ...],
        capture: _SourceCapture | None,
    ) -> None:
        with self._accept_lock:
            self._windows[compiled.work_epoch.epoch] = compiled
            self._topology = request.topology
            self._pending_units = pending_units
            if capture is not None:
                self._capture = capture

    async def _plan_and_publish(
        self,
        ledger: LogicalWorkLedger,
        request: LayoutCompileRequest,
        *,
        epoch: int,
        start_position: int,
        max_units: int | None = None,
        max_records: int | None = None,
        max_steps: int | None = None,
        max_descriptor_bytes: int | None = None,
        capture: _SourceCapture | None = None,
    ) -> SourceLayoutActivation:
        if not isinstance(ledger, LogicalWorkLedger):
            raise TypeError("layout publication requires LogicalWorkLedger")
        with self._accept_lock:
            if self._source_request is None and not self._windows:
                self._source_request = request

        async with self._publication_lock:
            total_records = request.cardinality
            batch = request.topology.global_cardinality
            full_steps = total_records // batch
            has_remainder = (total_records % batch) > 0
            final_tail_mode = TailLayoutMode.from_value(request.tail_layout_mode)
            available_steps = full_steps
            if full_steps == 0:
                if final_tail_mode is not TailLayoutMode.SYNTHESIZE_PADDING:
                    raise ValueError(
                        "source planning request has no complete global step"
                    )
                available_steps = 1

            if max_steps is not None:
                available_steps = min(available_steps, max_steps)
            if available_steps == 0:
                raise AssertionError("source planning lost its initial candidate step")

            candidate_steps = available_steps
            compiled = None
            candidate_request = request
            selected_pending_units: tuple[LayoutUnitDescriptor, ...] = ()
            last_failure: _PlanningBudgetFailure | None = None
            while candidate_steps > 0:
                is_final_candidate = full_steps == 0 or candidate_steps == full_steps
                if (
                    is_final_candidate
                    and has_remainder
                    and final_tail_mode is TailLayoutMode.SYNTHESIZE_PADDING
                ):
                    expected_steps = 1 if full_steps == 0 else candidate_steps + 1
                    if max_steps is not None and expected_steps > max_steps:
                        if full_steps == 0:
                            last_failure = _PlanningBudgetFailure(
                                "steps",
                                expected_steps,
                                max_steps,
                                candidate_request,
                                candidate_steps,
                            )
                            candidate_steps -= 1
                            continue
                        is_final_candidate = False
                candidate_records = (
                    total_records if is_final_candidate else candidate_steps * batch
                )
                candidate_units, pending_candidate = _split_units_at_cardinality(
                    request.units, candidate_records
                )
                candidate_mode = (
                    final_tail_mode
                    if is_final_candidate
                    else TailLayoutMode.RETAIN_PENDING
                )
                candidate_request = replace(
                    request, units=candidate_units, tail_layout_mode=candidate_mode
                )
                if max_records is not None and candidate_records > max_records:
                    last_failure = _PlanningBudgetFailure(
                        "records",
                        candidate_records,
                        max_records,
                        candidate_request,
                        candidate_steps,
                    )
                    candidate_steps -= 1
                    continue
                if max_units is not None and len(candidate_units) > max_units:
                    last_failure = _PlanningBudgetFailure(
                        "units",
                        len(candidate_units),
                        max_units,
                        candidate_request,
                        candidate_steps,
                    )
                    candidate_steps -= 1
                    continue
                compiled_candidate = await self._compile_prepared(
                    candidate_request, epoch=epoch, start_position=start_position
                )
                compiled_steps = (
                    compiled_candidate.work_epoch.end_step
                    - compiled_candidate.work_epoch.start_step
                )
                if max_steps is not None and compiled_steps > max_steps:
                    last_failure = _PlanningBudgetFailure(
                        "steps",
                        compiled_steps,
                        max_steps,
                        candidate_request,
                        candidate_steps,
                    )
                    candidate_steps -= 1
                    continue
                encoded_candidate = compiled_candidate.encoded_worker_layouts()
                encoded_candidate_bytes = sum(
                    len(value.encoded) for value in encoded_candidate
                )
                if (
                    max_descriptor_bytes is None
                    or encoded_candidate_bytes <= max_descriptor_bytes
                ):
                    compiled = compiled_candidate
                    selected_pending_units = pending_candidate
                    break
                last_failure = _PlanningBudgetFailure(
                    "encoded_descriptor_bytes",
                    encoded_candidate_bytes,
                    max_descriptor_bytes,
                    candidate_request,
                    candidate_steps,
                    tuple(len(value.encoded) for value in encoded_candidate),
                )
                candidate_steps -= 1
            if compiled is None:
                if last_failure is None:
                    raise AssertionError(
                        "source planning exhausted candidates without a budget failure"
                    )
                raise last_failure.to_error(start_position=start_position)
            request = candidate_request
            encoded_layouts = compiled.encoded_worker_layouts()
            reader_encoded_bytes = tuple(len(item.encoded) for item in encoded_layouts)
            reader_cardinalities = tuple(
                group.cardinality for group in request.topology.groups
            )
            source_digest = _source_request_digest(request)
            window_start_step = compiled.work_epoch.start_step
            window_end_step = compiled.work_epoch.end_step
            complete_steps = window_end_step - window_start_step
            if max_steps is not None and complete_steps > max_steps:
                raise SourcePlanningBudgetExceeded(
                    budget_name="steps",
                    observed=complete_steps,
                    limit=max_steps,
                    unit_count=len(request.units),
                    row_count=request.cardinality,
                    window_start_step=window_start_step,
                    window_end_step=window_end_step,
                    complete_steps=complete_steps,
                    reader_encoded_bytes=reader_encoded_bytes,
                    reader_cardinalities=reader_cardinalities,
                    source_digest=source_digest,
                )
            if max_units is not None and len(request.units) > max_units:
                raise SourcePlanningBudgetExceeded(
                    budget_name="units",
                    observed=len(request.units),
                    limit=max_units,
                    unit_count=len(request.units),
                    row_count=request.cardinality,
                    window_start_step=window_start_step,
                    window_end_step=window_end_step,
                    complete_steps=complete_steps,
                    reader_encoded_bytes=reader_encoded_bytes,
                    reader_cardinalities=reader_cardinalities,
                    source_digest=source_digest,
                )
            if max_descriptor_bytes is not None:
                encoded_size = sum(reader_encoded_bytes)
                if encoded_size > max_descriptor_bytes:
                    raise SourcePlanningBudgetExceeded(
                        budget_name="encoded_descriptor_bytes",
                        observed=encoded_size,
                        limit=max_descriptor_bytes,
                        unit_count=len(request.units),
                        row_count=request.cardinality,
                        window_start_step=window_start_step,
                        window_end_step=window_end_step,
                        complete_steps=complete_steps,
                        reader_encoded_bytes=reader_encoded_bytes,
                        reader_cardinalities=reader_cardinalities,
                        source_digest=source_digest,
                    )
            self._validate_prepared(
                request, compiled, epoch=epoch, start_position=start_position
            )
            activation = compiled.to_source_layout_activation(
                epoch=epoch,
                start_position=start_position,
                end_position=compiled.work_epoch.end_step,
                terminal=compiled.work_epoch.terminal,
            )
            prepared = ledger.prepare_publication(
                compiled.work_epoch,
                tuple(
                    WorkerLayoutDescriptor.decode(value)
                    for value in activation.descriptors
                ),
            )
            # All validation and compilation are complete. This owner-loop section
            # intentionally contains no await: planner and ledger become visible
            # as one publication image.
            self._install_prepared(
                request,
                compiled,
                pending_units=selected_pending_units + compiled.pending_units,
                capture=capture,
            )
            ledger.apply_publication(prepared)
            return activation

    async def plan_and_publish(
        self,
        ledger: LogicalWorkLedger,
        request: LayoutCompileRequest,
        *,
        epoch: int,
        start_position: int,
        max_units: int | None = None,
        max_records: int | None = None,
        max_steps: int | None = None,
        max_descriptor_bytes: int | None = None,
    ) -> SourceLayoutActivation:
        return await self._await_operation(
            functools.partial(
                self._plan_and_publish,
                ledger,
                request,
                epoch=epoch,
                start_position=start_position,
                max_units=max_units,
                max_records=max_records,
                max_steps=max_steps,
                max_descriptor_bytes=max_descriptor_bytes,
            )
        )

    async def _plan_next_window(
        self,
        ledger: LogicalWorkLedger,
        *,
        max_units: int | None,
        max_records: int | None,
        max_steps: int | None,
        max_descriptor_bytes: int | None,
    ) -> SourceLayoutActivation | None:
        with self._accept_lock:
            source_request = self._source_request
            pending_units = self._pending_units
            previous = self._windows.get(max(self._windows)) if self._windows else None
        if source_request is None or previous is None:
            return None
        if previous.work_epoch.terminal:
            return None
        pending_records = sum(unit.cardinality for unit in pending_units)
        if (
            pending_records < source_request.topology.global_cardinality
            and self._source_adapter is not None
        ):
            with self._accept_lock:
                adapter = self._source_adapter
                initial_cursor_factory = self._initial_cursor_factory
                budget = self._source_budget
                tail_layout_mode = self._source_tail_layout
                assignment_policy = self._source_assignment_policy
                capture = self._capture
            if (
                adapter is None
                or initial_cursor_factory is None
                or budget is None
                or tail_layout_mode is None
                or assignment_policy is None
            ):
                return None
            return await self._plan_source_scan(
                adapter,
                initial_cursor_factory,
                ledger,
                None if capture is None else capture.cursor,
                budget=budget,
                topology=source_request.topology,
                tail_layout_mode=tail_layout_mode,
                assignment_policy=assignment_policy,
                epoch=previous.work_epoch.epoch + 1,
                start_position=previous.work_epoch.end_step,
                max_steps=max_steps,
                max_descriptor_bytes=max_descriptor_bytes,
            )
        if not pending_units or (
            pending_records < source_request.topology.global_cardinality
            and TailLayoutMode.from_value(source_request.tail_layout_mode)
            is TailLayoutMode.RETAIN_PENDING
        ):
            return None
        request = replace(source_request, units=pending_units)
        return await self._plan_and_publish(
            ledger,
            request,
            epoch=previous.work_epoch.epoch + 1,
            start_position=previous.work_epoch.end_step,
            max_units=max_units,
            max_records=max_records,
            max_steps=max_steps,
            max_descriptor_bytes=max_descriptor_bytes,
        )

    async def plan_next_window(
        self,
        ledger: LogicalWorkLedger,
        *,
        max_units: int | None = None,
        max_records: int | None = None,
        max_steps: int | None = None,
        max_descriptor_bytes: int | None = None,
    ) -> SourceLayoutActivation | None:
        return await self._await_operation(
            functools.partial(
                self._plan_next_window,
                ledger,
                max_units=max_units,
                max_records=max_records,
                max_steps=max_steps,
                max_descriptor_bytes=max_descriptor_bytes,
            )
        )

    async def _plan_source_scan(
        self,
        adapter: IngestionSourceAdapter,
        initial_cursor_factory: Callable[
            [], OpaqueSourceCursor | None | Awaitable[OpaqueSourceCursor | None]
        ],
        ledger: LogicalWorkLedger,
        cursor: OpaqueSourceCursor | None,
        *,
        budget: SourceScanBudget,
        topology: ReaderTopologyDescriptor,
        tail_layout_mode: TailLayoutMode | str,
        assignment_policy: AssignmentPolicy,
        epoch: int,
        start_position: int,
        max_steps: int | None = None,
        max_descriptor_bytes: int | None = None,
    ) -> SourceLayoutActivation | None:
        self.bind_source_scan(
            adapter,
            initial_cursor_factory=initial_cursor_factory,
            budget=budget,
            tail_layout_mode=tail_layout_mode,
            assignment_policy=assignment_policy,
        )
        if not isinstance(adapter, IngestionSourceAdapter):
            raise TypeError("adapter does not implement scan_next")
        if not isinstance(budget, SourceScanBudget):
            raise TypeError("source scan result planning requires SourceScanBudget")
        requested_mode = TailLayoutMode.from_value(tail_layout_mode)
        advance = adapter.advance_cursor
        if not callable(advance):
            raise TypeError("source adapter must provide pure cursor advancement")
        with self._accept_lock:
            pending = self._pending_units
            capture = self._capture
            topology_before = self._topology
        if (
            topology_before is not None
            and topology_before.global_cardinality != topology.global_cardinality
        ):
            raise ValueError("source cannot change logical batch cardinality")
        if capture is not None and capture.tail_mode is not requested_mode:
            raise ValueError("source cannot change its terminal tail policy")
        if capture is None:
            if cursor is None:
                cursor = await self._adapter_lane.call(initial_cursor_factory)
            if cursor is None:
                return None
            if not isinstance(cursor, OpaqueSourceCursor):
                raise TypeError(
                    "source initial cursor factory must return an opaque cursor"
                )
            current_cursor = cursor
            anchor = cursor
            initial_cursor = cursor
        else:
            current_cursor = capture.cursor
            anchor = capture.anchor
            initial_cursor = capture.initial_cursor
        observations = () if capture is None else capture.observations
        identity = None if capture is None else capture.identity
        record_position = (
            self._resume_step * topology.global_cardinality
            if capture is None
            else capture.record_end
        )
        while True:
            scan_result = await self._adapter_lane.call(
                adapter.scan_next, current_cursor, budget
            )
            if not isinstance(scan_result, SourceScanResult):
                raise TypeError("scan adapter must return SourceScanResult")
            scan_result.validate_transition(current_cursor)
            scan_result.validate_budget(budget)
            scan_result_identity = scan_result.identity
            if identity is not None and (
                identity.source_identity != scan_result_identity.source_identity
                or identity.schema_digest != scan_result_identity.schema_digest
                or (identity != scan_result_identity and pending)
            ):
                raise ValueError("source scan result identity or schema changed")
            identity = scan_result_identity
            if scan_result.units:
                observations = observations + ((record_position, scan_result),)
                record_position += scan_result.records
            capture = _SourceCapture(
                scan_result.next_cursor,
                anchor,
                initial_cursor,
                identity,
                requested_mode,
                record_position,
                observations,
                advance,
            )
            if scan_result.status is SourceScanStatus.IDLE:
                with self._accept_lock:
                    self._pending_units = pending
                    self._capture = capture
                    self._topology = topology
                return None
            pending = pending + scan_result.units
            current_cursor = scan_result.next_cursor
            terminal = scan_result.status is SourceScanStatus.TERMINAL
            available_records = sum(unit.cardinality for unit in pending)
            complete_steps = available_records // topology.global_cardinality
            if not terminal and complete_steps == 0:
                continue
            if terminal and complete_steps == 0:
                if requested_mode is TailLayoutMode.SYNTHESIZE_PADDING:
                    request = LayoutCompileRequest(
                        source_identity=scan_result.source_identity,
                        view_identity=scan_result.view_identity,
                        schema_digest=scan_result.schema_digest,
                        lineage_digest=scan_result.lineage_digest,
                        units=pending,
                        topology=topology,
                        tail_layout_mode=requested_mode,
                        assignment_policy=assignment_policy,
                    )
                    activation = await self._plan_and_publish(
                        ledger,
                        request,
                        epoch=epoch,
                        start_position=start_position,
                        capture=capture,
                        max_steps=max_steps,
                        max_descriptor_bytes=max_descriptor_bytes,
                    )
                    return activation
                if requested_mode is TailLayoutMode.REJECT_REMAINDER:
                    raise ValueError("incomplete terminal global cardinality")
                with self._accept_lock:
                    self._pending_units = ()
                    self._capture = capture
                    self._topology = topology
                return None
            effective_mode = (
                requested_mode if terminal else TailLayoutMode.RETAIN_PENDING
            )
            request = LayoutCompileRequest(
                source_identity=scan_result.source_identity,
                view_identity=scan_result.view_identity,
                schema_digest=scan_result.schema_digest,
                lineage_digest=scan_result.lineage_digest,
                units=pending,
                topology=topology,
                tail_layout_mode=effective_mode,
                assignment_policy=assignment_policy,
            )
            return await self._plan_and_publish(
                ledger,
                request,
                epoch=epoch,
                start_position=start_position,
                capture=capture,
                max_steps=max_steps,
                max_descriptor_bytes=max_descriptor_bytes,
            )

    async def plan_source_scan(
        self,
        adapter: IngestionSourceAdapter,
        initial_cursor_factory: Callable[
            [], OpaqueSourceCursor | None | Awaitable[OpaqueSourceCursor | None]
        ],
        ledger: LogicalWorkLedger,
        cursor: OpaqueSourceCursor | None,
        *,
        budget: SourceScanBudget,
        topology: ReaderTopologyDescriptor,
        tail_layout_mode: TailLayoutMode | str,
        assignment_policy: AssignmentPolicy,
        epoch: int,
        start_position: int,
        max_steps: int | None = None,
        max_descriptor_bytes: int | None = None,
    ) -> SourceLayoutActivation | None:
        return await self._await_operation(
            functools.partial(
                self._plan_source_scan,
                adapter,
                initial_cursor_factory,
                ledger,
                cursor,
                budget=budget,
                topology=topology,
                tail_layout_mode=tail_layout_mode,
                assignment_policy=assignment_policy,
                epoch=epoch,
                start_position=start_position,
                max_steps=max_steps,
                max_descriptor_bytes=max_descriptor_bytes,
            )
        )

    @property
    def resume_step(self) -> int:
        return self._resume_step

    @property
    def resume_cursor(self) -> OpaqueSourceCursor | None:
        return None if self._capture is None else self._capture.cursor

    def checkpoint_state(self, cut: CheckpointCut) -> Mapping[str, Any]:
        if not isinstance(cut, CheckpointCut) or cut.position is None:
            raise ValueError("layout checkpoint requires a positioned cut")
        with self._accept_lock:
            latest = max(
                (window.work_epoch.end_step for window in self._windows.values()),
                default=self._resume_step,
            )
            if not self._durable_step <= cut.position <= latest:
                raise ValueError("layout checkpoint cut is outside retained progress")
            capture = self._capture
            if capture is None and self._windows:
                raise ValueError(
                    "layout checkpoint requires a source consumption cursor"
                )
            cardinality = (
                None if self._topology is None else self._topology.global_cardinality
            )
            cursor = None if capture is None else self._cursor_at_cut(cut.position)
            return {
                "cut_position": cut.position,
                "cursor": None if cursor is None else cursor.to_checkpoint(),
                "initial_cursor": None
                if capture is None
                else capture.initial_cursor.to_checkpoint(),
                "binding_identity": None
                if self._binding_identity is None
                else self._binding_identity.to_checkpoint(),
                "source_identity": None
                if capture is None
                else capture.identity.source_identity,
                "view_identity": None
                if capture is None
                else capture.identity.view_identity,
                "schema_digest": None
                if capture is None
                else capture.identity.schema_digest,
                "lineage_digest": None
                if capture is None
                else capture.identity.lineage_digest,
                "global_cardinality": cardinality,
                "tail_layout_mode": None
                if capture is None
                else capture.tail_mode.value,
            }

    def _cursor_at_cut(self, position: int) -> OpaqueSourceCursor:
        capture = self._capture
        if capture is None or self._topology is None:
            raise RuntimeError(
                "source cursor calculation requires a capture and reader topology"
            )
        if position == self._durable_step:
            return capture.anchor
        target = position * self._topology.global_cardinality
        for start, result in capture.observations:
            if target < start:
                raise ValueError("source observations do not cover checkpoint cut")
            if target <= start + result.records:
                if capture.advance is None:
                    raise RuntimeError("source cursor advancement is not bound")
                return result.cursor_at(target - start, capture.advance)
        return capture.cursor

    def prepare_checkpoint(
        self,
        checkpoint_id: str,
        permit: CheckpointCommitPermit,
        cut: CheckpointCut,
        state_digest: str,
    ) -> PreparedCheckpointTransaction[dict[str, Any]]:
        if not isinstance(permit, CheckpointCommitPermit):
            raise TypeError("planning checkpoint requires CheckpointCommitPermit")
        state = dict(self.checkpoint_state(cut))
        if metadata_digest(state) != state_digest:
            raise ValueError("planning checkpoint state digest is invalid")
        fingerprint = CheckpointTransactionFingerprint.create(
            permit=permit, cut=cut, state_digest=state_digest
        )
        self._checkpoint_transactions.prepare(checkpoint_id, fingerprint, state)
        return self._checkpoint_transactions.prepared(checkpoint_id)

    def apply_prepared(
        self, prepared_state: PreparedCheckpointTransaction[dict[str, Any]]
    ) -> None:
        if not isinstance(prepared_state, PreparedCheckpointTransaction):
            raise TypeError("planning checkpoint prepared state has invalid type")
        self._checkpoint_transactions.apply(
            prepared_state.checkpoint_id,
            prepared_state.fingerprint,
            lambda _state: None,
        )

    def abort_prepared(
        self, checkpoint_id: str, permit: CheckpointCommitPermit
    ) -> None:
        self._checkpoint_transactions.abort(checkpoint_id, permit)

    def reclaim_before(self, durable_cut: int) -> None:
        """Release planner history after the ledger has durably committed a cut."""
        if (
            not isinstance(durable_cut, int)
            or isinstance(durable_cut, bool)
            or durable_cut < 0
        ):
            raise ValueError("planner reclaim cut must be a non-negative integer")
        with self._accept_lock:
            if not self._windows:
                return
            if durable_cut < self._durable_step:
                raise ValueError("planner durable cut cannot move backwards")
            if self._capture is not None:
                anchor = self._cursor_at_cut(durable_cut)
                if self._topology is None:
                    raise RuntimeError(
                        "source reclaim requires the captured reader topology"
                    )
                record_cut = durable_cut * self._topology.global_cardinality
                self._capture = replace(
                    self._capture,
                    anchor=anchor,
                    observations=tuple(
                        (start, result)
                        for start, result in self._capture.observations
                        if start + result.records > record_cut
                    ),
                )
            self._durable_step = durable_cut
            ordered = tuple(self._windows.items())
            removable = tuple(
                epoch_id
                for epoch_id, compiled in ordered[:-1]
                if compiled.work_epoch.end_step <= durable_cut
            )
            for epoch_id in removable:
                self._windows.pop(epoch_id)

    def restore_state(self, state: Mapping[str, Any], cut: CheckpointCut) -> None:
        if not isinstance(cut, CheckpointCut) or cut.position is None:
            raise ValueError("layout restore requires a positioned cut")
        if not isinstance(state, Mapping) or set(state) != {
            "cut_position",
            "cursor",
            "initial_cursor",
            "binding_identity",
            "source_identity",
            "view_identity",
            "schema_digest",
            "lineage_digest",
            "global_cardinality",
            "tail_layout_mode",
        }:
            raise ValueError("layout checkpoint fields do not match")
        if (
            type(state["cut_position"]) is not int
            or state["cut_position"] != cut.position
        ):
            raise ValueError("layout checkpoint cut position does not match")
        cardinality = state["global_cardinality"]
        if cardinality is not None and (
            type(cardinality) is not int or cardinality <= 0
        ):
            raise ValueError("source checkpoint global cardinality must be positive")
        if cardinality is not None and (
            self._topology is None or self._topology.global_cardinality != cardinality
        ):
            raise ValueError(
                "source restore requires unchanged logical batch cardinality"
            )
        binding_value = state["binding_identity"]
        restored_binding = (
            None
            if binding_value is None
            else SourceBindingIdentity.from_checkpoint(binding_value)
        )
        capture = None
        if state["cursor"] is not None:
            identity = SourceScanIdentity(
                source_identity=_id(
                    state["source_identity"], "source checkpoint identity"
                ),
                view_identity=_id(state["view_identity"], "source checkpoint view"),
                schema_digest=_id(state["schema_digest"], "source checkpoint schema"),
                lineage_digest=_id(
                    state["lineage_digest"], "source checkpoint lineage"
                ),
            )
            if cardinality is None:
                raise ValueError("source checkpoint requires logical batch cardinality")
            cursor = OpaqueSourceCursor.from_checkpoint(state["cursor"])
            if state["initial_cursor"] is None:
                raise ValueError("source checkpoint requires its initial cursor")
            anchor = OpaqueSourceCursor.from_checkpoint(state["initial_cursor"])
            if cut.position == 0 and anchor != cursor:
                raise ValueError(
                    "source checkpoint initial cursor disagrees with zero cut"
                )
            capture = _SourceCapture(
                cursor,
                cursor,
                anchor,
                identity,
                TailLayoutMode.from_value(state["tail_layout_mode"]),
                cut.position * cardinality,
                (),
                None,
            )
        elif cut.position != 0 or any(
            state[key] is not None
            for key in (
                "initial_cursor",
                "source_identity",
                "view_identity",
                "schema_digest",
                "lineage_digest",
                "tail_layout_mode",
            )
        ):
            raise ValueError("source checkpoint lacks a consumption cursor")
        with self._accept_lock:
            if (
                self._binding_identity is not None
                and self._binding_identity != restored_binding
            ):
                raise ValueError("source binding identity differs from checkpoint")
            self._windows = {}
            self._pending_units = ()
            self._source_request = None
            self._capture = capture
            self._binding_identity = restored_binding
            self._resume_step = cut.position
            self._durable_step = cut.position
            self._checkpoint_transactions = PreparedStateStore()
