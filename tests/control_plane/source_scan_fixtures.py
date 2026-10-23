from __future__ import annotations

from dataclasses import replace

from fluxway.common.ingestion.source_scan import (
    OpaqueSourceCursor,
    SourceScanBudget,
    SourceScanResult,
    SourceScanStatus,
)


class InMemorySourceScanAdapter:
    """Deterministic source scan fixture for control-plane tests."""

    def close(self) -> None:
        pass

    def __init__(self, scan_results: tuple[SourceScanResult, ...]) -> None:
        self._scan_results = tuple(scan_results)
        if not self._scan_results or any(
            not isinstance(scan_result, SourceScanResult)
            for scan_result in self._scan_results
        ):
            raise TypeError(
                "in-memory scan result fixture requires SourceScanResult values"
            )
        identity = self._scan_results[0].identity
        if any(scan_result.identity != identity for scan_result in self._scan_results):
            raise ValueError("in-memory scan results must share source scan identity")
        self._record_count = sum(result.records for result in self._scan_results)

    @staticmethod
    def cursor(position: int) -> OpaqueSourceCursor:
        if type(position) is not int or position < 0:
            raise ValueError("in-memory position must be non-negative")
        return OpaqueSourceCursor.from_payload(position.to_bytes(8, "big"))

    def _position(self, cursor: OpaqueSourceCursor) -> int:
        if not isinstance(cursor, OpaqueSourceCursor) or len(cursor.payload) != 8:
            raise ValueError("in-memory cursor must encode a record position")
        position = int.from_bytes(cursor.payload, "big")
        if position > self._record_count:
            raise ValueError("in-memory cursor exceeds source records")
        return position

    def advance_cursor(
        self, cursor: OpaqueSourceCursor, records: int
    ) -> OpaqueSourceCursor:
        position = self._position(cursor)
        if (
            type(records) is not int
            or not 0 <= records <= self._record_count - position
        ):
            raise ValueError("consumed records exceed in-memory source")
        return self.cursor(position + records)

    def scan_next(
        self, cursor: OpaqueSourceCursor, budget: SourceScanBudget
    ) -> SourceScanResult:
        position = self._position(cursor)
        if not isinstance(budget, SourceScanBudget):
            raise TypeError("in-memory scan result budget must be typed")
        start = 0
        for result in self._scan_results:
            end = start + result.records
            if position < end:
                skip = position - start
                units = []
                cursors = []
                records = 0
                for unit in result.units:
                    if skip >= unit.cardinality:
                        skip -= unit.cardinality
                        continue
                    count = min(unit.cardinality - skip, budget.max_records - records)
                    units.append(unit.slice(skip, count))
                    cursors.append(self.cursor(position + records))
                    records += count
                    skip = 0
                    if records == budget.max_records or len(units) == budget.max_units:
                        break
                return replace(
                    result,
                    units=tuple(units),
                    unit_cursors=tuple(cursors),
                    records=records,
                    next_cursor=self.cursor(position + records),
                    status=SourceScanStatus.from_scan(
                        records=records,
                        exhausted=position + records == end,
                        finite=result.status is SourceScanStatus.TERMINAL,
                    ),
                )
            start = end
        last = self._scan_results[-1]
        return replace(
            last,
            units=(),
            unit_cursors=(),
            records=0,
            next_cursor=cursor,
            status=SourceScanStatus.from_scan(
                records=0,
                exhausted=True,
                finite=last.status is SourceScanStatus.TERMINAL,
            ),
        )
