from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from fluxway.spmd.worker.local_plan import (
    LocalStepNodeKind,
    LocalStepPlan,
    LocalStepPlanExecutor,
    MappingCallableRegistry,
)
from fluxway.spmd.worker.runtime import WorkerProcessCallRuntime
from fluxway.spmd.worker.user_callable_executor import UserCallableExecutor


@dataclass(frozen=True)
class BenchConfig:
    macro_steps: int
    microsteps: int
    stage0_s: float
    stage1_s: float
    send_s: float
    optimizer_s: float
    probe_s: float


async def event_loop_probe(stop: asyncio.Event, period_s: float) -> dict[str, float]:
    lags_ms: list[float] = []
    expected = time.perf_counter() + period_s
    while not stop.is_set():
        await asyncio.sleep(period_s)
        now = time.perf_counter()
        lags_ms.append(max(0.0, now - expected) * 1000.0)
        expected = now + period_s
    if not lags_ms:
        raise RuntimeError("event loop probe collected no samples.")
    sorted_lags = sorted(lags_ms)
    p95_index = min(len(sorted_lags) - 1, int(len(sorted_lags) * 0.95))
    return {
        "probe_count": float(len(lags_ms)),
        "event_loop_lag_p50_ms": statistics.median(lags_ms),
        "event_loop_lag_p95_ms": sorted_lags[p95_index],
        "event_loop_lag_max_ms": max(lags_ms),
    }


def build_micro_pipeline_plan(
    config: BenchConfig, *, lanes: int, serial: bool
) -> LocalStepPlan:
    node_kinds: list[int] = []
    callable_ids: list[str] = []
    resource_ids: list[int] = []
    dependencies: list[int] = []
    dependency_offsets: list[int] = [0]
    last_tail: int | None = None

    for microstep in range(config.microsteps):
        lane = microstep % lanes
        local_nodes: list[int] = []
        for suffix, resource_id in (
            ("fwd0", lane * 2),
            ("send_activation", -1),
            ("fwd1_bwd1", lane * 2 + 1),
            ("send_gradient", -1),
            ("bwd0", lane * 2),
        ):
            node_index = len(node_kinds)
            node_kinds.append(int(LocalStepNodeKind.CALL))
            callable_ids.append(f"m{microstep}:{suffix}")
            resource_ids.append(resource_id)
            local_deps: list[int] = []
            if local_nodes:
                local_deps.append(local_nodes[-1])
            if serial and not local_nodes and last_tail is not None:
                local_deps.append(last_tail)
            dependencies.extend(local_deps)
            dependency_offsets.append(len(dependencies))
            local_nodes.append(node_index)
        last_tail = local_nodes[-1]

    return LocalStepPlan(
        node_kinds=tuple(node_kinds),
        callable_ids=tuple(callable_ids),
        resource_ids=tuple(resource_ids),
        dependency_offsets=tuple(dependency_offsets),
        dependencies=tuple(dependencies),
        result_node=len(node_kinds) - 1,
    )


def build_registry(config: BenchConfig) -> MappingCallableRegistry:
    callables: dict[str, Any] = {}
    for microstep in range(config.microsteps):
        for suffix, delay in (
            ("fwd0", config.stage0_s),
            ("send_activation", config.send_s),
            ("fwd1_bwd1", config.stage1_s),
            ("send_gradient", config.send_s),
            ("bwd0", config.stage0_s),
        ):
            checksum = (microstep + 1) * 100 + len(suffix)

            def _make_run(delay_s: float, value: int) -> Callable[..., Any]:
                async def run(*_args: Any) -> int:
                    await asyncio.sleep(delay_s)
                    return value

                return run

            callables[f"m{microstep}:{suffix}"] = _make_run(delay, checksum)
    return MappingCallableRegistry(callables)


async def run_mode(
    config: BenchConfig, *, mode: str, lanes: int, serial: bool
) -> dict[str, Any]:
    executor = LocalStepPlanExecutor(
        UserCallableExecutor(
            worker_process_runtime=WorkerProcessCallRuntime(max_workers=1),
        )
    )
    plan = build_micro_pipeline_plan(config, lanes=lanes, serial=serial)
    registry = build_registry(config)
    stop = asyncio.Event()
    probe_task = asyncio.create_task(event_loop_probe(stop, config.probe_s))
    start = time.perf_counter()

    checksum = 0
    for _ in range(config.macro_steps):
        result = await executor.execute(plan, registry)
        checksum += int(result)
        await asyncio.sleep(config.optimizer_s)

    elapsed_s = time.perf_counter() - start
    stop.set()
    responsiveness = await probe_task
    return {
        "mode": mode,
        "elapsed_s": elapsed_s,
        "macro_steps_per_second": config.macro_steps / elapsed_s,
        "node_count": len(plan.node_kinds),
        "checksum": checksum,
        **responsiveness,
    }


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    config = BenchConfig(
        macro_steps=args.macro_steps,
        microsteps=args.microsteps,
        stage0_s=args.stage0_s,
        stage1_s=args.stage1_s,
        send_s=args.send_s,
        optimizer_s=args.optimizer_s,
        probe_s=args.probe_s,
    )
    serial = await run_mode(config, mode="serial", lanes=1, serial=True)
    single = await run_mode(config, mode="single_pipeline", lanes=1, serial=False)
    dual = await run_mode(config, mode="dual_pipeline", lanes=2, serial=False)
    return {
        "config": asdict(config),
        "results": [serial, single, dual],
        "single_pipeline_speedup_vs_serial": (
            single["macro_steps_per_second"] / serial["macro_steps_per_second"]
        ),
        "dual_pipeline_speedup_vs_single": (
            dual["macro_steps_per_second"] / single["macro_steps_per_second"]
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--macro-steps", type=int, default=4)
    parser.add_argument("--microsteps", type=int, default=8)
    parser.add_argument("--stage0-s", type=float, default=0.01)
    parser.add_argument("--stage1-s", type=float, default=0.014)
    parser.add_argument("--send-s", type=float, default=0.008)
    parser.add_argument("--optimizer-s", type=float, default=0.006)
    parser.add_argument("--probe-s", type=float, default=0.001)
    return parser.parse_args()


def main() -> None:
    result = asyncio.run(run_benchmark(parse_args()))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
