"""Per-worker local step DAG runtime."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import IntEnum
import inspect
from typing import Any, Protocol

from fluxway.spmd.worker.user_callable_executor import (
    USER_CALLABLE_DEFAULT_OPERATOR_LANE,
    UserCallableExecutor,
    UserCallableSpec,
)


class LocalStepNodeKind(IntEnum):
    # CALL invokes the user operator through UserCallableExecutor.
    CALL = 1
    # SEND registers output frame build/split/transport in the worker runtime.
    SEND = 2
    # WAIT assembles mailbox tensors/control state into user operator inputs.
    WAIT = 3
    # BARRIER observes SEND registration, clears local step state, and commits.
    BARRIER = 4
    AWAIT_SEND = 5


@dataclass(frozen=True, slots=True)
class LocalStepPlan:
    node_kinds: tuple[int, ...]
    callable_ids: tuple[str, ...]
    resource_ids: tuple[int, ...]
    dependency_offsets: tuple[int, ...]
    dependencies: tuple[int, ...]
    result_node: int
    _serial_chain: bool = field(init=False, repr=False, compare=False)

    def __init__(
        self,
        node_kinds: Sequence[int],
        callable_ids: Sequence[str],
        resource_ids: Sequence[int],
        dependency_offsets: Sequence[int],
        dependencies: Sequence[int],
        result_node: int,
    ) -> None:
        object.__setattr__(self, "node_kinds", tuple(node_kinds))
        object.__setattr__(self, "callable_ids", tuple(callable_ids))
        object.__setattr__(self, "resource_ids", tuple(resource_ids))
        object.__setattr__(self, "dependency_offsets", tuple(dependency_offsets))
        object.__setattr__(self, "dependencies", tuple(dependencies))
        object.__setattr__(self, "result_node", result_node)
        self.validate()
        object.__setattr__(self, "_serial_chain", self._derive_serial_chain())

    @classmethod
    def single_call(cls, callable_id: str, *, resource_id: int = 0) -> LocalStepPlan:
        if not callable_id:
            raise ValueError("callable_id must be non-empty.")
        return cls(
            node_kinds=(int(LocalStepNodeKind.CALL),),
            callable_ids=(callable_id,),
            resource_ids=(resource_id,),
            dependency_offsets=(0, 0),
            dependencies=(),
            result_node=0,
        )

    def validate(self) -> None:
        node_count = len(self.node_kinds)
        if node_count == 0:
            raise ValueError("LocalStepPlan must contain at least one node.")
        if len(self.callable_ids) != node_count:
            raise ValueError("callable_ids length must match node_kinds length.")
        if len(self.resource_ids) != node_count:
            raise ValueError("resource_ids length must match node_kinds length.")
        if len(self.dependency_offsets) != node_count + 1:
            raise ValueError("dependency_offsets length must be node_count + 1.")
        if self.dependency_offsets[0] != 0:
            raise ValueError("dependency_offsets must start at 0.")
        if self.dependency_offsets[-1] != len(self.dependencies):
            raise ValueError("dependency_offsets must end at dependency count.")
        previous = 0
        for offset in self.dependency_offsets:
            if offset < previous:
                raise ValueError("dependency_offsets must be monotonically increasing.")
            previous = offset
        if self.result_node < 0 or self.result_node >= node_count:
            raise ValueError(
                f"result_node={self.result_node} is outside node count {node_count}."
            )
        for index, raw_kind in enumerate(self.node_kinds):
            try:
                LocalStepNodeKind(raw_kind)
            except ValueError as exc:
                raise ValueError(
                    f"node {index} has unsupported kind {raw_kind}."
                ) from exc
        for dep in self.dependencies:
            if dep < 0 or dep >= node_count:
                raise ValueError(
                    f"dependency {dep} is outside node count {node_count}."
                )

    def dependencies_for(self, node_index: int) -> tuple[int, ...]:
        begin = self.dependency_offsets[node_index]
        end = self.dependency_offsets[node_index + 1]
        return self.dependencies[begin:end]

    def _derive_serial_chain(self) -> bool:
        """Determines whether the execution DAG is a strictly serial pipeline chain.

        A plan is a serial chain if node 0 has no dependencies, and every subsequent
        node i depends on its immediate predecessor i - 1 with all dependencies ordered
        strictly before i. When True, LocalStepPlanExecutor can bypass asyncio Task/Event
        scheduling and execute nodes sequentially in a direct loop with zero overhead.
        """
        return all(
            (
                self.dependencies_for(index) == ()
                if index == 0
                else all(dep < index for dep in self.dependencies_for(index))
                and index - 1 in self.dependencies_for(index)
            )
            for index in range(len(self.node_kinds))
        )


@dataclass(frozen=True, slots=True)
class LocalStepContext:
    op_name: str = ""
    step: int = 0


@dataclass(frozen=True, slots=True)
class LocalNodeSpec:
    callable_id: str
    fn: Callable[..., Any] | None = None
    user_callable: UserCallableSpec | None = None
    user_callable_factory: Callable[[tuple[Any, ...]], UserCallableSpec] | None = None
    user_runner: Callable[[tuple[Any, ...]], Any] | None = None

    async def run(self, dependency_results: tuple[Any, ...]) -> Any:
        if self.fn is None:
            raise TypeError(f"Local node {self.callable_id!r} has no runtime callable.")
        result = self.fn(*dependency_results)
        if inspect.isawaitable(result):
            return await result
        return result

    def user_spec_for(
        self, dependency_results: tuple[Any, ...]
    ) -> UserCallableSpec | None:
        if self.user_callable_factory is not None:
            return self.user_callable_factory(dependency_results)
        return self.user_callable


class LocalCallableRegistry(Protocol):
    def spec_for(
        self,
        kind: LocalStepNodeKind,
        callable_id: str,
        resource_id: int,
        step_context: LocalStepContext,
    ) -> LocalNodeSpec: ...


class MappingCallableRegistry:
    def __init__(self, callables: Mapping[str, Any]):
        self._callables = dict(callables)

    def spec_for(
        self,
        kind: LocalStepNodeKind,
        callable_id: str,
        resource_id: int,
        step_context: LocalStepContext = LocalStepContext(),
    ) -> LocalNodeSpec:
        del step_context
        fn = self._callables.get(callable_id)
        if fn is None:
            raise KeyError(f"Unknown local callable_id {callable_id!r}.")
        user_factory = None
        if kind is LocalStepNodeKind.CALL:
            user_factory = self._user_callable_factory(callable_id, fn, resource_id)
        return LocalNodeSpec(
            callable_id=callable_id,
            fn=fn,
            user_callable_factory=user_factory,
        )

    def _user_callable_factory(
        self,
        callable_id: str,
        fn: Callable[..., Any],
        resource_id: int,
    ) -> Callable[[tuple[Any, ...]], UserCallableSpec]:
        operator_is_coroutine_function = inspect.iscoroutinefunction(fn)

        def user_callable_factory(
            dependency_results: tuple[Any, ...],
        ) -> UserCallableSpec:
            def call_with_dependencies() -> Any:
                return fn(*dependency_results)

            return UserCallableSpec(
                callable_id=callable_id,
                fn=call_with_dependencies,
                resource_id=resource_id,
                operator_lane=USER_CALLABLE_DEFAULT_OPERATOR_LANE,
                operator_is_coroutine_function=operator_is_coroutine_function,
            )

        return user_callable_factory


class LocalStepPlanExecutor:
    def __init__(self, user_callable_executor: UserCallableExecutor) -> None:
        self._active_tasks: set[asyncio.Task[Any]] = set()
        self._active_executions: set[asyncio.Task[Any]] = set()
        self._user_callable_executor = user_callable_executor
        self._resource_locks: dict[int, asyncio.Lock] = {}

    async def execute(
        self,
        plan: LocalStepPlan,
        registry: LocalCallableRegistry,
        step_context: LocalStepContext = LocalStepContext(),
    ) -> Any:
        node_count = len(plan.node_kinds)
        node_results: list[Any] = [None] * node_count
        execution_task = asyncio.current_task()
        if execution_task is None:
            raise RuntimeError("LocalStepPlanExecutor requires an asyncio task.")
        self._active_executions.add(execution_task)
        try:
            if plan._serial_chain:
                for node_index in range(node_count):
                    await self._execute_ready_node(
                        plan,
                        registry,
                        step_context,
                        node_results,
                        node_index,
                    )
                return node_results[plan.result_node]
            return await self._execute_dag(
                plan,
                registry,
                step_context,
                node_results,
            )
        finally:
            self._active_executions.discard(execution_task)

    async def _execute_dag(
        self,
        plan: LocalStepPlan,
        registry: LocalCallableRegistry,
        step_context: LocalStepContext,
        node_results: list[Any],
    ) -> Any:
        node_count = len(plan.node_kinds)
        done_events = tuple(asyncio.Event() for _ in range(node_count))

        async def run_node(node_index: int) -> None:
            dependency_indexes = plan.dependencies_for(node_index)
            for dependency in dependency_indexes:
                await done_events[dependency].wait()
            await self._execute_ready_node(
                plan,
                registry,
                step_context,
                node_results,
                node_index,
            )
            done_events[node_index].set()

        tasks = [asyncio.create_task(run_node(index)) for index in range(node_count)]
        self._active_tasks.update(tasks)
        try:
            await asyncio.gather(*tasks)
        except Exception:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        finally:
            for task in tasks:
                self._active_tasks.discard(task)
        return node_results[plan.result_node]

    async def _execute_ready_node(
        self,
        plan: LocalStepPlan,
        registry: LocalCallableRegistry,
        step_context: LocalStepContext,
        node_results: list[Any],
        node_index: int,
    ) -> None:
        dependency_indexes = plan.dependencies_for(node_index)
        kind = LocalStepNodeKind(plan.node_kinds[node_index])
        resource_id = plan.resource_ids[node_index]
        spec = registry.spec_for(
            kind,
            plan.callable_ids[node_index],
            resource_id,
            step_context,
        )
        dependency_results = tuple(node_results[index] for index in dependency_indexes)
        lock = self._lock_for(resource_id)
        if lock is None:
            node_results[node_index] = await self._execute_node(
                kind,
                spec,
                dependency_results,
            )
            return
        async with lock:
            node_results[node_index] = await self._execute_node(
                kind,
                spec,
                dependency_results,
            )

    def _lock_for(self, resource_id: int) -> asyncio.Lock | None:
        if resource_id < 0:
            return None
        lock = self._resource_locks.get(resource_id)
        if lock is None:
            lock = asyncio.Lock()
            self._resource_locks[resource_id] = lock
        return lock

    async def _execute_node(
        self,
        kind: LocalStepNodeKind,
        spec: LocalNodeSpec,
        dependency_results: tuple[Any, ...],
    ) -> Any:
        if kind is LocalStepNodeKind.CALL:
            if spec.user_runner is not None:
                result = spec.user_runner(dependency_results)
                if inspect.isawaitable(result):
                    return await result
                return result
            user_callable = spec.user_spec_for(dependency_results)
            if user_callable is None:
                raise TypeError(
                    f"CALL node {spec.callable_id!r} must provide UserCallableSpec."
                )
            return await self._user_callable_executor.execute(user_callable)
        if kind in (
            LocalStepNodeKind.WAIT,
            LocalStepNodeKind.SEND,
            LocalStepNodeKind.BARRIER,
            LocalStepNodeKind.AWAIT_SEND,
        ):
            # Runtime nodes fail by propagating their callable exception. SEND
            # returns a worker-owned token; BARRIER observes token registration,
            # while delivery failures poison the worker and are fenced by drain.
            return await spec.run(dependency_results)
        raise ValueError(f"unsupported LocalStepPlan node kind {kind!r}.")

    async def drain_user_callables(self) -> None:
        await self._user_callable_executor.drain()

    @property
    def user_callable_executor(self) -> UserCallableExecutor:
        return self._user_callable_executor

    async def drain(self) -> None:
        while True:
            current = asyncio.current_task()
            tasks = [
                task
                for task in (*self._active_tasks, *self._active_executions)
                if task is not current and not task.done()
            ]
            if not tasks:
                return
            await asyncio.gather(*tasks)
