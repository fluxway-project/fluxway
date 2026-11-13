import json
import builtins
from pathlib import Path
from typing import Any

import pytest

import fluxway as fw
from fluxway.common.decorators import operator, spmd
from fluxway.common.operator_policy import AsyncPolicy, SchedulePolicy


_REP = {"type": "static", "axis": []}
_PHASE_ORDER = ("before", "after")
_PHASE_DEPENDENCIES = (
    ("before", "after", 0),
    ("after", "before", 1),
)
_LANE_ORDER = ("compute", "data")


def _condition_factory(minimum: int):
    def condition(value: int) -> bool:
        return value >= minimum

    return condition


_CONDITION = _condition_factory(0)


def _phase_policy(phase: str, lane: str) -> AsyncPolicy:
    return AsyncPolicy.pipelined(
        depth=2,
        schedule=SchedulePolicy.phase_graph(
            phase=phase,
            phase_order=_PHASE_ORDER,
            dependencies=_PHASE_DEPENDENCIES,
            lane=lane,
            lane_order=_LANE_ORDER,
            lane_priorities={"compute": 1000, "data": 100},
            phase_priorities={"before": 20, "after": 10},
        ),
    )


@spmd(
    resource_profile=fw.ResourceProfile(base={"CPU": 1}),
    mesh_axis_names=("data",),
    mesh_axis_shapes=(1,),
)
class _VisualizationSPMD:
    @operator(name="produce", output_sharding={"left": _REP, "right": _REP})
    def produce(self) -> dict[str, Any]:
        return {"left": 1, "right": 2}

    @operator(name="consume", input_sharding={"first": _REP, "second": _REP})
    def consume(self, first: Any, second: Any) -> None:
        del first, second

    @operator(name="before", async_policy=_phase_policy("before", "compute"))
    def before(self) -> None:
        return None

    @operator(name="after", async_policy=_phase_policy("after", "data"))
    def after(self) -> None:
        return None


def _build_graph():
    builder = fw.Builder()
    builder.add_node("source", _VisualizationSPMD, description="producer")
    builder.add_node(
        "target",
        _VisualizationSPMD,
        description="consumer",
        resource_profile=fw.ResourceProfile(
            base={"CPU": 2},
            scale=fw.ResourceScale(step={"memory": 1024}, max_steps=2),
        ),
    )
    builder.add_func_edge(
        "source",
        "produce",
        "target",
        "consume",
        condition_fn=_CONDITION,
        tensor_map={"left": "first", "right": "second"},
    )
    builder.add_func_edge(
        "target",
        "before",
        "target",
        "after",
        local_only=True,
    )
    return builder.build("visualization-test")


def test_visualize_graph_projects_graph_and_compiled_graph(tmp_path):
    source_graph = _build_graph()
    output = fw.visualize_graph(source_graph, tmp_path / "bundle")

    assert output == tmp_path / "bundle"
    assert [path.name for path in output.iterdir()] == ["model_explorer.json"]

    projection = json.loads((output / "model_explorer.json").read_text())
    dependency_graph, overlap_graph = projection["graphs"]
    plan = source_graph.compiled_graph
    assert plan is not None

    assert dependency_graph["id"] == "Dependencies"
    assert overlap_graph["id"] == "Potential overlap"
    assert dependency_graph["groupNodeAttributes"][""] == {
        "format": "fluxway.model_explorer.v1",
        "graph_name": source_graph.name,
        "graph_id": plan.graph_id,
        "graph_generation": str(plan.generation),
        "view": "dependencies",
    }
    assert overlap_graph["groupNodeAttributes"][""]["view"] == "potential_overlap"
    assert len(dependency_graph["nodes"]) == 4

    target = next(
        node for node in dependency_graph["nodes"] if node["label"] == "consume"
    )
    attrs = {item["key"]: item["value"] for item in target["attrs"]}
    assert attrs["spmd_id"] == "target"
    assert attrs["graph_generation"] == str(plan.generation)
    assert attrs["op_id"] == "consume"
    assert attrs["function_id"].endswith("._VisualizationSPMD.consume")
    compiled_consume = plan.spmd("target").operator("consume")
    assert attrs["input_signature_id"] == compiled_consume.input_signature_id
    assert attrs["async_mode"] == compiled_consume.spec.async_policy.mode
    assert attrs["schedule_kind"] == compiled_consume.spec.async_policy.schedule.kind
    assert attrs["schedule_phase_order"] == "[]"
    assert attrs["schedule_lane_order"] == "[]"
    assert len(target["incomingEdges"]) == 1
    input_metadata = {
        item["id"]: {attr["key"]: attr["value"] for attr in item["attrs"]}
        for item in target["inputsMetadata"]
    }
    assert len(input_metadata) == 1
    transfer_metadata = next(iter(input_metadata.values()))
    assert transfer_metadata["__tensor_tag"] == "2 tensors"
    assert json.loads(transfer_metadata["ports"]) == ["first", "second"]
    assert json.loads(transfer_metadata["transfers"]) == [
        {
            "logical_tuuid": 1,
            "source_port": "left",
            "target_port": "first",
            "transfer_code": 1,
        },
        {
            "logical_tuuid": 2,
            "source_port": "right",
            "target_port": "second",
            "transfer_code": 2,
        },
    ]
    assert {metadata["condition"] for metadata in input_metadata.values()} == {
        f"{__name__}._condition_factory.<locals>.condition"
    }
    assert plan.dependencies[0].spec.condition is _CONDITION

    before = next(
        node for node in dependency_graph["nodes"] if node["label"] == "before"
    )
    after = next(node for node in dependency_graph["nodes"] if node["label"] == "after")
    assert len(after["incomingEdges"]) == 1
    assert after["incomingEdges"][0]["sourceNodeId"] == before["id"]

    before_attrs = {item["key"]: item["value"] for item in before["attrs"]}
    after_attrs = {item["key"]: item["value"] for item in after["attrs"]}
    assert before_attrs["dependency_rank"] == "0"
    assert before_attrs["potential_overlap_group"] == "rank:0"
    assert after_attrs["dependency_rank"] == "1"
    assert after_attrs["potential_overlap_group"] == "rank:1"
    assert before_attrs["schedule_phase"] == "before"
    assert before_attrs["schedule_phase_order"] == '["before", "after"]'
    assert before_attrs["schedule_phase_dependencies"] == (
        '[["before", "after", 0], ["after", "before", 1]]'
    )
    assert before_attrs["schedule_lane"] == "compute"
    assert before_attrs["schedule_lane_order"] == '["compute", "data"]'
    assert json.loads(before_attrs["schedule_lane_priorities"]) == {
        "compute": 1000,
        "data": 100,
    }
    assert json.loads(before_attrs["spmd_resources_minimum"]) == {"CPU": "2"}
    assert json.loads(before_attrs["spmd_resources_maximum"]) == {
        "CPU": "2",
        "memory": "2048",
    }
    assert before["style"]["backgroundColor"] != after["style"]["backgroundColor"]

    assert dependency_graph["tasksData"]["edgeOverlaysDataListLeftPane"] == []
    assert dependency_graph["tasksData"]["edgeOverlaysDataListRightPane"] == []

    overlap_native_edges = [
        (edge["sourceNodeId"], node["id"], edge)
        for node in overlap_graph["nodes"]
        for edge in node["incomingEdges"]
    ]
    assert len(overlap_native_edges) == 2
    assert len({(source, target) for source, target, _ in overlap_native_edges}) == 2
    assert all(
        edge["sourceNodeOutputId"] == "0" and edge["targetNodeInputId"] == "0"
        for _, _, edge in overlap_native_edges
    )
    assert all(
        not node["inputsMetadata"] and not node["outputsMetadata"]
        for node in overlap_graph["nodes"]
    )
    relation_overlays = overlap_graph["tasksData"]["edgeOverlaysDataListLeftPane"][0]
    overlays = {overlay["name"]: overlay for overlay in relation_overlays["overlays"]}
    assert set(overlays) == {"Cross-step phase"}
    assert overlays["Cross-step phase"]["edgeColor"] == "#007B83"
    assert overlays["Cross-step phase"]["edgeWidth"] == 1
    assert len(overlays["Cross-step phase"]["edges"]) == 1
    cross_step_pairs = {
        (edge["sourceNodeId"], edge["targetNodeId"])
        for edge in overlays["Cross-step phase"]["edges"]
    }
    assert cross_step_pairs.isdisjoint(
        (source, target) for source, target, _ in overlap_native_edges
    )
    overlay_pairs = [
        (edge["sourceNodeId"], edge["targetNodeId"])
        for overlay in overlays.values()
        for edge in overlay["edges"]
    ]
    assert len(overlay_pairs) == len(set(overlay_pairs))
    assert overlap_graph["tasksData"]["edgeOverlaysDataListRightPane"] == []
    assert dependency_graph["layoutConfigs"] == {
        "edgeSep": 24,
        "keepLayersWithASingleChild": None,
        "nodeSep": 36,
        "rankSep": 96,
    }
    assert dependency_graph["groupNodeConfigs"]
    assert dependency_graph["groupNodeConfigs"][0] == {
        "backgroundColor": None,
        "borderColor": None,
        "layoutDirection": 1,
        "namespaceRegex": "^$",
        "textColor": None,
    }
    overlap_namespaces = {
        node["label"]: node["namespace"] for node in overlap_graph["nodes"]
    }
    assert overlap_namespaces["before"].startswith("rank:0/lane:compute/")
    assert overlap_namespaces["after"].startswith("rank:1/lane:data/")


def test_visualize_graph_reads_resources_from_compiled_declaration(tmp_path):
    source_graph = _build_graph()
    source_graph.nodes["target"].resource_profile = fw.ResourceProfile(base={"CPU": 99})

    output = fw.visualize_graph(source_graph, tmp_path / "bundle")
    projection = json.loads((output / "model_explorer.json").read_text())
    target = next(
        node for node in projection["graphs"][0]["nodes"] if node["label"] == "consume"
    )
    attrs = {item["key"]: item["value"] for item in target["attrs"]}

    assert json.loads(attrs["spmd_resources_minimum"]) == {"CPU": "2"}
    group = projection["graphs"][0]["groupNodeAttributes"]["target"]
    assert json.loads(group["resources_minimum"]) == {"CPU": "2"}


def test_visualize_graph_preserves_special_characters_and_empty_description(tmp_path):
    builder = fw.Builder()
    builder.add_node('source/<&"', _VisualizationSPMD, description="")
    builder.add_node("target", _VisualizationSPMD)
    builder.add_func_edge(
        'source/<&"',
        "produce",
        "target",
        "consume",
        tensor_map={"left": "first", "right": "second"},
    )
    graph = builder.build('graph/<&"')

    output = fw.visualize_graph(graph, tmp_path / "bundle")
    projection = json.loads((output / "model_explorer.json").read_text())

    assert projection["label"] == 'graph/<&"'
    source = next(
        node
        for node in projection["graphs"][0]["nodes"]
        if {item["key"]: item["value"] for item in node["attrs"]}["spmd_id"]
        == 'source/<&"'
    )
    assert source["namespace"] == "source%2F%3C%26%22"


def test_visualize_graph_protects_existing_directories(tmp_path):
    graph = _build_graph()
    destination = tmp_path / "bundle"
    destination.mkdir()

    with pytest.raises(FileExistsError):
        fw.visualize_graph(graph, destination)
    with pytest.raises(ValueError, match="must contain exactly"):
        fw.visualize_graph(graph, destination, overwrite=True)

    destination.rmdir()
    fw.visualize_graph(graph, destination)
    fw.visualize_graph(graph, destination, overwrite=True)

    projection_path = destination / "model_explorer.json"
    projection = json.loads(projection_path.read_text())
    projection["graphs"] = []
    projection_path.write_text(json.dumps(projection))
    with pytest.raises(ValueError, match="requires views"):
        fw.visualize_graph(graph, destination, overwrite=True)


def test_visualize_graph_restores_bundle_when_directory_swap_fails(
    monkeypatch, tmp_path
):
    graph = _build_graph()
    destination = fw.visualize_graph(graph, tmp_path / "bundle")
    original = {path.name: path.read_bytes() for path in destination.iterdir()}
    real_replace = Path.replace

    def fail_new_bundle_replace(path, target):
        if (
            path.parent == destination.parent
            and path.name.startswith(f".{destination.name}.")
            and not path.name.startswith(f".{destination.name}.backup.")
            and Path(target) == destination
        ):
            raise OSError("proof directory swap failure")
        return real_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_new_bundle_replace)
    with pytest.raises(OSError, match="proof directory swap failure"):
        fw.visualize_graph(graph, destination, overwrite=True)

    assert {path.name: path.read_bytes() for path in destination.iterdir()} == original


def test_view_graph_uses_model_explorer_config(monkeypatch, tmp_path):
    bundle = fw.visualize_graph(_build_graph(), tmp_path / "bundle")
    calls = {}

    import model_explorer  # pyrefly: ignore [missing-import]

    monkeypatch.setattr(
        model_explorer,
        "visualize_from_config",
        lambda **kwargs: calls.update(kwargs),
    )

    fw.view_graph(bundle, host="127.0.0.2", port=8090, open_browser=False)

    assert calls["host"] == "127.0.0.2"
    assert calls["port"] == 8090
    assert calls["no_open_in_browser"] is True
    assert calls["config"].model_sources == [
        {
            "url": str((bundle / "model_explorer.json").resolve()),
            "adapterId": "builtin_json",
        }
    ]


def test_visualization_dependency_and_service_failures_are_explicit(
    monkeypatch, tmp_path
):
    graph = _build_graph()
    bundle = fw.visualize_graph(graph, tmp_path / "bundle")
    real_import = builtins.__import__

    def reject_model_explorer(name, *args, **kwargs):
        if name == "model_explorer":
            raise ImportError("proof missing dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_model_explorer)
    with pytest.raises(FileExistsError):
        fw.visualize_graph(graph, bundle)
    with pytest.raises(ImportError, match=r"fluxway\[visualization\]"):
        fw.visualize_graph(graph, tmp_path / "missing-dependency")
    with pytest.raises(ImportError, match=r"fluxway\[visualization\]"):
        fw.visualize_graph(graph, bundle, overwrite=True)

    monkeypatch.setattr(builtins, "__import__", real_import)

    import model_explorer  # pyrefly: ignore [missing-import]

    def fail_service(**kwargs):
        raise RuntimeError("service failed")

    monkeypatch.setattr(model_explorer, "visualize_from_config", fail_service)
    with pytest.raises(RuntimeError, match="service failed"):
        fw.view_graph(bundle, open_browser=False)


def test_visualize_graph_rejects_graph_without_compiled_graph(tmp_path):
    with pytest.raises(RuntimeError, match="no compiled graph"):
        fw.visualize_graph(fw.Graph(name="uncompiled"), tmp_path / "bundle")
