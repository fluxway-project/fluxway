# 计算图可视化

[英文版](graph_visualization.md)

Fluxway 将编译后的计算图导出到 Google Model Explorer。
导出不会运行计算图、打开浏览器或启动服务器，只生成一个 `model_explorer.json` 文件。
该文件用于检查计算图，不能用来重建作业。

## 安装并导出完整示例

安装可选的查看器依赖：

```bash
python -m pip install "fluxway[visualization]"
```

从仓库根目录运行下例。输出目录 `tmp/graph-example` 必须尚不存在。
代码构建一条张量依赖和一条仅控制顺序的依赖，再检查导出的两个视图。
不需要 Ray 集群或 GPU。

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

`visualize_graph` 读取计算图的描述及其已有的 `CompiledGraph`，包括算子、端口、依赖、
传输编码、逻辑张量标识、函数标识、资源声明和 `AsyncPolicy`。
它不会从 JSON 重建这些事实，运行时仍传输原生 Python 对象。

目录已经存在时会抛出 `FileExistsError`。确实需要替换之前的 Fluxway 导出时，
显式传入 `overwrite=True`。Fluxway 先检查该目录确实是导出包，再原子替换。
目录交换失败时会恢复原导出，并继续抛出异常。

## 本地查看与 SSH 访问

`view_graph(bundle)` 会启动 Model Explorer 服务器，并阻塞到服务器停止。
没有桌面环境的机器可以使用下列函数，参数是已有导出目录。
服务器仅监听本机回环地址，不打开浏览器。

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

在保存导出包的机器上调用 `serve_graph("tmp/graph-example")`。
在自己的机器上，将平时使用的 SSH 目标作为下面函数的参数：

```bash
forward_graph() {
    ssh -N -L 8080:127.0.0.1:8080 "$1"
}
```

保持终端打开，然后访问 `http://127.0.0.1:8080`。两端的 8080 端口都需要可用。
如果希望复制文件后直接查看，不保持远程服务器运行，可以使用下列函数。
三个参数依次为 SSH 目标、远程导出目录的绝对路径和本地 JSON 输出路径；本地机器需要安装 Model Explorer。

```bash
copy_and_view_graph() {
    scp "$1:$2/model_explorer.json" "$3" &&
        model-explorer "$3"
}
```

## 阅读两个视图

| 视图 | 显示内容 |
| --- | --- |
| `Dependencies` | 每条编译依赖对应一条边，边的元数据列出全部张量传输 |
| `Potential overlap` | 同一步内的因果布局，按依赖层级和调度通道分组 |

两个视图来自同一份编译后的计算图。仅控制顺序的依赖没有张量传输，运行时以仅含头部的帧表示。
示例中，`produce` 到 `consume` 传输 `values`；`consume` 到 `finish` 只规定先后顺序。

`Potential overlap` 的淡色原生边表示同一步内的因果关系。
青绿色 `#007B83` 覆盖边表示 `AsyncPolicy` 中跨执行步的阶段约束，
这些约束不改变同一步内的依赖层级。`Dependencies` 视图没有覆盖边。

方框颜色表示结构角色：绿色为源，蓝色为转换，琥珀色为终点，紫色为依赖环成员，灰色为孤立算子。
Model Explorer 不支持虚线或点线边。

## 检查策略并关联执行轨迹

SPMD 对应命名空间，算子对应节点。选中节点即可查看设备网格、资源范围、函数、端口、依赖层级和调度策略。
Model Explorer 可能折叠只含一个算子的命名空间；需要显示每个 SPMD 时，开启保留单子节点层级的设置。

| 字段 | 用途 |
| --- | --- |
| `graph_id`、`graph_generation`、`spmd_id`、`op_id`、`function_id` | 标识编译后的计算图及可调用对象 |
| `input_signature_id`、`output_signature_id`、`dependency_code` | 标识端口和依赖 |
| `async_mode`、`async_depth`、`async_commit`、`async_max_staleness` | 描述并发限制 |
| `state_access`、`state_resource` | 描述状态串行化规则 |
| `schedule_kind`、`schedule_base_priority`、`schedule_phase`、`schedule_lane` | 描述调度选择 |
| `schedule_phase_order`、`schedule_phase_dependencies`、`schedule_phase_priorities` | 描述阶段关系 |
| `schedule_lane_order`、`schedule_lane_priorities` | 描述通道顺序和优先级 |

这些字段来自编译后的 `AsyncPolicy`，可视化模块不另建一份调度定义。
工作进程与导出共用 `compile_phase_schedule` 的语义。
可以在 [Perfetto 执行轨迹](user_profiler_zh.md)中搜索相同的计算图、SPMD、算子和函数标识，
对照调度规则与实际执行。

静态视图只表达可能的并发关系。实际调用、等待和关闭顺序见[算子执行时序](worker_process_operators_zh.md)；记录采集与文件写出顺序见[性能分析时序](user_profiler_zh.md)。`visualize_graph()` 同步写出静态文件，`view_graph()` 阻塞运行查看服务器，二者都不启动图执行。

## 理解重叠分组与规模边界

Fluxway 将计算图依赖与同一步内的阶段关系合并，折叠因果环，
再在得到的有向无环图上计算最长依赖层级。同层级的算子组成一组，组内按调度通道划分，
通道内再保留 SPMD。跨执行步关系仍作为覆盖边显示。

分组只表示成员可能同时就绪。资源竞争、条件、执行步和反压仍可能使它们串行运行。
实际排队、传输和设备耗时需要查看执行轨迹。

连通分量和依赖层级算法采用迭代实现，支持超过一万个算子的导出。
浏览器布局有独立开销，稠密图的显示时间仍可能明显长于导出时间。
