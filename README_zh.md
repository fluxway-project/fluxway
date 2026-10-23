# Fluxway

[English](README.md)

Fluxway 是面向异构集群的分布式计算编排库。它将 Python 算子编译为计算图，让不同 SPMD 工作进程组通过异步数据面协作，分别配置读取、存储和计算资源。

- 使用函数、类和装饰器构图，在算子内接入 JAX、PyTorch、DLPack 或原生扩展。
- 按资源声明、节点亲和约束和有序优化目标放置工作进程。
- 通过流水深度、状态访问策略、阶段依赖和背压约束并发执行。
- 协调来源租约、整图排空、检查点发布与新执行恢复。
- 导出编译图和性能轨迹，核对声明的依赖与实际执行。

[推荐训练示例](fluxway/example/recommendation/README_zh.md) 展示读取进程、CPU 稀疏表与数据并行计算的组合。并发与复制开销取决于算子、后端和部署，需按工作负载测量。公开入口提交编译图；算子内部可使用即时计算。

## 文档

从[文档总览](docs/README_zh.md)开始。复杂流程的时序图标明并发调用、屏障、取消和提交边界。

| 主题 | 文档 |
| --- | --- |
| 架构与执行 | [架构](docs/ARCHITECTURE_zh.md)、[执行生命周期](docs/execution_lifecycle_zh.md)、[检查点与恢复](docs/checkpoint_zh.md) |
| 数据读取 | [来源协调](docs/source_coordination_zh.md)、[来源任务协议](docs/source_spmd_formal_protocol_zh.md)、[读取接口](docs/base_reader_zh.md) |
| 数据交付 | [传输计划](docs/transfer_planning_zh.md)、[数据面运行时](docs/dataplane_runtime_zh.md)、[一致性协议](docs/coherence_protocol_zh.md) |
| 资源与算子 | [放置配方](docs/placement_recipe_zh.md)、[放置数学](docs/placement_mathematics_zh.md)、[工作进程算子](docs/worker_process_operators_zh.md) |
| 观察与诊断 | [用户性能分析](docs/user_profiler_zh.md)、[计算图可视化](docs/graph_visualization_zh.md) |
| 开发与发布 | [贡献指南](CONTRIBUTING_zh.md)、[版本发布](docs/releasing_zh.md) |

## 查看计算图

安装查看器依赖：

```bash
python -m pip install "fluxway[visualization]"
```

下例需要已构建的 `graph`，输出目录必须尚不存在。导出不启动图执行，查看器会阻塞运行本地服务器。

```python
import fluxway

bundle = fluxway.visualize_graph(graph, "tmp/training-graph")
fluxway.view_graph(bundle)
```

完整构图示例、远程查看和轨迹关联见[计算图可视化](docs/graph_visualization_zh.md)。
