# Fluxway 文档

英文版见 [README.md](README.md)。

先读总体架构，再按需要选择机制说明。每篇文档用简短说明配合代码，交代预期结果和适用范围。

## 运行示例

使用 Linux 和 Python 3.12 或 3.13，在当前环境安装 Fluxway：

```bash
python -m pip install fluxway
```

仅构图、配置策略或演示本地协议的例子不启动集群。接收编译图或运行时会话的函数属于接入示例，正文会说明这些参数由谁提供。
启动执行前先初始化 Ray；参与节点需要兼容的依赖，恢复时还要能访问检查点仓库。
可选依赖和设备要求见对应文档。

## 文档入口

| 主题 | 中文文档 | 英文文档 |
| --- | --- | --- |
| 总体架构 | [ARCHITECTURE_zh.md](ARCHITECTURE_zh.md) | [ARCHITECTURE.md](ARCHITECTURE.md) |
| 执行生命周期 | [execution_lifecycle_zh.md](execution_lifecycle_zh.md) | [execution_lifecycle.md](execution_lifecycle.md) |
| 检查点与恢复 | [checkpoint_zh.md](checkpoint_zh.md) | [checkpoint.md](checkpoint.md) |
| 来源协调 | [source_coordination_zh.md](source_coordination_zh.md) | [source_coordination.md](source_coordination.md) |
| 来源任务协议 | [source_spmd_formal_protocol_zh.md](source_spmd_formal_protocol_zh.md) | [source_spmd_formal_protocol.md](source_spmd_formal_protocol.md) |
| 读取接口 | [base_reader_zh.md](base_reader_zh.md) | [base_reader.md](base_reader.md) |
| 传输计划 | [transfer_planning_zh.md](transfer_planning_zh.md) | [transfer_planning.md](transfer_planning.md) |
| 数据面运行时 | [dataplane_runtime_zh.md](dataplane_runtime_zh.md) | [dataplane_runtime.md](dataplane_runtime.md) |
| 一致性协议 | [coherence_protocol_zh.md](coherence_protocol_zh.md) | [coherence_protocol.md](coherence_protocol.md) |
| 放置配方 | [placement_recipe_zh.md](placement_recipe_zh.md) | [placement_recipe.md](placement_recipe.md) |
| 放置数学 | [placement_mathematics_zh.md](placement_mathematics_zh.md) | [placement_mathematics.md](placement_mathematics.md) |
| 工作进程算子 | [worker_process_operators_zh.md](worker_process_operators_zh.md) | [worker_process_operators.md](worker_process_operators.md) |
| 图可视化 | [graph_visualization_zh.md](graph_visualization_zh.md) | [graph_visualization.md](graph_visualization.md) |
| 用户性能分析 | [user_profiler_zh.md](user_profiler_zh.md) | [user_profiler.md](user_profiler.md) |
| 版本发布 | [releasing_zh.md](releasing_zh.md) | [releasing.md](releasing.md) |

如何启动、观察和停止计算图，见[执行生命周期](execution_lifecycle_zh.md)。添加恢复能力前先读[检查点与恢复](checkpoint_zh.md)。编写数据源适配器还需遵守[来源协调](source_coordination_zh.md)和[来源任务协议](source_spmd_formal_protocol_zh.md)。

## 文档范围

这些文档说明 Fluxway 的公开接口与协议边界，内部接入示例会明确标注。本地示例用于解释行为，不能证明集群吞吐、崩溃恢复或模型收敛。具体业务配置、远程目录、凭据和验收结果由应用仓库维护。中英文版本覆盖相同机制，同步维护。

## 时序图阅读约定

时序图使用 Mermaid：实线箭头表示调用或消息提交，虚线箭头表示返回或完成通知；异步调用是否等待，以消息文字和屏障为准。`par` 分支可以并发推进，`alt` 表示互斥路径，`opt` 表示可选路径。纵向顺序表达必要的先后关系，不表示耗时比例；同一角色可以代表一组并发参与者。

| 要核对的顺序 | 文档 |
| --- | --- |
| 启动、恢复、准入、排空与取消 | [执行生命周期](execution_lifecycle_zh.md) |
| 扫描、编译、安装窗口与关闭 | [来源协调](source_coordination_zh.md) |
| 批量领取、乱序确认与连续进度 | [来源任务协议](source_spmd_formal_protocol_zh.md) |
| 异步读取与完整耗尽 | [读取接口](base_reader_zh.md) |
| 快照准备、提交、崩溃协调与读取保护 | [检查点与恢复](checkpoint_zh.md) |
| 计划交接、多目标交付与提交屏障 | [传输计划](transfer_planning_zh.md)、[数据面运行时](dataplane_runtime_zh.md) |
| 共享归还、独占授权、有序退休与输入释放 | [一致性协议](coherence_protocol_zh.md) |
| 放置就绪、线程执行与轨迹导出 | [放置配方](placement_recipe_zh.md)、[工作进程算子](worker_process_operators_zh.md)、[用户性能分析](user_profiler_zh.md) |
| 持续集成、包发布与版本发布 | [版本发布](releasing_zh.md) |
