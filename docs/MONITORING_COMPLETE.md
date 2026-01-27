# Fluxway监控系统完整文档

## 📊 监控指标总览

### 1. 训练指标（Training Metrics）

#### 1.1 基础指标
- **step** (int): 训练步数
  - 获取方式：训练循环传入 → `driver.execute_step(step)` → 框架自动记录
  - 使用示例：
    ```python
    for step in range(num_steps):
        driver.execute_step(step)  # step自动传递给监控系统
    ```

- **epoch** (Optional[int]): 训练轮数
  - 获取方式：通过`record_training_progress(epoch=...)`传入
  - 使用示例：
    ```python
    MonitoringFacade.record_training_progress(step=step, epoch=5, spmd_id="spmd_1")
    ```

- **loss** (Optional[float]): 损失值
  - 获取方式：通过`record_training_progress(loss=...)`传入
  - 使用示例：
    ```python
    # 方式1：从forward结果中提取
    result = executor.execute_forward(step)
    if isinstance(result, dict) and "loss" in result:
        MonitoringFacade.record_training_progress(step=step, loss=result["loss"])
    
    # 方式2：直接传入
    MonitoringFacade.record_training_progress(step=step, loss=0.1234)
    ```

#### 1.2 性能指标（Performance）
- **forward_time_ms** (Optional[float]): 前向传播耗时（毫秒）
  - 获取方式：框架自动记录（`StatefulExecutor.execute_forward()`）
  - 使用示例：
    ```python
    # 框架自动完成，用户无需关心
    executor.execute_forward(step)  # 自动记录forward_time_ms
    ```

- **backward_time_ms** (Optional[float]): 反向传播耗时（毫秒）
  - 获取方式：框架自动记录（`StatefulExecutor.execute_backward()`）
  - 使用示例：
    ```python
    # 框架自动完成，用户无需关心
    executor.execute_backward(step)  # 自动记录backward_time_ms
    ```

- **update_time_ms** (Optional[float]): 参数更新耗时（毫秒）
  - 获取方式：框架自动记录（`StatefulExecutor.execute_parameter_update()`）
  - 使用示例：
    ```python
    # 框架自动完成，用户无需关心
    executor.execute_parameter_update()  # 自动记录update_time_ms
    ```

- **total_step_time_ms** (Optional[float]): 总步耗时（毫秒）
  - 获取方式：框架自动计算（`SPMDDriver.execute_step()`）
  - 使用示例：
    ```python
    # 框架自动完成，用户无需关心
    driver.execute_step(step)  # 自动计算total_step_time_ms
    ```

- **throughput_steps_per_sec** (Optional[float]): 每秒训练步数
  - 获取方式：框架自动计算（基于`total_step_time_ms`）
  - 使用示例：
    ```python
    # 框架自动完成，用户无需关心
    driver.execute_step(step)  # 自动计算throughput_steps_per_sec
    ```

- **throughput_samples_per_sec** (Optional[float]): 每秒处理样本数
  - 获取方式：框架自动计算（需要提供`batch_size`）
  - 使用示例：
    ```python
    MonitoringFacade.record_training_progress(
        step=step, batch_size=32, spmd_id="spmd_1"
    )  # 自动计算throughput_samples_per_sec = batch_size / step_time
    ```

#### 1.3 质量指标（Quality）
- **learning_rate** (Optional[float]): 学习率
  - 获取方式：通过`record_training_progress(metrics={"learning_rate": ...})`传入
  - 使用示例：
    ```python
    MonitoringFacade.record_training_progress(
        step=step,
        metrics={"learning_rate": optimizer.param_groups[0]["lr"]},
        spmd_id="spmd_1"
    )
    ```

- **gradient_norm** (Optional[float]): 梯度范数
  - 获取方式：通过`record_gradient_metrics(gradient_norm=...)`传入
  - 使用示例：
    ```python
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    MonitoringFacade.record_gradient_metrics(
        step=step, gradient_norm=grad_norm, spmd_id="spmd_1"
    )
    ```

- **gradient_mean/std/max/min** (Optional[float]): 梯度统计
  - 获取方式：通过`record_gradient_metrics(...)`传入
  - 使用示例：
    ```python
    grads = [p.grad.flatten() for p in model.parameters() if p.grad is not None]
    all_grads = torch.cat(grads)
    MonitoringFacade.record_gradient_metrics(
        step=step,
        gradient_mean=all_grads.mean().item(),
        gradient_std=all_grads.std().item(),
        gradient_max=all_grads.max().item(),
        gradient_min=all_grads.min().item(),
        spmd_id="spmd_1"
    )
    ```

- **param_update_magnitude** (Optional[float]): 参数更新幅度
  - 获取方式：通过`record_training_progress(metrics={"param_update_magnitude": ...})`传入
  - 使用示例：
    ```python
    # 计算参数更新前后的差异
    update_magnitude = sum((p.data - p_old.data).norm() for p, p_old in zip(model.parameters(), old_params))
    MonitoringFacade.record_training_progress(
        step=step,
        metrics={"param_update_magnitude": update_magnitude},
        spmd_id="spmd_1"
    )
    ```

- **param_norm** (Optional[float]): 参数范数
  - 获取方式：通过`record_training_progress(metrics={"param_norm": ...})`传入
  - 使用示例：
    ```python
    param_norm = sum(p.data.norm() for p in model.parameters())
    MonitoringFacade.record_training_progress(
        step=step, metrics={"param_norm": param_norm}, spmd_id="spmd_1"
    )
    ```

#### 1.4 通信指标（Communication）
- **allreduce_time_ms** (Optional[float]): AllReduce通信耗时（毫秒）
  - 获取方式：通过`record_communication_metrics(allreduce_time_ms=...)`传入
  - 使用示例：
    ```python
    import time
    start = time.time()
    dist.all_reduce(gradients)  # AllReduce操作
    allreduce_time = (time.time() - start) * 1000
    MonitoringFacade.record_communication_metrics(
        step=step, allreduce_time_ms=allreduce_time, spmd_id="spmd_1"
    )
    ```

- **sync_wait_time_ms** (Optional[float]): 同步等待耗时（毫秒）
  - 获取方式：框架自动记录（`SPMDSynchronizer.wait_for_backward_complete()`）
  - 使用示例：
    ```python
    # 框架自动完成，用户无需关心
    synchronizer.wait_for_backward_complete(step)  # 自动记录sync_wait_time_ms
    ```

- **staleness** (Optional[int]): 数据陈旧度
  - 获取方式：通过`record_communication_metrics(staleness=...)`传入
  - 使用示例：
    ```python
    # 计算当前step与最新数据的步数差
    staleness = current_step - latest_data_step
    MonitoringFacade.record_communication_metrics(
        step=step, staleness=staleness, spmd_id="spmd_1"
    )
    ```

- **ulp_transfer_time_ms** / **ulp_assembly_time_ms** (Optional[float]): ULP传输/组装耗时
  - 获取方式：通过`record_communication_metrics(...)`传入
  - 使用示例：
    ```python
    import time
    transfer_start = time.time()
    # ULP传输操作
    transfer_time = (time.time() - transfer_start) * 1000
    
    assembly_start = time.time()
    # ULP组装操作
    assembly_time = (time.time() - assembly_start) * 1000
    
    MonitoringFacade.record_communication_metrics(
        step=step,
        ulp_transfer_time_ms=transfer_time,
        ulp_assembly_time_ms=assembly_time,
        spmd_id="spmd_1"
    )
    ```

#### 1.5 稳定性指标（Stability）
- **nan_inf_detected** (bool): 是否检测到NaN/Inf
  - 获取方式：框架自动检测（`record_gradient_metrics()`中自动检查）
  - 使用示例：
    ```python
    # 框架自动检测，用户无需关心
    MonitoringFacade.record_gradient_metrics(
        step=step, gradient_norm=grad_norm, spmd_id="spmd_1"
    )  # 自动检查梯度中是否有NaN/Inf
    ```

- **gradient_explosion** (bool): 是否梯度爆炸（norm > 100.0）
  - 获取方式：框架自动检测（`record_gradient_metrics()`中自动判断）
  - 使用示例：
    ```python
    # 框架自动检测，用户无需关心
    MonitoringFacade.record_gradient_metrics(
        step=step, gradient_norm=grad_norm, spmd_id="spmd_1"
    )  # 自动判断gradient_norm > 100.0
    ```

- **gradient_vanishing** (bool): 是否梯度消失（norm < 1e-6）
  - 获取方式：框架自动检测（`record_gradient_metrics()`中自动判断）
  - 使用示例：
    ```python
    # 框架自动检测，用户无需关心
    MonitoringFacade.record_gradient_metrics(
        step=step, gradient_norm=grad_norm, spmd_id="spmd_1"
    )  # 自动判断gradient_norm < 1e-6
    ```

- **loss_variance** (Optional[float]): Loss方差
  - 获取方式：框架自动计算（基于最近10个loss值）
  - 使用示例：
    ```python
    # 框架自动计算，用户只需传入loss
    MonitoringFacade.record_training_progress(
        step=step, loss=loss_value, spmd_id="spmd_1"
    )  # 自动计算最近10个loss的方差
    ```

#### 1.6 负载均衡指标（Load Balancing）
- **load_balance_score** / **slow_worker_ratio** (Optional[float])
  - 预留接口，待实现

#### 1.7 数据指标（Data）
- **batch_size** (Optional[int]): 批次大小
  - 获取方式：通过`record_training_progress(batch_size=...)`传入
  - 使用示例：
    ```python
    MonitoringFacade.record_training_progress(
        step=step, batch_size=32, spmd_id="spmd_1"
    )  # 用于计算throughput_samples_per_sec
    ```

- **data_loading_time_ms** (Optional[float]): 数据加载耗时
  - 获取方式：预留接口，待实现
  - 使用示例：
    ```python
    # TODO: 待实现
    # import time
    # start = time.time()
    # data = dataloader.next()
    # load_time = (time.time() - start) * 1000
    # MonitoringFacade.record_training_progress(
    #     step=step, metrics={"data_loading_time_ms": load_time}
    # )
    ```

### 2. Checkpoint指标

- **checkpoint/{event_type}_count**: Checkpoint事件计数
  - 获取方式：框架自动记录（`Worker.save_states()`, `Worker.load_states()`）
  - 事件类型：`save_attempt`, `save_complete`, `save_fail`, `load_complete`, `load_fail`
  - 使用示例：
    ```python
    # 框架自动完成，用户无需关心
    worker.save_states(checkpoint_id="ckpt_1")  # 自动记录save_attempt, save_complete等
    worker.load_states(checkpoint_id="ckpt_1")  # 自动记录load_complete, load_fail等
    ```

- **checkpoint/{event_type}_duration_seconds**: Checkpoint事件耗时（秒）
  - 获取方式：框架自动记录（`Worker.save_states()`, `Worker.load_states()`）
  - 使用示例：
    ```python
    # 框架自动完成，用户无需关心
    worker.save_states(checkpoint_id="ckpt_1")  # 自动记录save操作的耗时
    ```

### 3. 系统指标（System Metrics）

#### 3.1 CPU指标
- **cpu_percent**, **cpu_count**, **cpu_times_user**, **cpu_times_system**
  - 获取方式：框架自动收集（`WorkerMetricsCollector`通过`psutil`）
  - 使用示例：
    ```python
    # 框架自动完成，用户无需关心
    # Worker.send_heartbeat()每30秒自动收集并上报
    ```

#### 3.2 内存指标
- **memory_rss_mb**, **memory_vms_mb**, **memory_percent**
  - 获取方式：框架自动收集（`WorkerMetricsCollector`通过`psutil`）
  - 使用示例：
    ```python
    # 框架自动完成，用户无需关心
    # Worker.send_heartbeat()每30秒自动收集并上报
    ```

#### 3.3 GPU指标（Linux only）
- **gpu_ids**, **gpu_utilization_percent**, **gpu_memory_total_mb**, **gpu_memory_used_mb**, **gpu_memory_percent**, **gpu_temperature_c**
  - 获取方式：框架自动收集（`WorkerMetricsCollector`通过`pynvml`，Linux only）
  - 使用示例：
    ```python
    # 框架自动完成，用户无需关心
    # Worker.send_heartbeat()每30秒自动收集并上报（Linux系统）
    ```

#### 3.4 网络I/O指标
- **network_bytes_sent**, **network_bytes_recv**
  - 获取方式：框架自动收集（`WorkerMetricsCollector`通过`psutil`）
  - 使用示例：
    ```python
    # 框架自动完成，用户无需关心
    # Worker.send_heartbeat()每30秒自动收集并上报
    ```

#### 3.5 磁盘I/O指标
- **disk_read_bytes**, **disk_write_bytes**
  - 获取方式：框架自动收集（`WorkerMetricsCollector`通过`psutil`）
  - 使用示例：
    ```python
    # 框架自动完成，用户无需关心
    # Worker.send_heartbeat()每30秒自动收集并上报
    ```

### 4. 心跳指标（Heartbeat）

- **heartbeat_timestamp** (float): 心跳时间戳
  - 获取方式：框架自动记录（`Worker.send_heartbeat()`每30秒）
  - 使用示例：
    ```python
    # 框架自动完成，用户无需关心
    # Worker.send_heartbeat()每30秒自动发送心跳并记录时间戳
    ```

## 🔧 实现机制

### 1. 自动监控（框架层面）

监控在框架的关键位置自动集成，用户**完全无感知**：

- **SPMDDriver.execute_step()**: 自动记录step开始/结束时间、throughput
- **StatefulExecutor.execute_forward()**: 自动记录forward开始/结束时间、耗时
- **StatefulExecutor.execute_backward()**: 自动记录backward开始/结束时间、耗时
- **StatefulExecutor.execute_parameter_update()**: 自动记录update开始/结束时间、耗时
- **SPMDSynchronizer.wait_for_backward_complete()**: 自动记录同步等待时间
- **Worker.save_states() / load_states()**: 自动记录checkpoint事件
- **Worker.send_heartbeat()**: 自动收集系统指标并发送心跳

### 2. 批量收集机制

- **TrainingMetricsCollector**: 批量大小10个step，批量超时5.0秒
- **自动聚合**: 计算平均值、总和等
- **自动计算**: throughput_steps_per_sec, throughput_samples_per_sec, loss_variance

### 3. 后端写入

- **TensorBoard**: 路径格式 `{component_type}_{component_id}/training/{metric_name}`
- **Ray Dashboard**: 通过Ray的metrics API记录
- **Prometheus**: Prometheus metrics格式，带标签

## 📝 用户使用指南

### 用户代码示例（完全自动）

```python
class MyModel(UserModuleInterface):
    def forward(self, step: int):
        # ✅ 用户只需实现模型逻辑
        # ✅ 监控由框架自动完成！
        return self.model(input_data)
    
    def backward(self, step: int):
        # ✅ 用户只需实现backward逻辑
        return self.optimizer.step()
    
    def update_parameters(self):
        # ✅ 用户只需实现update逻辑
        return self.optimizer.apply_gradients()
```

### 可选：传入额外指标

如果需要记录额外的指标（如loss、learning_rate等），可以通过框架的API传入：

```python
# 在StatefulExecutor.execute_forward()中，如果forward返回了loss
result = self.user_module.forward(step)
if isinstance(result, dict) and 'loss' in result:
    MonitoringFacade.record_training_progress(
        step=step,
        spmd_id=self.spmd_id,
        loss=result['loss'],
        metrics={"learning_rate": self.optimizer.learning_rate}
    )
```

## 🎯 总结

### 自动监控的指标

- ✅ Step时间（开始/结束）
- ✅ Forward/Backward/Update时间
- ✅ Throughput（steps/sec, samples/sec）
- ✅ Checkpoint事件
- ✅ 系统指标（CPU、内存、GPU、网络、磁盘）
- ✅ 心跳

### 需要用户传入的指标（可选）

- ⚠️ Loss（如果forward不返回loss，需要用户传入）
- ⚠️ Epoch（需要用户传入）
- ⚠️ Learning rate（需要用户传入）
- ⚠️ Gradient metrics（需要用户传入）
- ⚠️ Communication metrics（部分自动，部分需要用户传入）

### 核心原则

**监控是自动的，用户无需手动添加监控代码！**

用户只需实现模型逻辑，监控完全由框架自动完成。

## 📚 示例代码

完整示例请参考：`examples/monitoring/test_monitoring.py`
