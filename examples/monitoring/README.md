# 监控系统示例

## 📋 文件说明

- `test_monitoring.py`: 完整的监控系统测试案例，展示用户代码和框架代码的区别

## 🚀 运行示例

### 1. 安装依赖

```bash
pip install -r requirements-monitor.txt
```

### 2. 运行测试

```bash
python examples/monitoring/test_monitoring.py
```

### 3. 查看监控数据

启动TensorBoard（自动刷新）：
```bash
tensorboard --logdir=./logs/tensorboard --reload_interval=5
```

访问：http://localhost:6006

## 📚 相关文档

详细文档请参考：`docs/MONITORING_COMPLETE.md`
