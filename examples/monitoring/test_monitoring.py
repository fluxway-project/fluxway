"""
Fluxway监控系统测试案例

本测试案例展示了监控系统的完整功能，并标注了哪些逻辑是用户不感知的（框架自动完成）。

用户只需实现模型逻辑，监控完全由框架自动完成。
"""

import time
import math
import logging
import os
import sys

# 添加项目路径（从examples/monitoring/到项目根目录）
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import ray
from fluxway.spmd.driver.driver import SPMDDriver
from fluxway.spmd.driver.stateful_executor.executor import StatefulExecutor
from fluxway.common.monitoring import MonitoringFacade

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# 用户代码：用户只需实现模型逻辑
# ============================================================================


class SimpleModel:
    """
    用户模型：用户只需实现forward、backward、update逻辑
    监控由框架自动完成，用户完全无感知
    """

    def __init__(self):
        self.lr = 0.01
        self.step_count = 0

    def forward(self, step: int, input_data=None):
        """
        用户实现：前向传播

        注意：用户不需要添加任何监控代码！
        框架会在StatefulExecutor.execute_forward()中自动记录：
        - forward开始/结束时间
        - forward耗时
        """
        # 模拟前向传播
        time.sleep(0.01)  # 模拟计算时间

        # 模拟loss（实际应该从模型计算得出）
        base_loss = 1.0
        loss = base_loss * (0.99**step) + 0.05 * math.sin(step / 10)

        return {"output": f"output_step_{step}", "loss": loss}

    def backward(self, step: int):
        """
        用户实现：反向传播

        注意：用户不需要添加任何监控代码！
        框架会在StatefulExecutor.execute_backward()中自动记录：
        - backward开始/结束时间
        - backward耗时
        """
        # 模拟反向传播
        time.sleep(0.01)  # 模拟计算时间

        # 模拟梯度（实际应该从backward计算得出）
        gradient_norm = 1.0 + 0.1 * math.sin(step / 20)

        return {"gradient_norm": gradient_norm, "gradient_mean": 0.01, "gradient_std": 0.1}

    def update_parameters(self):
        """
        用户实现：参数更新

        注意：用户不需要添加任何监控代码！
        框架会在StatefulExecutor.execute_parameter_update()中自动记录：
        - update开始/结束时间
        - update耗时
        """
        # 模拟参数更新
        time.sleep(0.005)  # 模拟计算时间
        self.step_count += 1
        return {"updated": True}


class UserModuleInterface:
    """
    用户模块接口：用户需要实现的接口
    """

    def __init__(self):
        self.model = SimpleModel()

    def forward(self, step: int):
        """用户实现：前向传播"""
        return self.model.forward(step)

    def backward(self, step: int):
        """用户实现：反向传播"""
        return self.model.backward(step)

    def update_parameters(self):
        """用户实现：参数更新"""
        return self.model.update_parameters()

    def save_states(self):
        """用户实现：保存状态（可选）"""
        return f"checkpoint_{int(time.time())}"

    def load_states(self, checkpoint_id: str):
        """用户实现：加载状态（可选）"""
        return True


# ============================================================================
# 框架代码：用户不感知，框架自动完成
# ============================================================================


class MonitoredExecutor(StatefulExecutor):
    """
    框架代码：扩展StatefulExecutor以集成用户模型

    这部分代码是框架层面的，用户不需要关心。
    框架会自动：
    1. 记录forward/backward/update的时间
    2. 提取loss、gradient等指标
    3. 上报到监控系统

    注意：实际的StatefulExecutor.execute_forward()等方法已经自动集成了监控，
    这里只是为了演示，实际使用时直接使用StatefulExecutor即可。
    """

    def __init__(self, spmd_id: str, user_module: UserModuleInterface):
        super().__init__(spmd_id)
        self.user_module = user_module

    def execute_forward(self, step: int):
        """
        框架代码：执行前向传播并自动监控

        用户不感知：
        - 自动记录forward_start/forward_end时间（框架已实现）
        - 自动计算forward耗时（框架已实现）
        - 自动提取loss并上报（框架已实现）

        注意：实际的StatefulExecutor.execute_forward()已经自动集成了监控，
        这里模拟框架的实现方式。
        """
        # ========== 框架自动监控：用户不感知 ==========
        # 实际的StatefulExecutor.execute_forward()已经自动集成了监控：
        MonitoringFacade.record_training_timing(step, "forward_start", self.spmd_id)
        forward_start = time.time()
        # ============================================

        # 调用用户代码（用户只需实现模型逻辑）
        result = self.user_module.forward(step)

        # ========== 框架自动监控：用户不感知 ==========
        # 实际的StatefulExecutor.execute_forward()已经自动集成了监控：
        forward_time = time.time() - forward_start
        MonitoringFacade.record_training_timing(step, "forward_end", self.spmd_id)
        MonitoringFacade.record_training_progress(
            step=step, spmd_id=self.spmd_id, metrics={"forward_time_ms": forward_time * 1000}
        )
        # ============================================

        # 如果forward返回了loss，额外记录（框架自动完成）
        if isinstance(result, dict) and "loss" in result:
            MonitoringFacade.record_training_progress(
                step=step, spmd_id=self.spmd_id, loss=result["loss"]
            )

        return result

    def execute_backward(self, step: int):
        """
        框架代码：执行反向传播并自动监控

        用户不感知：
        - 自动记录backward_start/backward_end时间（框架已实现）
        - 自动计算backward耗时（框架已实现）
        - 自动提取gradient指标并上报（框架已实现）

        注意：实际的StatefulExecutor.execute_backward()已经自动集成了监控，
        这里模拟框架的实现方式。
        """
        # ========== 框架自动监控：用户不感知 ==========
        # 实际的StatefulExecutor.execute_backward()已经自动集成了监控：
        MonitoringFacade.record_training_timing(step, "backward_start", self.spmd_id)
        backward_start = time.time()
        # ============================================

        # 调用用户代码（用户只需实现模型逻辑）
        result = self.user_module.backward(step)

        # ========== 框架自动监控：用户不感知 ==========
        # 实际的StatefulExecutor.execute_backward()已经自动集成了监控：
        backward_time = time.time() - backward_start
        MonitoringFacade.record_training_timing(step, "backward_end", self.spmd_id)
        MonitoringFacade.record_training_progress(
            step=step, spmd_id=self.spmd_id, metrics={"backward_time_ms": backward_time * 1000}
        )
        # ============================================

        # 如果backward返回了gradient指标，额外记录（框架自动完成）
        if isinstance(result, dict) and "gradient_norm" in result:
            MonitoringFacade.record_gradient_metrics(
                step=step,
                gradient_norm=result.get("gradient_norm"),
                gradient_mean=result.get("gradient_mean"),
                gradient_std=result.get("gradient_std"),
                spmd_id=self.spmd_id,
            )

        return result

    def execute_parameter_update(self):
        """
        框架代码：执行参数更新并自动监控

        用户不感知：
        - 自动记录update_start/update_end时间（框架已实现）
        - 自动计算update耗时（框架已实现）

        注意：实际的StatefulExecutor.execute_parameter_update()已经自动集成了监控，
        这里模拟框架的实现方式。
        """
        # ========== 框架自动监控：用户不感知 ==========
        # 实际的StatefulExecutor.execute_parameter_update()已经自动集成了监控：
        step = getattr(self, "_current_step", 0)
        MonitoringFacade.record_training_timing(step, "update_start", self.spmd_id)
        update_start = time.time()
        # ============================================

        # 调用用户代码（用户只需实现模型逻辑）
        result = self.user_module.update_parameters()

        # ========== 框架自动监控：用户不感知 ==========
        # 实际的StatefulExecutor.execute_parameter_update()已经自动集成了监控：
        update_time = time.time() - update_start
        MonitoringFacade.record_training_timing(step, "update_end", self.spmd_id)
        MonitoringFacade.record_training_progress(
            step=step, spmd_id=self.spmd_id, metrics={"update_time_ms": update_time * 1000}
        )
        # ============================================

        return result


class MonitoredDriver(SPMDDriver):
    """
    框架代码：扩展SPMDDriver以集成监控

    这部分代码是框架层面的，用户不需要关心。
    框架会自动：
    1. 记录step开始/结束时间（框架已实现）
    2. 计算throughput（框架已实现）
    3. 协调forward/backward/update的执行

    注意：实际的SPMDDriver.execute_step()已经自动集成了监控，
    这里只是为了演示，实际使用时直接使用SPMDDriver即可。
    """

    def __init__(self, spmd_id: str, user_module: UserModuleInterface):
        super().__init__(spmd_id)
        self.executor = MonitoredExecutor(spmd_id, user_module)
        self._current_step = 0

    def execute_step(self, step: int):
        """
        框架代码：执行一个训练步并自动监控

        用户不感知：
        - 自动记录step_start时间（框架已实现）
        - 自动协调forward/backward/update（框架已实现）
        - 自动记录step结束时间（框架已实现）
        - 自动计算throughput（框架已实现）

        注意：实际的SPMDDriver.execute_step()已经自动集成了监控：
        - MonitoringFacade.record_training_timing(step, "step_start", self.spmd_id)
        - step_start_time = time.time()
        - ... 执行forward/backward/update ...
        - MonitoringFacade.record_training_progress(step, spmd_id, metrics={"total_step_time_ms": ..., "throughput_steps_per_sec": ...})
        """
        self._current_step = step

        # ========== 框架自动监控：用户不感知 ==========
        # 实际的SPMDDriver.execute_step()已经自动集成了监控：
        MonitoringFacade.record_training_timing(step, "step_start", self.spmd_id)
        step_start_time = time.time()
        # ============================================

        # 执行前向传播（框架自动监控，调用用户代码）
        forward_result = self.executor.execute_forward(step)

        # 执行反向传播（框架自动监控，调用用户代码）
        backward_result = self.executor.execute_backward(step)

        # 执行参数更新（每4步一次，框架自动监控，调用用户代码）
        if step % 4 == 0:
            self.executor._current_step = step  # 设置step以便监控
            update_result = self.executor.execute_parameter_update()

        # ========== 框架自动监控：用户不感知 ==========
        # 实际的SPMDDriver.execute_step()已经自动集成了监控：
        step_time = time.time() - step_start_time
        step_time_ms = step_time * 1000
        throughput_steps_per_sec = 1.0 / step_time if step_time > 0 else 0.0

        MonitoringFacade.record_training_progress(
            step=step,
            spmd_id=self.spmd_id,
            batch_size=32,  # 假设batch_size=32
            metrics={
                "total_step_time_ms": step_time_ms,
                "throughput_steps_per_sec": throughput_steps_per_sec,
            },
        )
        # ============================================

        # 额外记录loss（框架自动完成，如果forward返回了loss）
        if isinstance(forward_result, dict) and "loss" in forward_result:
            MonitoringFacade.record_training_progress(
                step=step, spmd_id=self.spmd_id, loss=forward_result.get("loss")
            )

        return forward_result


# ============================================================================
# 测试主函数
# ============================================================================


def test_monitoring():
    """
    测试监控系统

    本测试展示了：
    1. 用户只需实现模型逻辑（SimpleModel）
    2. 框架自动完成所有监控（MonitoredDriver, MonitoredExecutor）
    3. 监控数据自动写入TensorBoard、Ray Dashboard等
    """

    # ========== 初始化Ray ==========
    logger.info("初始化Ray...")
    try:
        ray.init(address="auto", ignore_reinit_error=True)
        logger.info("Ray已连接")
    except Exception:
        logger.info("Ray连接失败，使用local_mode")
        ray.init(local_mode=True, ignore_reinit_error=True)

    # ========== 创建用户模型 ==========
    logger.info("创建用户模型...")
    user_module = UserModuleInterface()

    # ========== 创建SPMD Driver（框架自动集成监控） ==========
    logger.info("创建SPMD Driver（框架自动集成监控）...")
    spmd_id = "spmd_test"
    driver = MonitoredDriver(spmd_id, user_module)

    # ========== 启动监控（框架自动） ==========
    logger.info("启动监控系统（框架自动）...")
    try:
        daemon = MonitoringFacade.start_spmd_monitoring(
            spmd_id=spmd_id, check_interval=10.0, report_interval=30.0
        )
        logger.info("监控守护进程已启动")
    except Exception as e:
        logger.warning(f"监控守护进程启动失败: {e}")
        daemon = None

    # ========== 运行训练循环 ==========
    logger.info("开始训练循环...")
    num_steps = 100

    for step in range(num_steps):
        # 用户只需调用execute_step，监控自动完成
        result = driver.execute_step(step)

        # 每10步输出一次（可选）
        if step % 10 == 0:
            logger.info(f"Step {step}/{num_steps}: loss={result.get('loss', 0):.4f}")

        # 每20步保存checkpoint（框架自动监控checkpoint事件）
        if step > 0 and step % 20 == 0:
            checkpoint_id = user_module.save_states()
            logger.info(f"Checkpoint saved: {checkpoint_id}")
            # 框架会在Worker.save_states()中自动记录checkpoint事件

    logger.info("训练完成！")

    # ========== 清理 ==========
    if daemon:
        try:
            daemon.stop()
            daemon.wait()
            logger.info("监控守护进程已停止")
        except Exception:
            pass

    logger.info("测试完成！")
    logger.info("")
    logger.info("查看监控数据：")
    logger.info("1. TensorBoard: tensorboard --logdir=./logs/tensorboard --reload_interval=5")
    logger.info("   访问: http://localhost:6006")
    logger.info("2. Ray Dashboard: http://localhost:8265 (如果使用Ray集群)")


if __name__ == "__main__":
    test_monitoring()
