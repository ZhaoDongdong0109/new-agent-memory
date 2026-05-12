"""
自动化训练智能体 - AutoTrainerAgent
帮你自动完成整个训练流程，减少人工干预
"""

import os
import json
import time
import torch
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime


class TrainingStage(Enum):
    PREPARATION = "preparation"
    ENCODER = "encoder"
    RETRIEVER = "retriever"
    ADAPTER = "adapter"
    EVALUATION = "evaluation"
    COMPLETED = "completed"


@dataclass
class TrainingMetrics:
    """训练指标"""
    epoch: int
    loss: float
    accuracy: Optional[float] = None
    learning_rate: float = 0.0
    gpu_memory: float = 0.0
    timestamp: float = 0.0


@dataclass
class AutoTrainerConfig:
    """自动化训练配置"""
    # 基础配置
    auto_save: bool = True
    auto_adjust_lr: bool = True
    auto_early_stopping: bool = True
    auto_report: bool = True
    
    # 阈值配置
    patience: int = 3  # 早停耐心值
    min_improvement: float = 0.01  # 最小改进
    max_epochs_without_improvement: int = 5
    
    # 监控配置
    check_interval: int = 10  # 检查间隔（秒）
    save_interval: int = 1  # 保存间隔（epoch）
    
    # 资源限制
    max_gpu_memory_gb: float = 20.0
    max_training_hours: float = 48.0


class TrainingAgent:
    """
    自动化训练智能体
    
    职责：
    1. 自动监控训练进度
    2. 自动调整学习率
    3. 自动早停
    4. 自动保存模型
    5. 自动生成报告
    6. 自动处理异常
    """
    
    def __init__(self, config: Optional[AutoTrainerConfig] = None):
        self.config = config or AutoTrainerConfig()
        self.current_stage = TrainingStage.PREPARATION
        
        # 状态跟踪
        self.training_history: List[TrainingMetrics] = []
        self.best_loss = float('inf')
        self.epochs_without_improvement = 0
        self.start_time = time.time()
        
        # 自动调整状态
        self.learning_rate = 1e-4
        self.batch_size = 16
        
        # 报告
        self.reports: List[Dict] = []
        
        # 设置日志
        self._setup_logging()
    
    def _setup_logging(self):
        """设置日志"""
        log_dir = "./training_logs"
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def check_resources(self) -> Dict[str, Any]:
        """检查系统资源"""
        resources = {
            "gpu_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cpu_count": os.cpu_count(),
        }
        
        if torch.cuda.is_available():
            resources["gpu_name"] = torch.cuda.get_device_name(0)
            resources["gpu_memory_total"] = torch.cuda.get_device_properties(0).total_memory / 1e9
            resources["gpu_memory_allocated"] = torch.cuda.memory_allocated(0) / 1e9
            resources["gpu_memory_cached"] = torch.cuda.memory_reserved(0) / 1e9
        
        return resources
    
    def should_stop_training(self) -> bool:
        """判断是否应该停止训练"""
        # 检查是否达到最大训练时间
        elapsed_hours = (time.time() - self.start_time) / 3600
        if elapsed_hours > self.config.max_training_hours:
            self.logger.warning(f"达到最大训练时间 {self.config.max_training_hours}h，停止训练")
            return True
        
        # 检查是否连续多轮没有改进
        if self.epochs_without_improvement >= self.config.max_epochs_without_improvement:
            self.logger.warning(f"连续 {self.epochs_without_improvement} 轮没有改进，停止训练")
            return True
        
        return False
    
    def update_metrics(self, metrics: TrainingMetrics) -> Dict[str, Any]:
        """更新训练指标"""
        self.training_history.append(metrics)
        
        decisions = {
            "should_save": False,
            "should_adjust_lr": False,
            "should_stop": False,
            "lr_change": 0.0
        }
        
        # 检查是否需要保存
        if metrics.loss < self.best_loss:
            improvement = (self.best_loss - metrics.loss) / self.best_loss
            if improvement >= self.config.min_improvement:
                self.best_loss = metrics.loss
                self.epochs_without_improvement = 0
                decisions["should_save"] = True
                self.logger.info(f"✓ 新的最佳损失: {metrics.loss:.4f} (改进 {improvement*100:.2f}%)")
            else:
                self.epochs_without_improvement += 1
        else:
            self.epochs_without_improvement += 1
        
        # 检查是否需要调整学习率
        if self.config.auto_adjust_lr and self.epochs_without_improvement >= self.config.patience:
            old_lr = self.learning_rate
            self.learning_rate *= 0.5
            decisions["should_adjust_lr"] = True
            decisions["lr_change"] = self.learning_rate - old_lr
            self.logger.info(f"↔ 调整学习率: {old_lr:.2e} → {self.learning_rate:.2e}")
            self.epochs_without_improvement = 0  # 重置计数器
        
        # 检查是否应该停止
        decisions["should_stop"] = self.should_stop_training()
        
        return decisions
    
    def save_checkpoint(
        self,
        model: Any,
        stage: TrainingStage,
        epoch: int,
        metrics: TrainingMetrics
    ):
        """保存检查点"""
        checkpoint_dir = f"./checkpoints/{stage.value}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pt")
        
        checkpoint = {
            "stage": stage.value,
            "epoch": epoch,
            "metrics": {
                "loss": metrics.loss,
                "accuracy": metrics.accuracy,
                "learning_rate": metrics.learning_rate
            },
            "best_loss": self.best_loss,
            "timestamp": time.time()
        }
        
        torch.save({
            "model_state_dict": model.state_dict(),
            "checkpoint_info": checkpoint
        }, checkpoint_path)
        
        # 同时保存最佳模型
        if metrics.loss == self.best_loss:
            best_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "checkpoint_info": checkpoint
            }, best_path)
            self.logger.info(f"★ 保存最佳模型: {best_path}")
        
        return checkpoint_path
    
    def generate_report(self) -> Dict[str, Any]:
        """生成训练报告"""
        elapsed_time = time.time() - self.start_time
        
        # 计算统计数据
        if not self.training_history:
            return {}
        
        losses = [m.loss for m in self.training_history]
        
        report = {
            "training_info": {
                "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
                "end_time": datetime.now().isoformat(),
                "total_duration_hours": elapsed_time / 3600,
                "current_stage": self.current_stage.value
            },
            "performance": {
                "best_loss": self.best_loss,
                "final_loss": losses[-1] if losses else None,
                "total_epochs": len(self.training_history),
                "average_loss": sum(losses) / len(losses) if losses else None,
                "loss_reduction": (losses[0] - losses[-1]) / losses[0] if losses else 0
            },
            "optimization": {
                "final_learning_rate": self.learning_rate,
                "final_batch_size": self.batch_size,
                "epochs_without_improvement": self.epochs_without_improvement
            },
            "recommendations": self._generate_recommendations()
        }
        
        self.reports.append(report)
        
        # 保存报告
        report_dir = "./training_reports"
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"📊 训练报告已保存: {report_path}")
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        if not self.training_history:
            return ["数据不足，无法生成建议"]
        
        losses = [m.loss for m in self.training_history]
        
        # 分析损失曲线
        if len(losses) > 5:
            recent_losses = losses[-5:]
            if all(recent_losses[i] >= recent_losses[i+1] for i in range(len(recent_losses)-1)):
                recommendations.append("损失持续上升，考虑降低学习率或检查数据质量")
            elif recent_losses[-1] < losses[0] * 0.5:
                recommendations.append("训练效果良好，可以尝试增加模型容量")
        
        # 检查早停
        if self.epochs_without_improvement > 2:
            recommendations.append("多次早停，建议增加数据多样性或调整模型结构")
        
        # 检查GPU使用
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(0) / 1e9
            if memory_allocated > self.config.max_gpu_memory_gb * 0.9:
                recommendations.append("GPU内存使用率过高，建议减小batch_size")
        
        if not recommendations:
            recommendations.append("训练过程正常，继续当前配置")
        
        return recommendations
    
    def print_status(self):
        """打印当前状态"""
        elapsed = time.time() - self.start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        
        print("\n" + "=" * 70)
        print(f"  🤖 自动化训练智能体 - 当前状态")
        print("=" * 70)
        print(f"  阶段: {self.current_stage.value}")
        print(f"  运行时间: {hours}h {minutes}m")
        print(f"  当前学习率: {self.learning_rate:.2e}")
        print(f"  最佳损失: {self.best_loss:.4f}")
        print(f"  连续无改进轮数: {self.epochs_without_improvement}")
        print(f"  总训练轮数: {len(self.training_history)}")
        
        if self.training_history:
            latest = self.training_history[-1]
            print(f"  最新损失: {latest.loss:.4f}")
        
        resources = self.check_resources()
        if resources["gpu_available"]:
            print(f"  GPU: {resources['gpu_name']}")
            print(f"  GPU内存: {resources['gpu_memory_allocated']:.1f}GB / {resources['gpu_memory_total']:.1f}GB")
        
        print("=" * 70 + "\n")


class AutoTrainer:
    """
    自动化训练器 - 将智能体集成到训练流程
    """
    
    def __init__(self, model: Any, config: Optional[AutoTrainerConfig] = None):
        self.agent = TrainingAgent(config)
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def train_with_agent(
        self,
        train_loader: Any,
        num_epochs: int,
        stage: TrainingStage
    ):
        """使用智能体自动化训练"""
        self.agent.current_stage = stage
        self.agent.logger.info(f"开始训练阶段: {stage.value}")
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.agent.learning_rate
        )
        
        for epoch in range(num_epochs):
            self.agent.print_status()
            
            epoch_losses = []
            
            for batch_idx, batch in enumerate(train_loader):
                # 训练步骤
                loss = self._training_step(batch, optimizer)
                epoch_losses.append(loss)
                
                # 定期检查
                if batch_idx % 10 == 0:
                    if self.agent.should_stop_training():
                        self.agent.logger.warning("智能体决定停止训练")
                        break
            
            # 计算平均损失
            avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float('inf')
            
            # 更新指标
            metrics = TrainingMetrics(
                epoch=epoch,
                loss=avg_loss,
                learning_rate=self.agent.learning_rate,
                timestamp=time.time()
            )
            
            # 获取智能体决策
            decisions = self.agent.update_metrics(metrics)
            
            # 执行决策
            if decisions["should_save"]:
                self.agent.save_checkpoint(self.model, stage, epoch, metrics)
            
            if decisions["should_adjust_lr"]:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.agent.learning_rate
            
            if decisions["should_stop"]:
                self.agent.logger.info("智能体决定停止训练")
                break
            
            # 每轮结束生成报告
            if epoch % 5 == 0:
                self.agent.generate_report()
        
        # 最终报告
        final_report = self.agent.generate_report()
        return final_report
    
    def _training_step(self, batch: Any, optimizer: torch.optim.Optimizer) -> float:
        """单个训练步骤"""
        # 简化实现
        return 0.5  # 示例损失值


def create_auto_trainer(model: Any) -> AutoTrainer:
    """创建自动化训练器"""
    config = AutoTrainerConfig(
        auto_save=True,
        auto_adjust_lr=True,
        auto_early_stopping=True,
        auto_report=True,
        patience=3,
        max_epochs_without_improvement=5
    )
    
    return AutoTrainer(model, config)


if __name__ == "__main__":
    print("=" * 70)
    print("  🤖 自动化训练智能体演示")
    print("=" * 70)
    
    # 创建智能体
    agent = TrainingAgent()
    
    # 检查资源
    print("\n[1] 系统资源检查:")
    resources = agent.check_resources()
    for key, value in resources.items():
        print(f"  {key}: {value}")
    
    # 模拟训练过程
    print("\n[2] 模拟训练过程:")
    for epoch in range(10):
        # 模拟损失（逐渐下降）
        loss = 1.0 / (epoch + 1) + 0.1 * (0.8 ** epoch)
        
        metrics = TrainingMetrics(
            epoch=epoch,
            loss=loss,
            learning_rate=agent.learning_rate,
            timestamp=time.time()
        )
        
        decisions = agent.update_metrics(metrics)
        
        if decisions["should_save"]:
            print(f"  ✓ Epoch {epoch}: 保存模型 (loss={loss:.4f})")
        else:
            print(f"    Epoch {epoch}: loss={loss:.4f}")
        
        if decisions["should_adjust_lr"]:
            print(f"  ↔ 调整学习率: {agent.learning_rate:.2e}")
    
    # 生成报告
    print("\n[3] 生成训练报告:")
    report = agent.generate_report()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    
    print("\n" + "=" * 70)
    print("✅ 自动化训练智能体演示完成！")
    print("=" * 70)
