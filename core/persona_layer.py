"""
人格适应层 - 类人记忆系统

核心理念：
- 不预设人格，从行为反馈中学习用户是什么人
- 系统默认主动提及旧记忆，用户通过反馈调整行为
- 收集显式和隐式信号，持续更新用户偏好模型

反馈信号设计：
| 用户行为 | 信号方向 | 分值 |
|---------|---------|------|
| 主动提及后，用户继续聊这个话题 | 正向 | +0.2 |
| 主动提及后，用户说"你居然记得这个"/"对！" | 强烈正向 | +0.3 |
| 主动提及后，用户转移话题 | 负向 | -0.15 |
| 主动提及后，用户说"别老提这个" | 强烈负向 | -0.4 |
| 用户主动提起旧事 | 正向 | +0.2 |
| 重建的内容被纠正 | 正向 | +0.05 |

学习阈值：
- 收集满10个信号后，根据兴趣度决定开启或关闭该行为
- 之后每20个信号重新评估一次
"""

import time
import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class BehaviorType(Enum):
    """可被学习的行为类型"""
    ACTIVE_RECALL = "active_recall"  # 主动提及旧记忆


@dataclass
class BehaviorPreference:
    """单个行为的偏好模型"""
    enabled: bool = True              # 是否启用该行为（默认开启，谨慎收集信号）
    interest_score: float = 0.5       # 用户兴趣度 0.0~1.0（初始中立）
    signals_collected: int = 0        # 已收集的信号数量
    last_evaluated: float = 0.0       # 上次评估时间戳
    
    # 信号历史（用于分析趋势）
    signal_history: List[float] = field(default_factory=list)
    
    # 评估阈值
    INITIAL_SIGNALS_THRESHOLD = 10     # 初始评估需要的信号数
    PERIODIC_EVALUATION_THRESHOLD = 20  # 周期性评估需要的信号数


@dataclass
class UserPersonaProfile:
    """
    用户人格侧写
    
    追踪用户对各种系统行为的偏好，
    所有偏好从行为反馈中学习，不预设
    """
    user_id: str = "default"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    # 行为偏好
    behaviors: Dict[str, BehaviorPreference] = field(default_factory=dict)
    
    # 交互统计
    total_interactions: int = 0
    total_active_recalls_triggered: int = 0  # 系统主动提及的次数
    total_user_accepted: int = 0            # 用户接受的次数
    total_user_rejected: int = 0            # 用户拒绝的次数
    
    # 其他可扩展的偏好（未来）
    # response_length: enum[short, medium, long]
    # tone: enum[formal, casual, warm]
    
    def __post_init__(self):
        # 初始化所有行为偏好
        if BehaviorType.ACTIVE_RECALL.value not in self.behaviors:
            self.behaviors[BehaviorType.ACTIVE_RECALL.value] = BehaviorPreference()


class PersonaLayer:
    """
    人格适应层
    
    通过反馈学习用户偏好，动态调整系统行为
    """
    
    # 反馈信号分值
    SIGNAL_CONTINUE_TOPIC = 0.2       # 继续聊这个话题
    SIGNAL_EXPLICIT_POSITIVE = 0.3    # "你居然记得这个"
    SIGNAL_IGNORE = -0.15            # 转移话题
    SIGNAL_EXPLICIT_NEGATIVE = -0.4  # "别老提这个"
    SIGNAL_USER_INITIATED = 0.2       # 用户主动提起旧事
    SIGNAL_CORRECTION = 0.05         # 重建内容被纠正
    
    # 学习阈值
    INITIAL_EVALUATION_SIGNALS = 10
    PERIODIC_EVALUATION_SIGNALS = 20
    
    def __init__(self, profile: Optional[UserPersonaProfile] = None):
        self.profile = profile or UserPersonaProfile()
    
    # ============ 反馈收集 ============
    
    def record_signal(
        self,
        behavior: BehaviorType,
        signal_type: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        记录一个反馈信号
        
        Args:
            behavior: 行为类型
            signal_type: 信号类型（continue_topic/explicit_positive/ignore/
                        explicit_negative/user_initiated/correction）
            context: 额外上下文
            
        Returns:
            更新后的兴趣度
        """
        context = context or {}
        
        # 获取信号分值
        signal_value = self._get_signal_value(signal_type)
        
        # 获取或创建行为偏好
        behavior_key = behavior.value
        if behavior_key not in self.profile.behaviors:
            self.profile.behaviors[behavior_key] = BehaviorPreference()
        
        pref = self.profile.behaviors[behavior_key]
        
        # 更新信号历史
        pref.signal_history.append(signal_value)
        if len(pref.signal_history) > 50:  # 保留最近50条
            pref.signal_history = pref.signal_history[-50:]
        
        # 更新兴趣度（指数移动平均）
        old_score = pref.interest_score
        pref.interest_score = (
            old_score * 0.8 + signal_value * 0.2
        )
        pref.interest_score = max(0.0, min(1.0, pref.interest_score))
        
        pref.signals_collected += 1
        pref.last_evaluated = time.time()
        self.profile.updated_at = time.time()
        
        # 更新全局统计
        if signal_type in ("explicit_positive", "continue_topic"):
            self.profile.total_user_accepted += 1
        elif signal_type in ("explicit_negative", "ignore"):
            self.profile.total_user_rejected += 1
        
        # 检查是否需要评估
        if pref.signals_collected >= self.INITIAL_EVALUATION_SIGNALS:
            if pref.last_evaluated == 0 or pref.signals_collected == self.INITIAL_EVALUATION_SIGNALS:
                self._evaluate_behavior(behavior_key, pref)
        
        return pref.interest_score
    
    def _get_signal_value(self, signal_type: str) -> float:
        """获取信号分值"""
        signal_map = {
            "continue_topic": self.SIGNAL_CONTINUE_TOPIC,
            "explicit_positive": self.SIGNAL_EXPLICIT_POSITIVE,
            "ignore": self.SIGNAL_IGNORE,
            "explicit_negative": self.SIGNAL_EXPLICIT_NEGATIVE,
            "user_initiated": self.SIGNAL_USER_INITIATED,
            "correction": self.SIGNAL_CORRECTION,
        }
        return signal_map.get(signal_type, 0.0)
    
    def _evaluate_behavior(self, behavior_key: str, pref: BehaviorPreference):
        """
        评估行为是否应该开启或关闭
        
        评估逻辑：
        - 兴趣度 > 0.6 → 开启（enabled=True）
        - 兴趣度 < 0.3 → 关闭（enabled=False）
        - 0.3 ~ 0.6 → 保持现状，继续收集信号
        """
        if pref.signals_collected < self.INITIAL_EVALUATION_SIGNALS:
            return
        
        # 周期性评估：每20个信号重新评估一次
        if pref.signals_collected % self.PERIODIC_EVALUATION_SIGNALS != 0:
            if pref.signals_collected != self.INITIAL_EVALUATION_SIGNALS:
                return
        
        old_enabled = pref.enabled
        
        if pref.interest_score > 0.6:
            pref.enabled = True
        elif pref.interest_score < 0.3:
            pref.enabled = False
        # else: 0.3~0.6，保持现状
        
        if old_enabled != pref.enabled:
            pref.last_evaluated = time.time()
    
    # ============ 行为查询 ============
    
    def is_behavior_enabled(self, behavior: BehaviorType) -> bool:
        """查询行为是否应该执行"""
        behavior_key = behavior.value
        if behavior_key not in self.profile.behaviors:
            return True  # 默认开启
        return self.profile.behaviors[behavior_key].enabled
    
    def get_interest_score(self, behavior: BehaviorType) -> float:
        """获取用户对某行为的兴趣度"""
        behavior_key = behavior.value
        if behavior_key not in self.profile.behaviors:
            return 0.5  # 默认中立
        return self.profile.behaviors[behavior_key].interest_score
    
    def should_trigger_active_recall(
        self,
        base_probability: float = 0.3,
    ) -> bool:
        """
        决定是否触发主动提及
        
        基于：
        1. 用户偏好（是否开启）
        2. 当前兴趣度（越高越可能触发）
        3. 基础概率
        
        Args:
            base_probability: 基础触发概率（当兴趣度=0.5时使用）
        """
        if not self.is_behavior_enabled(BehaviorType.ACTIVE_RECALL):
            return False
        
        interest = self.get_interest_score(BehaviorType.ACTIVE_RECALL)
        
        # 兴趣度越高，触发概率越高
        # 兴趣度0.5时用base_probability
        # 兴趣度1.0时概率翻倍（上限1.0）
        trigger_prob = base_probability * (0.5 + interest)
        trigger_prob = min(1.0, trigger_prob)
        
        import random
        return random.random() < trigger_prob
    
    # ============ 反馈快捷方法 ============
    
    def on_active_recall_continue(self):
        """主动提及后用户继续聊这个话题"""
        return self.record_signal(BehaviorType.ACTIVE_RECALL, "continue_topic")
    
    def on_active_recall_explicit_positive(self):
        """主动提及后用户表示惊喜"""
        return self.record_signal(BehaviorType.ACTIVE_RECALL, "explicit_positive")
    
    def on_active_recall_ignore(self):
        """主动提及后用户转移话题"""
        return self.record_signal(BehaviorType.ACTIVE_RECALL, "ignore")
    
    def on_active_recall_explicit_negative(self):
        """主动提及后用户表示厌烦"""
        return self.record_signal(BehaviorType.ACTIVE_RECALL, "explicit_negative")
    
    def on_user_initiated_recall(self):
        """用户主动提起旧事"""
        self.profile.total_interactions += 1
        return self.record_signal(BehaviorType.ACTIVE_RECALL, "user_initiated")
    
    def on_reconstruction_corrected(self):
        """重建内容被用户纠正"""
        return self.record_signal(BehaviorType.ACTIVE_RECALL, "correction")
    
    def on_interaction(self):
        """记录一次普通交互"""
        self.profile.total_interactions += 1
        if self.should_trigger_active_recall():
            self.profile.total_active_recalls_triggered += 1
    
    # ============ 统计与调试 ============
    
    def get_behavior_stats(self, behavior: BehaviorType) -> Dict[str, Any]:
        """获取行为统计"""
        behavior_key = behavior.value
        if behavior_key not in self.profile.behaviors:
            return {"enabled": True, "interest_score": 0.5, "signals": 0}
        
        pref = self.profile.behaviors[behavior_key]
        recent_signals = pref.signal_history[-10:] if pref.signal_history else []
        
        return {
            "enabled": pref.enabled,
            "interest_score": pref.interest_score,
            "signals_collected": pref.signals_collected,
            "recent_signal_trend": recent_signals,
            "last_evaluated": pref.last_evaluated,
        }
    
    def get_profile_summary(self) -> Dict[str, Any]:
        """获取人格侧写摘要"""
        recall_stats = self.get_behavior_stats(BehaviorType.ACTIVE_RECALL)
        
        return {
            "user_id": self.profile.user_id,
            "total_interactions": self.profile.total_interactions,
            "active_recalls_triggered": self.profile.total_active_recalls_triggered,
            "user_accepted": self.profile.total_user_accepted,
            "user_rejected": self.profile.total_user_rejected,
            "active_recall_preference": recall_stats,
        }
    
    def export_profile(self) -> Dict[str, Any]:
        """导出用户偏好配置（用于持久化）"""
        return {
            "user_id": self.profile.user_id,
            "created_at": self.profile.created_at,
            "updated_at": self.profile.updated_at,
            "behaviors": {
                k: {
                    "enabled": v.enabled,
                    "interest_score": v.interest_score,
                    "signals_collected": v.signals_collected,
                    "signal_history": v.signal_history[-20:],  # 只保留最近20条
                    "last_evaluated": v.last_evaluated,
                }
                for k, v in self.profile.behaviors.items()
            },
            "total_interactions": self.profile.total_interactions,
            "total_active_recalls_triggered": self.profile.total_active_recalls_triggered,
            "total_user_accepted": self.profile.total_user_accepted,
            "total_user_rejected": self.profile.total_user_rejected,
        }
    
    @classmethod
    def from_profile(cls, profile_data: Dict[str, Any]) -> "PersonaLayer":
        """从导出的配置加载"""
        profile = UserPersonaProfile(
            user_id=profile_data.get("user_id", "default"),
            created_at=profile_data.get("created_at", time.time()),
            updated_at=profile_data.get("updated_at", time.time()),
            total_interactions=profile_data.get("total_interactions", 0),
            total_active_recalls_triggered=profile_data.get("total_active_recalls_triggered", 0),
            total_user_accepted=profile_data.get("total_user_accepted", 0),
            total_user_rejected=profile_data.get("total_user_rejected", 0),
        )
        
        behaviors_data = profile_data.get("behaviors", {})
        for behavior_key, behavior_data in behaviors_data.items():
            profile.behaviors[behavior_key] = BehaviorPreference(
                enabled=behavior_data.get("enabled", True),
                interest_score=behavior_data.get("interest_score", 0.5),
                signals_collected=behavior_data.get("signals_collected", 0),
                signal_history=behavior_data.get("signal_history", []),
                last_evaluated=behavior_data.get("last_evaluated", 0.0),
            )
        
        return cls(profile=profile)


# ============ 测试 ============

if __name__ == "__main__":
    import random
    
    print("=" * 60)
    print("人格适应层 - 测试")
    print("=" * 60)
    
    layer = PersonaLayer()
    
    print("\n[1] 初始状态...")
    print(f"  主动提及开启: {layer.is_behavior_enabled(BehaviorType.ACTIVE_RECALL)}")
    print(f"  兴趣度: {layer.get_interest_score(BehaviorType.ACTIVE_RECALL):.2f}")
    
    print("\n[2] 模拟用户反馈...")
    # 模拟：用户连续3次对主动提及表示惊喜
    for i in range(3):
        score = layer.on_active_recall_explicit_positive()
        print(f"  第{i+1}次惊喜反馈 → 兴趣度: {score:.2f}")
    
    # 模拟：用户继续聊话题
    score = layer.on_active_recall_continue()
    print(f"  继续聊话题 → 兴趣度: {score:.2f}")
    
    # 模拟：用户转移话题
    score = layer.on_active_recall_ignore()
    print(f"  转移话题 → 兴趣度: {score:.2f}")
    
    print("\n[3] 当前状态...")
    print(f"  主动提及开启: {layer.is_behavior_enabled(BehaviorType.ACTIVE_RECALL)}")
    print(f"  兴趣度: {layer.get_interest_score(BehaviorType.ACTIVE_RECALL):.2f}")
    print(f"  统计: {layer.get_profile_summary()}")
    
    print("\n[4] 模拟用户持续不感兴趣...")
    for i in range(10):
        layer.on_active_recall_explicit_negative()
    print(f"  兴趣度: {layer.get_interest_score(BehaviorType.ACTIVE_RECALL):.2f}")
    print(f"  主动提及开启: {layer.is_behavior_enabled(BehaviorType.ACTIVE_RECALL)}")
    
    print("\n[5] 导出配置...")
    exported = layer.export_profile()
    print(f"  导出: {exported}")
    
    print("\n[6] 从配置加载...")
    new_layer = PersonaLayer.from_profile(exported)
    print(f"  兴趣度: {new_layer.get_interest_score(BehaviorType.ACTIVE_RECALL):.2f}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
