"""
人格适应层 - 类人记忆系统

核心理念：
- 系统不预设人格，而是通过用户反馈学习用户偏好
- 行为初始默认开启，通过反馈逐渐调整
- 每个行为偏好独立追踪，最终构建用户人格侧写

核心行为：主动提及
- 系统默认主动提及旧记忆
- 用户通过反馈告诉系统他是否喜欢这个行为
- 系统收集足够的信号后，决定开启或关闭该行为
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum


@dataclass
class BehaviorPreference:
    """单个行为的偏好"""
    enabled: bool = True           # 默认开启
    interest_score: float = 0.5    # 初始兴趣度（中立）
    signals_collected: int = 0     # 收集到的信号数量
    signals_history: List[float] = field(default_factory=list)  # 信号历史
    
    # 学习阈值
    LEARNING_THRESHOLD = 10        # 收集多少信号后开始调整
    RECHECK_THRESHOLD = 20         # 之后每多少信号重新评估
    
    def add_signal(self, signal: float):
        """
        添加反馈信号
        signal > 0: 正向反馈（用户喜欢这个行为）
        signal < 0: 负向反馈（用户不喜欢这个行为）
        """
        self.signals_collected += 1
        self.signals_history.append(signal)
        
        # 只保留最近50个信号
        if len(self.signals_history) > 50:
            self.signals_history = self.signals_history[-50:]
        
        # 更新兴趣度（指数移动平均）
        self.interest_score = (
            self.interest_score * 0.8 
            + signal * 0.2
        )
        
        # 阈值触发决策
        if self.signals_collected <= self.LEARNING_THRESHOLD:
            # 还在收集阶段，保持默认
            return
        
        if self.signals_collected == self.LEARNING_THRESHOLD + 1:
            # 第一次达到阈值，做出决策
            self._make_decision()
        elif (self.signals_collected - self.LEARNING_THRESHOLD) % self.RECHECK_THRESHOLD == 0:
            # 之后定期重新评估
            self._make_decision()
    
    def _make_decision(self):
        """根据当前兴趣度做出开关决策"""
        if self.interest_score > 0.6:
            self.enabled = True
        elif self.interest_score < 0.35:
            self.enabled = False
        # 0.35~0.6之间保持现状
    
    def should_show(self) -> bool:
        """是否应该执行这个行为"""
        return self.enabled


class PersonaLayer:
    """
    人格适应层
    
    追踪用户对系统各种行为的偏好，
    通过反馈信号逐渐构建用户的"人格侧写"。
    
    当前追踪的行为：
    1. active_recall: 主动提及旧记忆
    2. detailed_response: 详细回复 vs 简洁回复
    3. emotional_sharing: 分享系统自身感受
    4. challenging_opinions: 挑战用户观点
    
    扩展设计：
    以后可以在这里添加更多行为偏好的追踪
    """
    
    # 主动提及的反馈信号
    SIGNAL_POSITIVE_STRONG = 0.3   # "你居然记得这个！"
    SIGNAL_POSITIVE = 0.2          # 继续聊这个话题
    SIGNAL_NEUTRAL = 0.05          # 轻微关注
    SIGNAL_NEGATIVE = -0.15        # 转移话题
    SIGNAL_NEGATIVE_STRONG = -0.4   # "别老提这个"
    
    # 主动提及的触发限制
    MAX_RECALLS_PER_CONVERSATION = 3  # 一轮对话最多主动提及几次
    RECALL_COOLDOWN = 5              # 触发一次后至少隔几轮再触发
    
    def __init__(self):
        # 各行为的偏好
        self.preferences: Dict[str, BehaviorPreference] = {
            "active_recall": BehaviorPreference(
                enabled=True,  # 默认主动提及
                interest_score=0.5,
            ),
            "detailed_response": BehaviorPreference(
                enabled=False,
                interest_score=0.5,
            ),
            "emotional_sharing": BehaviorPreference(
                enabled=True,
                interest_score=0.5,
            ),
            "challenging_opinions": BehaviorPreference(
                enabled=False,
                interest_score=0.5,
            ),
        }
        
        # 主动提及的会话状态
        self.conversation_recall_count = 0  # 本轮已主动提及次数
        self.conversation_turns = 0         # 本轮对话轮数
        self.last_recall_turn = -999        # 上次主动提及是哪一轮
        self.conversation_id: Optional[str] = None
        
        # 用户ID（如果有）
        self.user_id: Optional[str] = None
    
    def start_conversation(self, conversation_id: str):
        """开始新的一轮对话"""
        if self.conversation_id != conversation_id:
            # 新对话，重置计数
            self.conversation_id = conversation_id
            self.conversation_recall_count = 0
            self.conversation_turns = 0
            self.last_recall_turn = -999
    
    def end_conversation(self):
        """结束当前对话"""
        self.conversation_recall_count = 0
        self.conversation_turns = 0
    
    def record_turn(self):
        """记录一轮对话"""
        self.conversation_turns += 1
    
    def should_active_recall(
        self,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        判断是否应该主动提及旧记忆
        
        检查条件：
        1. 用户偏好是否开启
        2. 本轮对话是否还有配额
        3. 是否在冷却期
        """
        context = context or {}
        pref = self.preferences["active_recall"]
        
        if not pref.should_show():
            return False
        
        if self.conversation_recall_count >= self.MAX_RECALLS_PER_CONVERSATION:
            return False
        
        if self.conversation_turns - self.last_recall_turn < self.RECALL_COOLDOWN:
            return False
        
        # 可以主动提及
        return True
    
    def record_active_recall(self):
        """记录一次主动提及"""
        self.conversation_recall_count += 1
        self.last_recall_turn = self.conversation_turns
    
    def receive_feedback(
        self,
        behavior: str,
        feedback_type: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        """
        接收用户对某行为的反馈
        
        Args:
            behavior: 行为名称 ("active_recall", etc.)
            feedback_type: 反馈类型
            context: 额外上下文
        """
        if behavior not in self.preferences:
            return
        
        pref = self.preferences[behavior]
        signal = self._get_signal(behavior, feedback_type, context)
        pref.add_signal(signal)
    
    def _get_signal(
        self, 
        behavior: str, 
        feedback_type: str,
        context: Optional[Dict[str, Any]],
    ) -> float:
        """根据反馈类型获取信号值"""
        context = context or {}
        
        if behavior == "active_recall":
            if feedback_type == "continued_discussion":
                return self.SIGNAL_POSITIVE
            elif feedback_type == "expressed_pleasure":
                return self.SIGNAL_POSITIVE_STRONG
            elif feedback_type == "changed_topic":
                return self.SIGNAL_NEGATIVE
            elif feedback_type == "explicitly_disapprove":
                return self.SIGNAL_NEGATIVE_STRONG
            elif feedback_type == "corrected_content":
                return self.SIGNAL_NEUTRAL  # 纠正说明在关注，算正向
            elif feedback_type == "user_raised":
                return 0.2  # 用户主动提起旧事，正向
        elif behavior == "detailed_response":
            if feedback_type == "user_read_full":
                return self.SIGNAL_POSITIVE
            elif feedback_type == "user_skipped":
                return self.SIGNAL_NEGATIVE
        elif behavior == "emotional_sharing":
            if feedback_type == "user_responded_positively":
                return self.SIGNAL_POSITIVE
            elif feedback_type == "user_ignored":
                return self.SIGNAL_NEGATIVE
        
        return 0.0
    
    def get_user_persona_summary(self) -> Dict[str, Any]:
        """
        获取用户人格侧写摘要
        """
        summary = {}
        for name, pref in self.preferences.items():
            summary[name] = {
                "enabled": pref.enabled,
                "interest_score": pref.interest_score,
                "signals_collected": pref.signals_collected,
            }
        return summary
    
    def get_active_recall_status(self) -> Dict[str, Any]:
        """获取主动提及的状态（用于调试）"""
        pref = self.preferences["active_recall"]
        return {
            "enabled": pref.enabled,
            "interest_score": pref.interest_score,
            "signals_collected": pref.signals_collected,
            "conversation_recall_count": self.conversation_recall_count,
            "conversation_turns": self.conversation_turns,
            "can_recall": self.should_active_recall(),
        }


# ============ 测试 ============

if __name__ == "__main__":
    print("=" * 60)
    print("人格适应层 - 测试")
    print("=" * 60)
    
    layer = PersonaLayer()
    
    print("\n[1] 初始状态...")
    print(f"  主动提及: {layer.get_active_recall_status()}")
    
    print("\n[2] 模拟对话...")
    layer.start_conversation("conv_001")
    
    # 第1轮：系统尝试主动提及
    layer.record_turn()
    if layer.should_active_recall():
        print("  轮次1: 系统主动提及旧记忆")
        layer.record_active_recall()
    else:
        print("  轮次1: 不适合主动提及")
    
    # 第2-5轮
    for turn in range(2, 6):
        layer.record_turn()
        print(f"  轮次{turn}: can_recall={layer.should_active_recall()}")
    
    print("\n[3] 收集反馈...")
    # 用户表示喜欢
    layer.receive_feedback("active_recall", "continued_discussion")
    print(f"  用户继续讨论了该话题, interest_score={layer.preferences['active_recall'].interest_score:.2f}")
    
    layer.receive_feedback("active_recall", "expressed_pleasure")
    print(f"  用户表示惊喜, interest_score={layer.preferences['active_recall'].interest_score:.2f}")
    
    print("\n[4] 继续对话...")
    layer.record_turn()
    if layer.should_active_recall():
        print("  系统决定再次主动提及")
        layer.record_active_recall()
    
    # 收集更多信号
    for _ in range(8):
        layer.receive_feedback("active_recall", "continued_discussion")
    
    print(f"\n[5] 达到学习阈值后的决策...")
    print(f"  收集了{layer.preferences['active_recall'].signals_collected}个信号")
    print(f"  interest_score={layer.preferences['active_recall'].interest_score:.2f}")
    print(f"  enabled={layer.preferences['active_recall'].enabled}")
    
    print("\n[6] 人格侧写摘要...")
    summary = layer.get_user_persona_summary()
    for behavior, status in summary.items():
        print(f"  {behavior}: {status}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
