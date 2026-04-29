"""
情绪推断引擎 - 类人记忆系统

核心设计：
- 情绪系数不是固定值，而是带有随机分布的概率变量
- 同一情绪标签在不同上下文下可能对应不同的系数
- 越极端的情绪，spread越大，不确定性越高
"""

import random
import math
from typing import Optional, Dict, Any
from dataclasses import dataclass


# 情绪参数：base是均值，spread是标准差
# spread越大，同样的情绪标签在不同场景下系数波动越大
EMOTION_PARAMS = {
    "平静":     {"base": 0.7, "spread": 0.2},
    "开心":     {"base": 1.0, "spread": 0.3},
    "好奇":     {"base": 1.1, "spread": 0.3},
    "困惑":     {"base": 1.3, "spread": 0.4},
    "期待":     {"base": 1.4, "spread": 0.4},
    "难过":     {"base": 1.6, "spread": 0.5},
    "焦虑":     {"base": 1.8, "spread": 0.6},
    "兴奋":     {"base": 1.8, "spread": 0.5},
    "愤怒":     {"base": 2.0, "spread": 0.7},
    "恐惧":     {"base": 2.2, "spread": 0.7},
    "崩溃":     {"base": 2.5, "spread": 0.8},
}

# 默认参数（未知情绪）
DEFAULT_PARAMS = {"base": 1.0, "spread": 0.5}


@dataclass
class EmotionResult:
    """情绪推断结果"""
    emotion_tag: str           # 情绪标签
    coefficient: float          # 采样得到的情绪系数
    base_value: float           # 基础均值
    spread_value: float        # 分布标准差
    confidence: float           # 推断置信度 (0~1)
    user_override: Optional[str] = None  # 用户是否覆盖了推断


class EmotionEngine:
    """
    情绪推断引擎
    
    设计原则：
    - 系统先推断，用户可以纠正
    - 情绪系数由概率分布采样得出，而非固定查表
    - 上下文方差让同一情绪在不同场景下系数不同
    - 用户纠正行为本身也是信号，会被记录用于系统校准
    """
    
    def __init__(self):
        # 全局互动计数器（用于上下文方差计算）
        self.global_interaction_count = 0
        
        # 用户纠正历史（用于识别系统偏差）
        # emotion_tag -> list of (inferred_coef, user_corrected_coef)
        self.user_corrections: Dict[str, list] = {}
        
    def _calculate_context_variance(
        self,
        text: str,
        topic_repeat_count: int = 0,
    ) -> float:
        """
        计算上下文方差
        方差会让情绪系数在基础值上下波动
        """
        variance = 0.0
        
        # 同一话题重复提及 → 方差增加
        variance += topic_repeat_count * 0.05
        
        # 标点符号强度
        exclamation_count = text.count('!')
        question_count = text.count('?')
        ellipsis_count = text.count('...')
        
        variance += min(exclamation_count, 3) * 0.08
        variance += min(question_count, 3) * 0.05
        variance += min(ellipsis_count, 2) * 0.1
        
        # 句子长度异常值
        words = text.split()
        if len(words) > 50:
            variance += (len(words) - 50) * 0.01
        
        # 全部大写（情绪强烈）
        if text.isupper() and len(text) > 10:
            variance += 0.3
        
        return variance
    
    def infer_emotion(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
        user_override: Optional[str] = None,
    ) -> EmotionResult:
        """
        推断文本的情绪标签和系数
        
        Args:
            text: 用户输入的文本
            context: 上下文信息，包含：
                - topic_repeat_count: 当前话题在对话中重复了多少次
                - interaction_count: 全局互动计数
                - 最近的话题列表
            user_override: 用户手动指定的情绪标签（可选）
        """
        context = context or {}
        topic_repeat = context.get("topic_repeat_count", 0)
        
        # 用户指定了情绪 → 使用用户指定的，但仍然采样（保持不确定性）
        if user_override:
            emotion_tag = user_override
        else:
            # 系统推断（这里用简单规则，后续可替换为LLM）
            emotion_tag = self._system_infer(text)
        
        # 获取情绪参数
        params = EMOTION_PARAMS.get(emotion_tag, DEFAULT_PARAMS)
        base = params["base"]
        spread = params["spread"]
        
        # 计算上下文方差
        context_variance = self._calculate_context_variance(text, topic_repeat)
        
        # 采样得到最终系数（高斯分布）
        raw_coef = random.gauss(base + context_variance, spread)
        
        # 系数范围限制（允许一定超限，但设软天花板）
        # 软天花板：超出范围的概率本来就很低，这里只是保险
        coefficient = max(0.3, min(4.0, raw_coef))
        
        # 推断置信度：如果用户之前纠正过类似情绪，置信度降低
        confidence = self._calculate_confidence(emotion_tag)
        
        return EmotionResult(
            emotion_tag=emotion_tag,
            coefficient=coefficient,
            base_value=base,
            spread_value=spread,
            confidence=confidence,
            user_override=user_override if user_override else None,
        )
    
    def _system_infer(self, text: str) -> str:
        """
        系统端情绪推断（简单规则版）
        后续可以替换为LLM推断
        """
        text_lower = text.lower()
        
        # 关键词匹配（简化版）
        emotion_keywords = {
            "崩溃": ["崩溃", "受不了", "绝望", "完了"],
            "愤怒": ["生气", "愤怒", "恼火", "讨厌", "烦死了"],
            "恐惧": ["害怕", "担心", "怕", "焦虑", "紧张", "慌"],
            "兴奋": ["太棒了", "激动", "兴奋", "开心死了", "超开心"],
            "焦虑": ["焦虑", "纠结", "不安", "忐忑"],
            "难过": ["难过", "伤心", "失落", "沮丧", "郁闷"],
            "期待": ["期待", "希望", "憧憬", "想"],
            "困惑": ["困惑", "不懂", "怎么回事", "为什么", "?"],
            "好奇": ["好奇", "想知道", "问问"],
            "开心": ["开心", "高兴", "快乐", "哈哈"],
            "平静": ["平静", "淡定", "还好", "一般"],
        }
        
        # 匹配得分
        scores = {}
        for emotion, keywords in emotion_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[emotion] = score
        
        if not scores:
            return "平静"
        
        # 返回得分最高的情绪
        return max(scores, key=scores.get)
    
    def _calculate_confidence(self, emotion_tag: str) -> float:
        """根据用户纠正历史计算置信度"""
        if emotion_tag not in self.user_corrections:
            return 0.8  # 初始置信度较高
        
        corrections = self.user_corrections[emotion_tag]
        if len(corrections) < 3:
            return 0.7
        
        # 如果用户频繁纠正，置信度降低
        return max(0.3, 0.8 - len(corrections) * 0.05)
    
    def record_correction(
        self,
        emotion_tag: str,
        inferred_coef: float,
        corrected_coef: float,
    ):
        """
        记录用户的情绪纠正行为
        这个信号会用于调整未来的推断置信度
        """
        if emotion_tag not in self.user_corrections:
            self.user_corrections[emotion_tag] = []
        
        self.user_corrections[emotion_tag].append((inferred_coef, corrected_coef))
        
        # 保留最近20条纠正记录
        if len(self.user_corrections[emotion_tag]) > 20:
            self.user_corrections[emotion_tag] = self.user_corrections[emotion_tag][-20:]
    
    def get_calibration_hint(self, emotion_tag: str) -> Optional[Dict[str, float]]:
        """
        获取校准提示
        如果系统持续对某种情绪推断偏差，用户可以查看并手动调整参数
        """
        if emotion_tag not in self.user_corrections:
            return None
        
        corrections = self.user_corrections[emotion_tag]
        if len(corrections) < 5:
            return None
        
        # 计算平均偏差
        avg_diff = sum(
            corrected - inferred 
            for inferred, corrected in corrections
        ) / len(corrections)
        
        return {
            "avg_bias": avg_diff,
            "sample_count": len(corrections),
            "suggested_adjustment": avg_diff * 0.3,  # 建议的调整量
        }


# ============ 测试 ============

if __name__ == "__main__":
    engine = EmotionEngine()
    
    print("=" * 60)
    print("情绪推断引擎 - 测试")
    print("=" * 60)
    
    test_texts = [
        "今天工作特别顺利，老板夸我了！",
        "我真的很焦虑，不知道该怎么办...",
        "崩溃了，完全不知道哪里出问题了！",
        "还好吧，今天就那样。",
        "我想知道你这个设计是怎么做出来的？",
    ]
    
    print("\n[1] 情绪推断测试（同一文本采样5次）...")
    test_text = "我最近特别焦虑这件事，晚上都睡不好..."
    
    for i in range(5):
        result = engine.infer_emotion(test_text)
        print(f"  第{i+1}次采样: {result.emotion_tag}, 系数={result.coefficient:.3f}")
    
    print(f"\n[2] 不同文本的情绪推断...")
    for text in test_texts:
        result = engine.infer_emotion(text)
        print(f"  文本: {text[:20]}...")
        print(f"    → {result.emotion_tag}, 系数={result.coefficient:.3f} (base={result.base_value}, spread={result.spread_value})")
    
    print("\n[3] 用户纠正测试...")
    engine.record_correction("焦虑", 1.8, 2.3)
    engine.record_correction("焦虑", 1.9, 2.1)
    engine.record_correction("焦虑", 1.7, 2.4)
    
    hint = engine.get_calibration_hint("焦虑")
    if hint:
        print(f"  检测到系统对'焦虑'情绪持续低估")
        print(f"  平均偏差: {hint['avg_bias']:.3f}")
        print(f"  建议调整: {hint['suggested_adjustment']:.3f}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
