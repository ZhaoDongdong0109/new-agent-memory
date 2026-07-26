"""
记忆类型与衰减节奏 - 类人记忆系统

记忆科学的基本事实：不同类型的记忆有不同的遗忘曲线。
程序性/叙事性记忆（怎么做事、亲身经历的故事）在人脑中远比
一次性交互细节持久；偏好和事实居中。

本模块是这一分层的单一事实来源：
- MemoryType: 记忆类型枚举（贯穿 chunk 数据结构与抽取管线）
- HALFLIFE_MULTIPLIER_BY_TYPE: 各类型相对基准半衰期的倍率，
  由 MemoryLayerCore.calc_weight 在实时权重计算中使用

历史说明：早期版本在这里实现了一套独立的 AdaptiveWeightSystem +
MemoryItem（基于互动次数的衰减、情绪系数采样、注意力漂移）。
那套系统从未被运行时实例化过，与实际执行的 MemoryLayerCore 权重
模型是两套平行的现实。它的核心思想——按类型区分衰减节奏——
已经并入实时模型（见下方倍率表，比例继承自原 HALFLIFE_BY_TYPE
500/400/300/200/150），其余死代码已删除。
"""

from enum import Enum


class MemoryType(Enum):
    """记忆类型，影响衰减节奏"""
    STORY = "story"             # 用户故事/经历，衰减最慢
    IDEA = "idea"               # 想法/观点，中慢
    PREFERENCE = "preference"   # 用户偏好，中等
    FACT = "fact"               # 事实信息，中快
    INTERACTION = "interaction" # 我们之间发生的事，衰减最快


# 各类型记忆的半衰期倍率（以 INTERACTION 为基准 1.0）。
# 比例继承自原始设计的互动次数半衰期表 500/400/300/200/150。
# MemoryLayerCore 用 decay_half_life * multiplier 得到该类型的
# 实际时间半衰期，因此"故事比交互细节持久 3 倍多"这一原始意图
# 现在是运行时行为，而不是文档里的愿望。
HALFLIFE_MULTIPLIER_BY_TYPE = {
    MemoryType.STORY: 500 / 150,        # ≈3.33
    MemoryType.IDEA: 400 / 150,         # ≈2.67
    MemoryType.PREFERENCE: 300 / 150,   # 2.0
    MemoryType.FACT: 200 / 150,         # ≈1.33
    MemoryType.INTERACTION: 1.0,
}


def halflife_multiplier(memory_type) -> float:
    """
    返回某个记忆类型的半衰期倍率。

    容忍字符串值（历史数据可能存的是 "story" 而不是枚举）。
    未知类型使用 1.0（基准衰减）。
    """
    if isinstance(memory_type, str):
        try:
            memory_type = MemoryType(memory_type)
        except ValueError:
            return 1.0
    return HALFLIFE_MULTIPLIER_BY_TYPE.get(memory_type, 1.0)


# ============ ACT-R 基线激活参数 ============
#
# ACT-R 的记忆强度模型（Anderson & Schooler 1991，30 年实证验证）：
#
#   B_i = ln( Σ_j t_j^(-d) )     t_j = 距第 j 次使用的时间
#   P   = 1 / (1 + exp(-(B - τ) / s))   （检索概率，逻辑斯蒂映射）
#
# 一个方程同时产生：幂律遗忘（比指数衰减更符合人类数据）、
# 频率效应（用得多记得牢）、近因效应（刚用过记得清）。
# 注：间隔效应需要 Pavlik & Anderson (2005) 的激活依赖衰减扩展，
# 朴素 BLA 不包含它——留作下一轮（需要按次记录衰减率状态）。
#
# d 是衰减速率：越小衰减越慢。按记忆类型分层
# （故事最持久 -> 交互细节最易忘），持久性排序与
# HALFLIFE_MULTIPLIER_BY_TYPE 一致。

ACTR_DECAY_BY_TYPE = {
    MemoryType.STORY: 0.38,
    MemoryType.IDEA: 0.42,
    MemoryType.PREFERENCE: 0.44,
    MemoryType.FACT: 0.47,
    MemoryType.INTERACTION: 0.50,   # ACT-R 默认 d=0.5
}

# 逻辑斯蒂映射参数：τ 为激活阈值，s 为噪声尺度。
# 校准锚点（INTERACTION，单次编码）：
#   1 分钟前编码  -> B≈-2.0 -> P≈0.97（刚记住）
#   7 天未使用    -> B≈-6.7 -> P≈0.24（半衰附近，对应原 7 天半衰期设计）
#   120 天未使用  -> B≈-8.1 -> P≈0.07（可被降级）
ACTR_THRESHOLD = -5.5
ACTR_NOISE_SCALE = 1.0


def actr_decay(memory_type) -> float:
    """返回某个记忆类型的 ACT-R 衰减速率 d。容忍字符串值。"""
    if isinstance(memory_type, str):
        try:
            memory_type = MemoryType(memory_type)
        except ValueError:
            return 0.50
    return ACTR_DECAY_BY_TYPE.get(memory_type, 0.50)
