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
    PROCEDURE = "procedure"     # 程序性知识/操作流程，最持久（骑车不会忘）
    STORY = "story"             # 用户故事/经历，衰减很慢
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
    MemoryType.PROCEDURE: 600 / 150,    # 4.0（程序性记忆最持久）
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
    MemoryType.PROCEDURE: 0.35,
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

# ============ Pavlik & Anderson (2005) 间隔效应 ============
#
# 朴素 BLA 的盲区：每次使用贡献相同的痕迹，突击复习 5 次与
# 分散复习 5 次编码强度一样——与 100 年的间隔效应实证相悖。
# Pavlik & Anderson 的扩展：每次使用事件的衰减速率取决于
# 复习瞬间的激活水平：
#
#   d_j = c * e^(m_j) + α        m_j = 第 j 次使用时的激活 B
#
# 激活很高时复习（刚用过就再用）-> 该次痕迹衰减快，边际收益小；
# 激活接近阈值时复习（快忘了才复习）-> 痕迹接近基线衰减，最耐久。
#
# 本移植取 α = 类型基线 d（保持既有单事件校准锚点不动：首次
# 编码前激活为 -inf，e^(-inf)=0，d_0 = 类型 d），c 沿用论文
# 拟合值 0.277。上限 0.95 保证幂律积分收敛。
PAVLIK_SPACING_C = 0.277
PAVLIK_DECAY_CEIL = 0.95


def pavlik_event_decay(activation: float, base_d: float) -> float:
    """某次使用事件的衰减速率：复习时激活越高，该次痕迹越易衰减"""
    import math
    return min(PAVLIK_DECAY_CEIL, base_d + PAVLIK_SPACING_C * math.exp(activation))


def actr_decay(memory_type) -> float:
    """返回某个记忆类型的 ACT-R 衰减速率 d。容忍字符串值。"""
    if isinstance(memory_type, str):
        try:
            memory_type = MemoryType(memory_type)
        except ValueError:
            return 0.50
    return ACTR_DECAY_BY_TYPE.get(memory_type, 0.50)
