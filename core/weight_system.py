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
