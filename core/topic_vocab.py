"""
统一主题词汇表 - 查询侧与写入侧的共同语言

背景：此前查询解析（retrieval.parse_query）产出英文主题标签
（"food"、"work"...），而写入侧抽取器（entity_extractor）产出中文
主题标签（"美食"、"工作"...）。两套词汇零交集，导致自然语言查询
永远无法通过主题锚点命中自动抽取的记忆——检索与唤醒双双失效。

本模块提供单一事实来源：
- SYNONYM_GROUPS: 双语同义标签组
- expand_topics(): 把任意一侧的标签扩展为整个同义组，
  使集合交集匹配（matches_query / calc_wake_score / topic_index）
  不再依赖双方恰好使用同一语言
- extract_query_topics(): 查询侧的表面词 -> 主题标签

写入侧数据无需迁移：已存储的中文/英文标签都会被查询侧的
扩展标签覆盖到。
"""

from typing import Dict, Iterable, List, Set

# 双语同义标签组：同组内任何标签都视为同一主题
SYNONYM_GROUPS: List[Set[str]] = [
    {"food", "dining", "meal", "美食", "餐饮"},
    {"travel", "trip", "旅行", "出行"},
    {"work", "business", "工作"},
    {"meeting", "会议"},
    {"project", "项目"},
    {"tech", "programming", "coding", "技术", "编程"},
    {"test", "testing", "测试"},
    {"entertainment", "movie", "film", "娱乐", "影视"},
    {"music", "音乐"},
    {"health", "sport", "exercise", "健康", "运动"},
    {"education", "study", "learning", "教育", "学习"},
    {"experience", "经验"},
    {"procedure", "policy", "程序", "策略"},
]

# 表面词 -> 主题标签（查询侧使用；与 entity_extractor.TOPIC_KEYWORDS 对齐）
SURFACE_KEYWORDS: Dict[str, Set[str]] = {
    "吃": {"美食", "餐饮"},
    "饭": {"美食", "餐饮"},
    "餐厅": {"美食", "餐饮"},
    "旅行": {"旅行", "出行"},
    "出差": {"工作", "出行"},
    "会议": {"工作", "会议"},
    "项目": {"工作", "项目"},
    "代码": {"技术", "编程"},
    "程序": {"技术", "编程"},
    "测试": {"技术", "测试"},
    "电影": {"娱乐", "影视"},
    "音乐": {"娱乐", "音乐"},
    "运动": {"健康", "运动"},
    "学习": {"教育", "学习"},
}

# 术语 -> 同义组 的反向索引（模块加载时构建一次）
_TERM_TO_GROUP: Dict[str, Set[str]] = {}
for _group in SYNONYM_GROUPS:
    for _term in _group:
        _TERM_TO_GROUP[_term.lower()] = _group


def expand_topics(topics: Iterable[str]) -> Set[str]:
    """
    把标签集合扩展为包含所有同义形式的集合。

    未收录的标签原样保留，因此扩展永远不会丢失信息。
    """
    expanded: Set[str] = set()
    for topic in topics:
        expanded.add(topic)
        group = _TERM_TO_GROUP.get(topic.lower())
        if group:
            expanded.update(group)
    return expanded


def count_topic_groups(topics: Iterable[str]) -> int:
    """
    统计标签集合覆盖的"概念组"数量。

    同一同义组的多个标签只算一个概念；未收录的标签各算一个。
    唤醒得分等比例计算必须用概念数做分母——否则 expand_topics
    扩展出的同义形式会人为放大分母，稀释真实的匹配强度。
    """
    seen_groups = set()
    ungrouped = 0
    for topic in topics:
        group = _TERM_TO_GROUP.get(topic.lower())
        if group is None:
            ungrouped += 1
        else:
            seen_groups.add(id(group))
    return len(seen_groups) + ungrouped


def extract_query_topics(text: str) -> Set[str]:
    """
    从查询文本提取主题标签（已扩展同义形式）。
    """
    topics: Set[str] = set()
    for keyword, tags in SURFACE_KEYWORDS.items():
        if keyword in text:
            topics.update(tags)
    return expand_topics(topics)
