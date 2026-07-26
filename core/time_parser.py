"""
确定性时间表达解析 - 类人记忆系统

把查询里的自然语言时间表达解析为数值时间窗口 [t_start, t_end]，
把记忆的 time_absolute 解析为数值有效范围。纯 stdlib、规则表驱动、
可注入时钟（now_fn）——冻结时钟即可获得完全可复现的测试。

背景（LongMemEval 消融实验）：时间感知的索引与查询扩展带来 +11.3%
召回提升，是已验证的最大单机制增益。此前系统对时间字段做精确字符串
相等匹配（"昨天" 只能匹配 "昨天"），任何换一种说法的时间线索都失效。

设计约束：查询里没有时间表达时返回 None，管线行为与原来逐字节一致。
"""

from __future__ import annotations

import calendar
import re
import time
from datetime import datetime, timedelta
from typing import Callable, Optional, Tuple

# 时间窗口：epoch 秒的闭区间
TimeWindow = Tuple[float, float]

# 现在时/当前态标记：出现时查询问的是"当前事实"，
# 检索应排除已被取代（invalid_at 已设置）的记忆
PRESENT_MARKERS = ("现在", "目前", "当前", "如今", "current", "currently", "now", "these days")

# 过去时标记：查询显式问历史状态，被取代的记忆是合法答案
PAST_MARKERS = (
    "以前", "之前", "曾经", "过去", "原来",
    "used to", "use to", "previously", "before", "formerly",
)


def _day_range(dt: datetime) -> TimeWindow:
    start = datetime(dt.year, dt.month, dt.day)
    return start.timestamp(), (start + timedelta(days=1)).timestamp() - 1


def _month_range(year: int, month: int) -> TimeWindow:
    start = datetime(year, month, 1)
    last_day = calendar.monthrange(year, month)[1]
    end = datetime(year, month, last_day, 23, 59, 59)
    return start.timestamp(), end.timestamp()


def _year_range(year: int) -> TimeWindow:
    return datetime(year, 1, 1).timestamp(), datetime(year, 12, 31, 23, 59, 59).timestamp()


def _week_range(dt: datetime, weeks_back: int = 0) -> TimeWindow:
    monday = datetime(dt.year, dt.month, dt.day) - timedelta(days=dt.weekday(), weeks=weeks_back)
    return monday.timestamp(), (monday + timedelta(days=7)).timestamp() - 1


def parse_query_window(
    text: str,
    now_fn: Callable[[], float] = time.time,
) -> Optional[TimeWindow]:
    """
    解析查询文本中的时间表达为时间窗口。

    支持（中英文）：昨天/前天/今天、上周、上个月、去年、
    N天前、N个月前、N年前、YYYY年、YYYY年M月、YYYY-MM(-DD)、
    yesterday、last week/month/year、N days/months/years ago。

    无时间表达返回 None（管线保持原行为）。
    """
    if not text:
        return None

    lowered = text.lower()
    now = datetime.fromtimestamp(now_fn())

    # ---- 绝对时间（优先级最高，最具体）----
    m = re.search(r"(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})日?", lowered)
    if m:
        try:
            return _day_range(datetime(int(m.group(1)), int(m.group(2)), int(m.group(3))))
        except ValueError:
            pass

    m = re.search(r"(\d{4})[-/年](\d{1,2})月?(?![\d日])", lowered)
    if m:
        year, month = int(m.group(1)), int(m.group(2))
        if 1 <= month <= 12:
            return _month_range(year, month)

    m = re.search(r"(\d{4})年", lowered)
    if m:
        return _year_range(int(m.group(1)))

    # ---- 相对时间：中文 ----
    m = re.search(r"(\d+)\s*天前", lowered)
    if m:
        return _day_range(now - timedelta(days=int(m.group(1))))

    m = re.search(r"(\d+)\s*个月前", lowered)
    if m:
        months_back = int(m.group(1))
        year, month = now.year, now.month - months_back
        while month <= 0:
            month += 12
            year -= 1
        return _month_range(year, month)

    m = re.search(r"(\d+)\s*年前", lowered)
    if m:
        return _year_range(now.year - int(m.group(1)))

    if "前天" in lowered:
        return _day_range(now - timedelta(days=2))
    if "昨天" in lowered:
        return _day_range(now - timedelta(days=1))
    if "今天" in lowered:
        return _day_range(now)
    if "上周" in lowered or "上星期" in lowered:
        return _week_range(now, weeks_back=1)
    if "上个月" in lowered:
        year, month = (now.year, now.month - 1) if now.month > 1 else (now.year - 1, 12)
        return _month_range(year, month)
    if "去年" in lowered:
        return _year_range(now.year - 1)

    # ---- 相对时间：英文 ----
    m = re.search(r"(\d+)\s*days?\s+ago", lowered)
    if m:
        return _day_range(now - timedelta(days=int(m.group(1))))

    m = re.search(r"(\d+)\s*months?\s+ago", lowered)
    if m:
        months_back = int(m.group(1))
        year, month = now.year, now.month - months_back
        while month <= 0:
            month += 12
            year -= 1
        return _month_range(year, month)

    m = re.search(r"(\d+)\s*years?\s+ago", lowered)
    if m:
        return _year_range(now.year - int(m.group(1)))

    if "yesterday" in lowered:
        return _day_range(now - timedelta(days=1))
    if "last week" in lowered:
        return _week_range(now, weeks_back=1)
    if "last month" in lowered:
        year, month = (now.year, now.month - 1) if now.month > 1 else (now.year - 1, 12)
        return _month_range(year, month)
    if "last year" in lowered:
        return _year_range(now.year - 1)

    return None


def query_tense(text: str) -> Optional[str]:
    """
    识别查询的时态倾向：
    - "present": 问当前事实（应排除已被取代的记忆）
    - "past": 问历史状态（被取代的记忆是合法答案）
    - None: 无明确时态标记
    """
    if not text:
        return None
    lowered = text.lower()
    for marker in PAST_MARKERS:
        if marker in lowered:
            return "past"
    for marker in PRESENT_MARKERS:
        if marker in lowered:
            return "present"
    return None


def chunk_time_range(chunk) -> TimeWindow:
    """
    解析一条记忆的数值时间范围。

    优先解析 time_absolute（支持 YYYY-MM-DD / YYYY-MM / YYYY 及
    对应的中文格式）；解析失败时退回 created_at 的当日范围。
    """
    raw = getattr(chunk, "time_absolute", None)
    if raw:
        text = str(raw).strip()
        m = re.match(r"^(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})日?$", text)
        if m:
            try:
                return _day_range(datetime(int(m.group(1)), int(m.group(2)), int(m.group(3))))
            except ValueError:
                pass
        m = re.match(r"^(\d{4})[-/年](\d{1,2})月?$", text)
        if m:
            month = int(m.group(2))
            if 1 <= month <= 12:
                return _month_range(int(m.group(1)), month)
        m = re.match(r"^(\d{4})年?$", text)
        if m:
            return _year_range(int(m.group(1)))

    created = getattr(chunk, "created_at", None) or time.time()
    return _day_range(datetime.fromtimestamp(created))


def windows_overlap(a: TimeWindow, b: TimeWindow) -> bool:
    """两个时间窗口是否有交集"""
    return a[0] <= b[1] and b[0] <= a[1]
