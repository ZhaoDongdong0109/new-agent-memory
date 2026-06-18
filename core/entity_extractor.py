"""
实体/关系抽取器

从文本提取人物、地点、时间、主题、关键词、情绪。

使用规则 + 模式匹配，无需外部依赖。
"""

import re
from typing import Dict, List, Optional, Set, Tuple


class EntityExtractor:
    """
    实体/关系抽取器

    从文本提取结构化信息：
    - 人物（persons）
    - 地点（location）
    - 时间（time_absolute, time_relative, time_context）
    - 主题（topics）
    - 关键词（keywords）
    - 情绪（emotion_valence, emotion_intensity）
    """

    # 地点词典
    LOCATIONS = [
        "北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "南京",
        "家里", "公司", "办公室", "餐厅", "酒店", "机场", "车站",
        "学校", "医院", "图书馆", "公园", "商场",
    ]

    # 时间上下文词典
    TIME_CONTEXTS = {
        "早上": "早上", "早晨": "早上", "上午": "上午",
        "中午": "中午", "午饭": "中午", "午餐": "中午",
        "下午": "下午", "傍晚": "傍晚",
        "晚上": "晚上", "晚饭": "晚上", "晚餐": "晚上",
        "夜里": "夜里", "深夜": "深夜",
    }

    # 主题词典
    TOPIC_KEYWORDS = {
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

    # 正面情绪词
    POSITIVE_WORDS = [
        "开心", "高兴", "快乐", "满意", "喜欢", "爱", "好", "棒",
        "优秀", "成功", "顺利", "精彩", "美好", "幸福", "感谢",
    ]

    # 负面情绪词
    NEGATIVE_WORDS = [
        "难过", "伤心", "失望", "生气", "愤怒", "讨厌", "恨", "糟",
        "失败", "困难", "问题", "错误", "糟糕", "痛苦", "抱歉",
    ]

    def extract_persons(self, text: str) -> Set[str]:
        """
        提取人物

        模式：
        - 和XXX一起
        - XXX说
        - XXX的
        - 与XXX
        """
        persons = set()

        # 模式1：和/与/跟 + 人名 + 一起/吃饭/见...
        patterns = [
            r"和(.{1,4}?)(一起|吃饭|去|见|聊|讨论|开会)",
            r"与(.{1,4}?)(一起|吃饭|去|见|聊|讨论|开会)",
            r"跟(.{1,4}?)(一起|吃饭|去|见|聊|讨论|开会)",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                name = match[0].strip()
                if self._is_valid_name(name):
                    persons.add(name)

        return persons

    def extract_location(self, text: str) -> Optional[str]:
        """
        提取地点

        使用词典匹配。
        """
        for location in self.LOCATIONS:
            if location in text:
                return location
        return None

    def extract_time(
        self, text: str
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        提取时间

        Returns:
            (time_absolute, time_relative, time_context)
        """
        time_absolute = None
        time_relative = None
        time_context = None

        # 绝对时间：YYYY-MM-DD 或 YYYY年MM月DD日
        abs_patterns = [
            r"(\d{4})-(\d{1,2})-(\d{1,2})",
            r"(\d{4})年(\d{1,2})月(\d{1,2})日",
        ]
        for pattern in abs_patterns:
            match = re.search(pattern, text)
            if match:
                year, month, day = match.groups()
                time_absolute = f"{year}-{int(month):02d}-{int(day):02d}"
                break

        # 相对时间
        rel_patterns = {
            r"(\d+)年前": lambda m: f"{m.group(1)}年前",
            r"(\d+)天前": lambda m: f"{m.group(1)}天前",
            r"(\d+)个月前": lambda m: f"{m.group(1)}个月前",
            r"昨天": lambda m: "昨天",
            r"今天": lambda m: "今天",
            r"明天": lambda m: "明天",
            r"上周": lambda m: "上周",
            r"这周": lambda m: "这周",
            r"下周": lambda m: "下周",
            r"上个月": lambda m: "上个月",
            r"这个月": lambda m: "这个月",
            r"下个月": lambda m: "下个月",
            r"去年": lambda m: "去年",
            r"今年": lambda m: "今年",
            r"明年": lambda m: "明年",
        }

        for pattern, extractor in rel_patterns.items():
            match = re.search(pattern, text)
            if match:
                time_relative = extractor(match)
                break

        # 时间上下文
        for keyword, context in self.TIME_CONTEXTS.items():
            if keyword in text:
                time_context = context
                break

        return time_absolute, time_relative, time_context

    def extract_topics(self, text: str) -> Set[str]:
        """
        提取主题

        使用关键词 -> 主题映射。
        """
        topics = set()

        for keyword, topic_set in self.TOPIC_KEYWORDS.items():
            if keyword in text:
                topics.update(topic_set)

        return topics

    def extract_keywords(self, text: str, top_k: int = 10) -> Set[str]:
        """
        提取关键词

        简单实现：基于词频。
        """
        # 分词（简单实现）
        terms = self._tokenize(text)

        # 统计词频
        term_counts: Dict[str, int] = {}
        for term in terms:
            if len(term) >= 2:  # 过滤单字
                term_counts[term] = term_counts.get(term, 0) + 1

        # 按频率排序，取 top_k
        sorted_terms = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)
        return {term for term, _ in sorted_terms[:top_k]}

    def extract_emotion(self, text: str) -> Tuple[float, float]:
        """
        提取情绪

        Returns:
            (valence, intensity)
            valence: -1.0（负面）到 1.0（正面）
            intensity: 0.0（弱）到 1.0（强）
        """
        positive_count = 0
        negative_count = 0

        for word in self.POSITIVE_WORDS:
            if word in text:
                positive_count += 1

        for word in self.NEGATIVE_WORDS:
            if word in text:
                negative_count += 1

        total = positive_count + negative_count
        if total == 0:
            return 0.0, 0.0

        # 计算 valence
        valence = (positive_count - negative_count) / total

        # 计算 intensity（基于情绪词密度）
        intensity = min(1.0, total / 5.0)

        return valence, intensity

    def extract_all(self, text: str) -> Dict:
        """
        提取所有实体信息

        Returns:
            {
                "persons": Set[str],
                "location": Optional[str],
                "time_absolute": Optional[str],
                "time_relative": Optional[str],
                "time_context": Optional[str],
                "topics": Set[str],
                "keywords": Set[str],
                "emotion_valence": float,
                "emotion_intensity": float,
            }
        """
        persons = self.extract_persons(text)
        location = self.extract_location(text)
        time_absolute, time_relative, time_context = self.extract_time(text)
        topics = self.extract_topics(text)
        keywords = self.extract_keywords(text)
        emotion_valence, emotion_intensity = self.extract_emotion(text)

        return {
            "persons": persons,
            "location": location,
            "time_absolute": time_absolute,
            "time_relative": time_relative,
            "time_context": time_context,
            "topics": topics,
            "keywords": keywords,
            "emotion_valence": emotion_valence,
            "emotion_intensity": emotion_intensity,
        }

    def _is_valid_name(self, name: str) -> bool:
        """验证是否是有效的人名"""
        # 过滤太短或太长的
        if len(name) < 2 or len(name) > 4:
            return False

        # 过滤常见非人名
        invalid_words = {"一起", "吃饭", "去", "见", "聊", "讨论", "开会", "什么", "怎么", "为什么"}
        if name in invalid_words:
            return False

        # 检查是否包含中文字符
        if not re.search(r'[一-鿿]', name):
            return False

        return True

    def _tokenize(self, text: str) -> List[str]:
        """简单分词"""
        terms = []

        # 中文字符（2-gram）
        chinese_chars = re.findall(r'[一-鿿]', text)
        for i in range(len(chinese_chars)):
            terms.append(chinese_chars[i])
            if i + 1 < len(chinese_chars):
                terms.append(chinese_chars[i] + chinese_chars[i + 1])

        # 英文单词
        english_words = re.findall(r'[a-zA-Z]+', text.lower())
        terms.extend(english_words)

        return terms
