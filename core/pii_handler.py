"""
PII 处理器

检测、脱敏、匿名化个人身份信息 (PII)。

支持的 PII 类型：
- 手机号
- 邮箱
- 身份证号
- 人名（需要 NER）
"""

import re
from typing import Dict, List, Optional, Set, Tuple

# IPv4 单个八位组：0-255
_IPV4_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"


class PIIHandler:
    """
    PII 处理器

    检测和脱敏文本中的个人身份信息。
    """

    # PII 模式
    # 数字类模式使用 (?<!\d) / (?!\d) 边界保护，避免命中更长数字串的子串
    # （例如订单号、编号中嵌入的"手机号"片段）。
    PII_PATTERNS = {
        "phone": r"(?<!\d)1[3-9]\d{9}(?!\d)",
        "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "id_card": r"(?<!\d)\d{17}[\dXx](?!\d)",
        # 银行卡号还需通过 Luhn 校验（见 detect 中的过滤）
        "bank_card": r"(?<!\d)\d{16,19}(?!\d)",
        # IP 要求每个八位组在 0-255 范围内，且不能是更长点分数字串
        # （如版本号 10.2.3.4.5）的一部分
        "ip_address": (
            r"(?<!\d)(?<!\d\.)"
            + r"(?:" + _IPV4_OCTET + r"\.){3}" + _IPV4_OCTET
            + r"(?!\d)(?!\.\d)"
        ),
    }

    def __init__(self, custom_patterns: Dict[str, str] = None):
        """
        Args:
            custom_patterns: 自定义 PII 模式
        """
        self.patterns = dict(self.PII_PATTERNS)
        if custom_patterns:
            self.patterns.update(custom_patterns)

    def detect(self, text: str) -> List[Dict]:
        """
        检测文本中的 PII

        Args:
            text: 输入文本

        Returns:
            PII 列表，每个元素是 {"type": str, "value": str, "start": int, "end": int}
        """
        pii_list = []

        for pii_type, pattern in self.patterns.items():
            for match in re.finditer(pattern, text):
                value = match.group()
                # 银行卡号必须通过 Luhn 校验，避免把订单号等长数字串当作 PII
                if pii_type == "bank_card" and not self._luhn_valid(value):
                    continue
                pii_list.append({
                    "type": pii_type,
                    "value": value,
                    "start": match.start(),
                    "end": match.end(),
                })

        # 按位置排序（同一起点时较长的匹配在前，便于处理重叠）
        pii_list.sort(key=lambda x: (x["start"], -(x["end"] - x["start"])))
        return pii_list

    @staticmethod
    def _luhn_valid(number: str) -> bool:
        """
        Luhn 校验（银行卡号合法性检查）

        Args:
            number: 纯数字字符串

        Returns:
            是否通过校验
        """
        total = 0
        for i, ch in enumerate(reversed(number)):
            digit = int(ch)
            if i % 2 == 1:
                digit *= 2
                if digit > 9:
                    digit -= 9
            total += digit
        return total % 10 == 0

    @staticmethod
    def _merge_spans(pii_list: List[Dict]) -> List[Tuple[int, int]]:
        """
        合并重叠或相邻的 PII 区间

        Args:
            pii_list: 已按起始位置排序的 PII 列表

        Returns:
            合并后的 (start, end) 区间列表，互不重叠
        """
        merged: List[List[int]] = []
        for pii in pii_list:
            if merged and pii["start"] <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], pii["end"])
            else:
                merged.append([pii["start"], pii["end"]])
        return [(start, end) for start, end in merged]

    @staticmethod
    def _select_non_overlapping(pii_list: List[Dict]) -> List[Dict]:
        """
        从已排序的 PII 列表中挑选互不重叠的匹配

        重叠时保留起始位置更靠前（同一起点时更长）的匹配。

        Args:
            pii_list: 已按 (start, -长度) 排序的 PII 列表

        Returns:
            互不重叠的 PII 列表
        """
        selected = []
        last_end = -1
        for pii in pii_list:
            if pii["start"] >= last_end:
                selected.append(pii)
                last_end = pii["end"]
        return selected

    def redact(
        self,
        text: str,
        replacement: str = "[REDACTED]",
        pii_types: Optional[List[str]] = None,
    ) -> str:
        """
        脱敏文本中的 PII

        Args:
            text: 输入文本
            replacement: 替换文本
            pii_types: 要脱敏的 PII 类型（None 表示全部）

        Returns:
            脱敏后的文本
        """
        pii_list = self.detect(text)

        # 过滤类型
        if pii_types:
            pii_list = [pii for pii in pii_list if pii["type"] in pii_types]

        # 先合并重叠/相邻区间，再从后往前替换
        # （区间基于原文计算，重叠区间若逐个替换会用到失效的偏移，损坏相邻文本）
        spans = self._merge_spans(pii_list)
        result = text
        for start, end in reversed(spans):
            result = result[:start] + replacement + result[end:]

        return result

    def has_pii(self, text: str) -> bool:
        """
        检查文本是否包含 PII

        Args:
            text: 输入文本

        Returns:
            是否包含 PII
        """
        return len(self.detect(text)) > 0

    def get_pii_types(self, text: str) -> Set[str]:
        """
        获取文本中包含的 PII 类型

        Args:
            text: 输入文本

        Returns:
            PII 类型集合
        """
        return {pii["type"] for pii in self.detect(text)}

    def anonymize_text(self, text: str) -> str:
        """
        匿名化文本

        对不同类型的 PII 使用不同的匿名化策略。

        Args:
            text: 输入文本

        Returns:
            匿名化后的文本
        """
        pii_list = self.detect(text)

        # 匿名化需要按类型区分策略，无法合并区间；
        # 先剔除重叠匹配，再基于原文偏移从后往前替换
        pii_list = self._select_non_overlapping(pii_list)

        result = text
        for pii in reversed(pii_list):
            pii_type = pii["type"]
            value = pii["value"]

            if pii_type == "phone":
                # 手机号：保留前3后4
                anonymized = value[:3] + "****" + value[-4:]
            elif pii_type == "email":
                # 邮箱：保留首字母和域名
                local, domain = value.split("@")
                anonymized = local[0] + "***@" + domain
            elif pii_type == "id_card":
                # 身份证：保留前4后4
                anonymized = value[:4] + "**********" + value[-4:]
            else:
                # 其他：直接替换
                anonymized = f"[{pii_type.upper()}]"

            result = result[:pii["start"]] + anonymized + result[pii["end"]:]

        return result
