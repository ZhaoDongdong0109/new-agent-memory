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


class PIIHandler:
    """
    PII 处理器

    检测和脱敏文本中的个人身份信息。
    """

    # PII 模式
    PII_PATTERNS = {
        "phone": r"1[3-9]\d{9}",
        "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "id_card": r"\d{17}[\dXx]",
        "bank_card": r"\d{16,19}",
        "ip_address": r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",
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
                pii_list.append({
                    "type": pii_type,
                    "value": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                })

        # 按位置排序
        pii_list.sort(key=lambda x: x["start"])
        return pii_list

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

        # 从后往前替换（避免位置偏移）
        result = text
        for pii in reversed(pii_list):
            result = result[:pii["start"]] + replacement + result[pii["end"]:]

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
