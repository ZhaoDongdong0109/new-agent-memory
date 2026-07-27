"""
会话工作记忆 - 多轮对话缓冲

大脑升级第一刀：此前 CLI 聊天每轮独立调 run_turn，不维护任何
消息历史——代词指代、"上面那个"必断，跨轮连续性全靠捞上一轮
固化的 STORY 记忆。这不是类人记忆，是每轮失忆。

ConversationBuffer 是会话级工作记忆（对应人类的语音回路/情景
缓冲）：滚动保留最近 N 轮原文；溢出的旧轮不丢弃，而是交还给
调用方归档进长期记忆（INTERACTION 类型，自然衰减）——工作记忆
有限、长期记忆兜底，与认知架构其余部分同一设计语言。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ConversationTurn:
    role: str          # "user" / "assistant"
    content: str
    timestamp: float = field(default_factory=lambda: time.time())

    def to_dict(self) -> Dict:
        return {"role": self.role, "content": self.content, "timestamp": self.timestamp}


class ConversationBuffer:
    """滚动的多轮对话缓冲（会话级工作记忆）"""

    def __init__(self, max_turns: int = 16, max_render_chars: int = 2000):
        self.max_turns = max_turns
        self.max_render_chars = max_render_chars
        self.turns: List[ConversationTurn] = []

    def add(self, role: str, content: str) -> None:
        if not content:
            return
        self.turns.append(ConversationTurn(role=role, content=content))

    def pop_overflow(self) -> List[ConversationTurn]:
        """取出超出容量的最旧轮次（调用方负责归档进长期记忆）"""
        if len(self.turns) <= self.max_turns:
            return []
        overflow = self.turns[: len(self.turns) - self.max_turns]
        self.turns = self.turns[len(self.turns) - self.max_turns:]
        return overflow

    def to_messages(self) -> List[Dict[str, str]]:
        """OpenAI 消息数组形态（供原生多消息客户端使用）"""
        return [{"role": t.role, "content": t.content} for t in self.turns]

    def render_context(self) -> str:
        """渲染为提示文本块（近轮优先，字符预算内从新到旧回填）"""
        if not self.turns:
            return ""
        lines: List[str] = []
        used = 0
        for turn in reversed(self.turns):
            speaker = "用户" if turn.role == "user" else "助手"
            line = f"{speaker}: {turn.content}"
            if used + len(line) > self.max_render_chars and lines:
                break
            lines.append(line)
            used += len(line)
        return "\n".join(reversed(lines))

    def to_dict(self) -> Dict:
        return {"max_turns": self.max_turns, "turns": [t.to_dict() for t in self.turns]}
