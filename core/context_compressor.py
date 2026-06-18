"""
ContextCompressor - 短期上下文压缩器

职责：
1. 压缩工具日志
2. 压缩 scratchpad
3. 去重重复消息
4. 生成结构化摘要
"""

from typing import Dict, List, Optional, Tuple


class ContextCompressor:
    """
    短期上下文压缩器

    将冗长的上下文压缩为简洁的摘要，
    降低 token 成本，提高长任务成功率。
    """

    def __init__(self, max_tokens: int = 2000):
        """
        Args:
            max_tokens: 压缩后的最大 token 数
        """
        self.max_tokens = max_tokens

    def compress(self, messages: List[Dict]) -> str:
        """
        压缩消息列表为摘要

        Args:
            messages: 消息列表，每个消息是 dict

        Returns:
            压缩后的摘要文本
        """
        if not messages:
            return ""

        # 1. 去重
        unique = self._deduplicate(messages)

        # 2. 分类
        tool_logs, chat_messages, system_messages = self._categorize(unique)

        # 3. 压缩各类消息
        compressed_parts = []

        if tool_logs:
            compressed_parts.append(self._compress_tool_logs(tool_logs))

        if chat_messages:
            compressed_parts.append(self._compress_chat(chat_messages))

        if system_messages:
            compressed_parts.append(self._compress_system(system_messages))

        # 4. 合并并截断
        result = "\n\n".join(compressed_parts)
        return self._truncate(result, self.max_tokens)

    def compress_episodes(self, episodes: List[Dict]) -> str:
        """
        压缩 episode 列表为摘要

        Args:
            episodes: episode 列表

        Returns:
            压缩后的摘要文本
        """
        if not episodes:
            return ""

        parts = []

        for i, episode in enumerate(episodes, 1):
            goal = episode.get("goal", "")
            action = episode.get("action", "")
            success = episode.get("success", False)
            lesson = episode.get("lesson", "")

            part = f"[{i}] "
            if goal:
                part += f"目标: {goal} | "
            part += f"动作: {action} | 结果: {'成功' if success else '失败'}"
            if lesson:
                part += f" | 教训: {lesson}"

            parts.append(part)

        result = "\n".join(parts)
        return self._truncate(result, self.max_tokens)

    def _deduplicate(self, messages: List[Dict]) -> List[Dict]:
        """去重重复消息"""
        seen = set()
        unique = []

        for msg in messages:
            content = msg.get("content", "")
            if not content:
                continue

            # 使用内容哈希去重
            content_hash = hash(content)
            if content_hash not in seen:
                seen.add(content_hash)
                unique.append(msg)

        return unique

    def _categorize(
        self, messages: List[Dict]
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """分类消息"""
        tool_logs = []
        chat_messages = []
        system_messages = []

        for msg in messages:
            msg_type = msg.get("message_type", "text")

            if msg_type in ("tool_result", "tool_call"):
                tool_logs.append(msg)
            elif msg_type == "system":
                system_messages.append(msg)
            else:
                chat_messages.append(msg)

        return tool_logs, chat_messages, system_messages

    def _compress_tool_logs(self, logs: List[Dict]) -> str:
        """压缩工具日志"""
        if not logs:
            return ""

        summaries = []

        for log in logs:
            msg_type = log.get("message_type", "")
            tool_name = log.get("metadata", {}).get("tool_name", "unknown")
            content = log.get("content", "")

            if msg_type == "tool_call":
                # 工具调用：只记录工具名和参数摘要
                args = log.get("metadata", {}).get("arguments", {})
                args_summary = str(args)[:50] if args else ""
                summaries.append(f"[调用] {tool_name}({args_summary})")
            elif msg_type == "tool_result":
                # 工具结果：只记录成功/失败和输出摘要
                success = log.get("metadata", {}).get("success", True)
                output = content[:100] if content else "无输出"
                status = "✓" if success else "✗"
                summaries.append(f"[结果] {tool_name} {status}: {output}")

        if not summaries:
            return ""

        return "工具日志:\n" + "\n".join(summaries[-10:])  # 只保留最近 10 条

    def _compress_chat(self, messages: List[Dict]) -> str:
        """压缩聊天消息"""
        if not messages:
            return ""

        # 合并连续的同角色消息
        merged = []
        current_role = None
        current_content = []

        for msg in messages:
            role = msg.get("sender_name", msg.get("role", "unknown"))
            content = msg.get("content", "")

            if role == current_role:
                current_content.append(content)
            else:
                if current_role and current_content:
                    merged.append(f"{current_role}: {' '.join(current_content)}")
                current_role = role
                current_content = [content]

        # 处理最后一批
        if current_role and current_content:
            merged.append(f"{current_role}: {' '.join(current_content)}")

        if not merged:
            return ""

        return "对话:\n" + "\n".join(merged[-15:])  # 只保留最近 15 条

    def _compress_system(self, messages: List[Dict]) -> str:
        """压缩系统消息"""
        if not messages:
            return ""

        # 系统消息通常较短，直接合并
        contents = []
        for msg in messages:
            content = msg.get("content", "")
            if content:
                contents.append(content)

        if not contents:
            return ""

        return "系统:\n" + "\n".join(contents[-5:])  # 只保留最近 5 条

    def _truncate(self, text: str, max_tokens: int) -> str:
        """截断到指定 token 数"""
        if not text:
            return ""

        # 粗略估计：1 个中文字符 ≈ 2 tokens，1 个英文单词 ≈ 1 token
        max_chars = max_tokens * 2

        if len(text) <= max_chars:
            return text

        # 截断并添加省略号
        return text[:max_chars] + "\n... (已压缩)"

    def summarize_context(self, context: Dict) -> str:
        """
        总结上下文

        Args:
            context: 上下文字典

        Returns:
            总结文本
        """
        parts = []

        # 目标
        if context.get("goal"):
            parts.append(f"目标: {context['goal']}")

        # 当前状态
        if context.get("status"):
            parts.append(f"状态: {context['status']}")

        # 关键信息
        if context.get("key_info"):
            parts.append(f"关键信息: {context['key_info']}")

        # 待办事项
        if context.get("todo"):
            parts.append(f"待办: {context['todo']}")

        return " | ".join(parts) if parts else ""
