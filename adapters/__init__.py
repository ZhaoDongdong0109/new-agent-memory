"""Agent adapters for connecting to the Communication Hub."""

from .base_adapter import BaseAdapter
from .claude_adapter import ClaudeAdapter
from .codex_adapter import CodexAdapter
from .hub_sub_agent import main as run_hub_sub_agent

__all__ = [
    "BaseAdapter",
    "ClaudeAdapter",
    "CodexAdapter",
    "run_hub_sub_agent",
]
