"""Public package interface for new-agent-memory."""

from main import HumanLikeMemorySystem
from memory_chunk import MemoryChunk, MemoryLayer
from retrieval import ReconstructionResult, ReviewResult
from core.attention_system import AttentionOS, FocusWorkspace, Goal, ProcedureMemory
from core.agent_system import (
    ActionResult,
    AgentAction,
    CognitiveAgent,
    ExperienceEpisode,
    ExperienceLayer,
    Observation,
)
from core.weight_system import MemoryType

__version__ = "0.1.0"

__all__ = [
    "HumanLikeMemorySystem",
    "ActionResult",
    "AgentAction",
    "AttentionOS",
    "CognitiveAgent",
    "ExperienceEpisode",
    "ExperienceLayer",
    "FocusWorkspace",
    "Goal",
    "MemoryChunk",
    "MemoryLayer",
    "MemoryType",
    "Observation",
    "ProcedureMemory",
    "ReconstructionResult",
    "ReviewResult",
    "__version__",
]
