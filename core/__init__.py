"""Core components for new-agent-memory."""

from core.emotion_engine import EmotionEngine, EmotionResult
from core.persona_layer import BehaviorType, PersonaLayer
from core.weight_system import AdaptiveWeightSystem, MemoryItem, MemoryType, WeightResult

__all__ = [
    "AdaptiveWeightSystem",
    "BehaviorType",
    "EmotionEngine",
    "EmotionResult",
    "MemoryItem",
    "MemoryType",
    "PersonaLayer",
    "WeightResult",
]
