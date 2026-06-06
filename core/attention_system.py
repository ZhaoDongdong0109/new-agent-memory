"""Goal-driven attention orchestration for agent memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set
import math
import re
import time
import uuid

from memory_chunk import MemoryChunk


def _now() -> float:
    return time.time()


def _tokenize(text: str) -> Set[str]:
    """Small mixed Chinese/English tokenizer for explainable scoring."""
    if not text:
        return set()

    lowered = text.lower()
    tokens = set(re.findall(r"[a-zA-Z0-9_]+", lowered))
    chinese_spans = re.findall(r"[\u4e00-\u9fff]{2,}", lowered)
    tokens.update(chinese_spans)
    for span in chinese_spans:
        tokens.update(span[i : i + 2] for i in range(max(0, len(span) - 1)))
    return tokens


def _overlap_score(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = {x for x in left if x}
    right_set = {x for x in right if x}
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / max(1, len(left_set | right_set))


@dataclass
class Goal:
    """A durable objective that should influence attention."""

    objective: str
    id: str = field(default_factory=lambda: f"goal_{uuid.uuid4().hex[:10]}")
    status: str = "active"
    constraints: List[str] = field(default_factory=list)
    open_loops: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    priority: float = 0.7
    created_at: float = field(default_factory=_now)
    updated_at: float = field(default_factory=_now)

    def tokens(self) -> Set[str]:
        text = " ".join([self.objective, *self.constraints, *self.open_loops])
        return _tokenize(text)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "objective": self.objective,
            "status": self.status,
            "constraints": self.constraints,
            "open_loops": self.open_loops,
            "evidence": self.evidence,
            "priority": self.priority,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Goal":
        return cls(
            id=data.get("id", f"goal_{uuid.uuid4().hex[:10]}"),
            objective=data.get("objective", ""),
            status=data.get("status", "active"),
            constraints=list(data.get("constraints", [])),
            open_loops=list(data.get("open_loops", [])),
            evidence=list(data.get("evidence", [])),
            priority=float(data.get("priority", 0.7)),
            created_at=float(data.get("created_at", _now())),
            updated_at=float(data.get("updated_at", _now())),
        )


class GoalStack:
    """Stack of active and historical goals."""

    def __init__(self, goals: Optional[List[Goal]] = None):
        self.goals: List[Goal] = goals or []

    def push(
        self,
        objective: str,
        constraints: Optional[Sequence[str]] = None,
        open_loops: Optional[Sequence[str]] = None,
        priority: float = 0.7,
    ) -> Goal:
        goal = Goal(
            objective=objective,
            constraints=list(constraints or []),
            open_loops=list(open_loops or []),
            priority=max(0.0, min(1.0, priority)),
        )
        self.goals.append(goal)
        return goal

    def active(self) -> Optional[Goal]:
        for goal in reversed(self.goals):
            if goal.status == "active":
                return goal
        return None

    def update(
        self,
        goal_id: str,
        *,
        status: Optional[str] = None,
        evidence: Optional[Sequence[str]] = None,
        open_loops: Optional[Sequence[str]] = None,
    ) -> Optional[Goal]:
        goal = self.get(goal_id)
        if not goal:
            return None
        if status is not None:
            goal.status = status
        if evidence is not None:
            goal.evidence = list(evidence)
        if open_loops is not None:
            goal.open_loops = list(open_loops)
        goal.updated_at = _now()
        return goal

    def get(self, goal_id: str) -> Optional[Goal]:
        for goal in self.goals:
            if goal.id == goal_id:
                return goal
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {"goals": [goal.to_dict() for goal in self.goals]}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GoalStack":
        return cls([Goal.from_dict(item) for item in data.get("goals", [])])


@dataclass
class ProcedureMemory:
    """A reusable way of doing work."""

    title: str
    steps: List[str]
    triggers: Set[str] = field(default_factory=set)
    id: str = field(default_factory=lambda: f"proc_{uuid.uuid4().hex[:10]}")
    importance: float = 0.6
    confidence: float = 0.6
    use_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    created_at: float = field(default_factory=_now)
    last_used: float = field(default_factory=_now)

    def tokens(self) -> Set[str]:
        text = " ".join([self.title, *self.steps, *sorted(self.triggers)])
        return _tokenize(text) | set(self.triggers)

    def record_use(self, success: Optional[bool] = None):
        self.use_count += 1
        self.last_used = _now()
        if success is True:
            self.success_count += 1
            self.confidence = min(1.0, self.confidence + 0.05)
        elif success is False:
            self.failure_count += 1
            self.confidence = max(0.0, self.confidence - 0.08)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "steps": self.steps,
            "triggers": sorted(self.triggers),
            "importance": self.importance,
            "confidence": self.confidence,
            "use_count": self.use_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "created_at": self.created_at,
            "last_used": self.last_used,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcedureMemory":
        return cls(
            id=data.get("id", f"proc_{uuid.uuid4().hex[:10]}"),
            title=data.get("title", ""),
            steps=list(data.get("steps", [])),
            triggers=set(data.get("triggers", [])),
            importance=float(data.get("importance", 0.6)),
            confidence=float(data.get("confidence", 0.6)),
            use_count=int(data.get("use_count", 0)),
            success_count=int(data.get("success_count", 0)),
            failure_count=int(data.get("failure_count", 0)),
            created_at=float(data.get("created_at", _now())),
            last_used=float(data.get("last_used", _now())),
        )


@dataclass
class AttentionScore:
    """Decomposed score for auditability."""

    item_id: str
    item_type: str
    final_score: float
    goal_relevance: float = 0.0
    query_relevance: float = 0.0
    user_importance: float = 0.0
    recency: float = 0.0
    frequency: float = 0.0
    emotional_salience: float = 0.0
    uncertainty_need: float = 0.0
    novelty: float = 0.0
    distraction_penalty: float = 0.0
    staleness_penalty: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class FocusItem:
    """One selected item in the focus workspace."""

    id: str
    item_type: str
    content: str
    score: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class FocusWorkspace:
    """Bounded attention context for a turn."""

    query: str
    active_goal: Optional[Goal]
    memories: List[FocusItem]
    procedures: List[FocusItem]
    audit: List[Dict[str, Any]]
    created_at: float = field(default_factory=_now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "active_goal": self.active_goal.to_dict() if self.active_goal else None,
            "memories": [item.to_dict() for item in self.memories],
            "procedures": [item.to_dict() for item in self.procedures],
            "audit": self.audit,
            "created_at": self.created_at,
        }

    def to_prompt_context(self) -> str:
        lines: List[str] = []
        if self.active_goal:
            lines.append(f"Goal: {self.active_goal.objective}")
            if self.active_goal.constraints:
                lines.append("Constraints: " + "; ".join(self.active_goal.constraints))
        if self.procedures:
            lines.append("Procedures:")
            for item in self.procedures:
                lines.append(f"- {item.content}")
        if self.memories:
            lines.append("Memories:")
            for item in self.memories:
                lines.append(f"- {item.content}")
        return "\n".join(lines)


class AttentionScorer:
    """Explainable scorer for memories and procedures."""

    def __init__(
        self,
        memory_threshold: float = 0.22,
        procedure_threshold: float = 0.25,
        stale_after_seconds: float = 30 * 24 * 3600,
    ):
        self.memory_threshold = memory_threshold
        self.procedure_threshold = procedure_threshold
        self.stale_after_seconds = stale_after_seconds

    def score_memory(self, chunk: MemoryChunk, query: str, active_goal: Optional[Goal]) -> AttentionScore:
        query_tokens = _tokenize(query)
        goal_tokens = active_goal.tokens() if active_goal else set()
        chunk_tokens = self._chunk_tokens(chunk)

        query_relevance = _overlap_score(query_tokens, chunk_tokens)
        goal_relevance = _overlap_score(goal_tokens, chunk_tokens)
        user_importance = max(0.0, min(1.0, chunk.importance))
        recency = self._recency_score(chunk.last_accessed)
        frequency = min(1.0, math.log1p(chunk.access_count) / math.log(11))
        emotional_salience = max(0.0, min(1.0, abs(chunk.emotion_valence) * max(0.2, chunk.emotion_intensity)))
        uncertainty_need = 0.25 if chunk.review_status in {"pending", "questionable"} else 0.0
        novelty = 1.0 / (1.0 + chunk.access_count)

        distraction_penalty = 0.0
        if query_tokens and query_relevance + goal_relevance < 0.08:
            distraction_penalty = 0.18

        staleness_penalty = self._staleness_penalty(chunk.last_accessed)

        final = (
            goal_relevance * 0.30
            + query_relevance * 0.30
            + user_importance * 0.16
            + recency * 0.08
            + frequency * 0.06
            + emotional_salience * 0.06
            + uncertainty_need * 0.03
            + novelty * 0.01
            - distraction_penalty
            - staleness_penalty
        )
        final = max(0.0, min(1.0, final))

        return AttentionScore(
            item_id=chunk.id,
            item_type="memory",
            final_score=final,
            goal_relevance=goal_relevance,
            query_relevance=query_relevance,
            user_importance=user_importance,
            recency=recency,
            frequency=frequency,
            emotional_salience=emotional_salience,
            uncertainty_need=uncertainty_need,
            novelty=novelty,
            distraction_penalty=distraction_penalty,
            staleness_penalty=staleness_penalty,
        )

    def score_procedure(
        self,
        procedure: ProcedureMemory,
        query: str,
        active_goal: Optional[Goal],
    ) -> AttentionScore:
        query_tokens = _tokenize(query)
        goal_tokens = active_goal.tokens() if active_goal else set()
        proc_tokens = procedure.tokens()

        query_relevance = _overlap_score(query_tokens, proc_tokens)
        goal_relevance = _overlap_score(goal_tokens, proc_tokens)
        recency = self._recency_score(procedure.last_used)
        frequency = min(1.0, math.log1p(procedure.use_count) / math.log(11))
        novelty = 1.0 / (1.0 + procedure.use_count)

        final = (
            goal_relevance * 0.32
            + query_relevance * 0.32
            + procedure.importance * 0.14
            + procedure.confidence * 0.12
            + recency * 0.04
            + frequency * 0.03
            + novelty * 0.03
        )
        final = max(0.0, min(1.0, final))

        return AttentionScore(
            item_id=procedure.id,
            item_type="procedure",
            final_score=final,
            goal_relevance=goal_relevance,
            query_relevance=query_relevance,
            user_importance=procedure.importance,
            recency=recency,
            frequency=frequency,
            novelty=novelty,
        )

    def _chunk_tokens(self, chunk: MemoryChunk) -> Set[str]:
        text_parts = [
            chunk.content,
            chunk.summary,
            chunk.location or "",
            chunk.location_detail or "",
            chunk.time_absolute or "",
            chunk.time_relative or "",
            chunk.time_context or "",
            " ".join(chunk.persons),
            " ".join(chunk.topics),
            " ".join(chunk.keywords),
            " ".join(chunk.emotion_tags),
        ]
        return _tokenize(" ".join(text_parts)) | set(chunk.topics) | set(chunk.keywords) | set(chunk.persons)

    def _recency_score(self, timestamp: float) -> float:
        age = max(0.0, _now() - timestamp)
        return math.exp(-age / (7 * 24 * 3600))

    def _staleness_penalty(self, timestamp: float) -> float:
        age = max(0.0, _now() - timestamp)
        if age <= self.stale_after_seconds:
            return 0.0
        return min(0.18, (age - self.stale_after_seconds) / (180 * 24 * 3600))


class AttentionOS:
    """Coordinates goals, procedures, gating, and focus audits."""

    def __init__(
        self,
        goal_stack: Optional[GoalStack] = None,
        procedures: Optional[List[ProcedureMemory]] = None,
        scorer: Optional[AttentionScorer] = None,
    ):
        self.goal_stack = goal_stack or GoalStack()
        self.procedures = procedures or []
        self.scorer = scorer or AttentionScorer()
        self.workspace_history: List[Dict[str, Any]] = []

    def start_goal(
        self,
        objective: str,
        constraints: Optional[Sequence[str]] = None,
        open_loops: Optional[Sequence[str]] = None,
        priority: float = 0.7,
    ) -> Goal:
        return self.goal_stack.push(objective, constraints, open_loops, priority)

    def update_goal(
        self,
        goal_id: str,
        *,
        status: Optional[str] = None,
        evidence: Optional[Sequence[str]] = None,
        open_loops: Optional[Sequence[str]] = None,
    ) -> Optional[Goal]:
        return self.goal_stack.update(goal_id, status=status, evidence=evidence, open_loops=open_loops)

    def add_procedure(
        self,
        title: str,
        steps: Sequence[str],
        triggers: Optional[Sequence[str]] = None,
        importance: float = 0.6,
        confidence: float = 0.6,
    ) -> ProcedureMemory:
        procedure = ProcedureMemory(
            title=title,
            steps=list(steps),
            triggers=set(triggers or []),
            importance=max(0.0, min(1.0, importance)),
            confidence=max(0.0, min(1.0, confidence)),
        )
        self.procedures.append(procedure)
        return procedure

    def build_focus(
        self,
        query: str,
        memories: Sequence[MemoryChunk],
        memory_limit: int = 5,
        procedure_limit: int = 3,
    ) -> FocusWorkspace:
        active_goal = self.goal_stack.active()

        memory_scores = [
            (chunk, self.scorer.score_memory(chunk, query, active_goal))
            for chunk in memories
        ]
        memory_scores.sort(key=lambda pair: pair[1].final_score, reverse=True)

        procedure_scores = [
            (procedure, self.scorer.score_procedure(procedure, query, active_goal))
            for procedure in self.procedures
        ]
        procedure_scores.sort(key=lambda pair: pair[1].final_score, reverse=True)

        selected_memories = [
            FocusItem(
                id=chunk.id,
                item_type="memory",
                content=chunk.content,
                score=score.final_score,
                reason=self._reason(score),
            )
            for chunk, score in memory_scores
            if score.final_score >= self.scorer.memory_threshold
        ][:memory_limit]

        selected_procedures = [
            FocusItem(
                id=procedure.id,
                item_type="procedure",
                content=f"{procedure.title}: " + " -> ".join(procedure.steps),
                score=score.final_score,
                reason=self._reason(score),
            )
            for procedure, score in procedure_scores
            if score.final_score >= self.scorer.procedure_threshold
        ][:procedure_limit]

        audit = [score.to_dict() for _, score in memory_scores[:10]]
        audit.extend(score.to_dict() for _, score in procedure_scores[:10])

        workspace = FocusWorkspace(
            query=query,
            active_goal=active_goal,
            memories=selected_memories,
            procedures=selected_procedures,
            audit=audit,
        )
        self.workspace_history.append(workspace.to_dict())
        if len(self.workspace_history) > 50:
            self.workspace_history = self.workspace_history[-50:]
        return workspace

    def record_procedure_use(self, procedure_id: str, success: Optional[bool] = None) -> bool:
        for procedure in self.procedures:
            if procedure.id == procedure_id:
                procedure.record_use(success)
                return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal_stack": self.goal_stack.to_dict(),
            "procedures": [procedure.to_dict() for procedure in self.procedures],
            "workspace_history": self.workspace_history[-50:],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AttentionOS":
        attention = cls(
            goal_stack=GoalStack.from_dict(data.get("goal_stack", {})),
            procedures=[ProcedureMemory.from_dict(item) for item in data.get("procedures", [])],
        )
        attention.workspace_history = list(data.get("workspace_history", []))[-50:]
        return attention

    def _reason(self, score: AttentionScore) -> str:
        parts = []
        if score.goal_relevance:
            parts.append(f"goal={score.goal_relevance:.2f}")
        if score.query_relevance:
            parts.append(f"query={score.query_relevance:.2f}")
        if score.user_importance:
            parts.append(f"importance={score.user_importance:.2f}")
        if score.distraction_penalty:
            parts.append(f"distraction=-{score.distraction_penalty:.2f}")
        return ", ".join(parts) or "baseline attention"
