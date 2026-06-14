"""Self/world modeling and reflective state for memory-driven agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set
import re
import time
import uuid


def _now() -> float:
    return time.time()


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _shorten(text: str, limit: int = 180) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _tokenize(text: str) -> Set[str]:
    if not text:
        return set()

    lowered = text.lower()
    tokens = set(re.findall(r"[a-zA-Z0-9_]+", lowered))
    chinese_spans = re.findall(r"[\u4e00-\u9fff]{2,}", lowered)
    tokens.update(chinese_spans)
    for span in chinese_spans:
        tokens.update(span[i : i + 2] for i in range(max(0, len(span) - 1)))
    return tokens


def _overlap(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = {item for item in left if item}
    right_set = {item for item in right if item}
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / max(1, len(left_set | right_set))


@dataclass
class DriveState:
    """A simple homeostatic drive.

    `value` is the current satisfaction level. `urgency` is the gap between the
    target and the current value, weighted by importance.
    """

    name: str
    value: float
    target: float
    importance: float
    description: str = ""
    updated_at: float = field(default_factory=_now)

    @property
    def urgency(self) -> float:
        return _clamp((self.target - self.value) * self.importance)

    def adjust(self, delta: float):
        self.value = _clamp(self.value + delta)
        self.updated_at = _now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "target": self.target,
            "importance": self.importance,
            "description": self.description,
            "urgency": self.urgency,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DriveState":
        return cls(
            name=data.get("name", ""),
            value=float(data.get("value", 0.5)),
            target=float(data.get("target", 0.75)),
            importance=float(data.get("importance", 0.5)),
            description=data.get("description", ""),
            updated_at=float(data.get("updated_at", _now())),
        )


@dataclass
class WorldEntity:
    """Something the agent has observed in its world."""

    name: str
    kind: str = "unknown"
    evidence: List[str] = field(default_factory=list)
    affordances: Set[str] = field(default_factory=set)
    confidence: float = 0.5
    updated_at: float = field(default_factory=_now)

    def observe(self, evidence: str, confidence_delta: float = 0.03):
        if evidence and evidence not in self.evidence:
            self.evidence.append(_shorten(evidence, 160))
            self.evidence = self.evidence[-6:]
        self.confidence = _clamp(self.confidence + confidence_delta)
        self.updated_at = _now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "evidence": self.evidence,
            "affordances": sorted(self.affordances),
            "confidence": self.confidence,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorldEntity":
        return cls(
            name=data.get("name", ""),
            kind=data.get("kind", "unknown"),
            evidence=list(data.get("evidence", [])),
            affordances=set(data.get("affordances", [])),
            confidence=float(data.get("confidence", 0.5)),
            updated_at=float(data.get("updated_at", _now())),
        )


@dataclass
class WorldBelief:
    """A confidence-scored belief the agent can revise."""

    subject: str
    predicate: str
    object: str
    confidence: float = 0.5
    source: str = "observation"
    evidence: List[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: f"belief_{uuid.uuid4().hex[:10]}")
    created_at: float = field(default_factory=_now)
    updated_at: float = field(default_factory=_now)

    def key(self) -> str:
        return f"{self.subject}|{self.predicate}|{self.object}"

    def tokens(self) -> Set[str]:
        return _tokenize(" ".join([self.subject, self.predicate, self.object, *self.evidence]))

    def reinforce(self, confidence: float, evidence: Optional[str] = None):
        self.confidence = _clamp((self.confidence * 0.75) + (confidence * 0.25))
        if evidence and evidence not in self.evidence:
            self.evidence.append(_shorten(evidence, 160))
            self.evidence = self.evidence[-6:]
        self.updated_at = _now()

    def to_prompt_line(self) -> str:
        return f"{self.subject} {self.predicate} {self.object} (confidence={self.confidence:.2f})"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "confidence": self.confidence,
            "source": self.source,
            "evidence": self.evidence,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorldBelief":
        return cls(
            id=data.get("id", f"belief_{uuid.uuid4().hex[:10]}"),
            subject=data.get("subject", ""),
            predicate=data.get("predicate", ""),
            object=data.get("object", ""),
            confidence=float(data.get("confidence", 0.5)),
            source=data.get("source", "observation"),
            evidence=list(data.get("evidence", [])),
            created_at=float(data.get("created_at", _now())),
            updated_at=float(data.get("updated_at", _now())),
        )


@dataclass
class ActionExpectation:
    """A prediction made before acting."""

    action_name: str
    expected_success: float
    expected_cost: float = 0.0
    rationale: str = ""
    created_at: float = field(default_factory=_now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_name": self.action_name,
            "expected_success": self.expected_success,
            "expected_cost": self.expected_cost,
            "rationale": self.rationale,
            "created_at": self.created_at,
        }


@dataclass
class ReflectionNote:
    """A compact post-action learning note."""

    episode_id: str
    outcome: str
    insight: str
    uncertainty: str = ""
    next_question: str = ""
    confidence: float = 0.5
    id: str = field(default_factory=lambda: f"reflection_{uuid.uuid4().hex[:10]}")
    created_at: float = field(default_factory=_now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "episode_id": self.episode_id,
            "outcome": self.outcome,
            "insight": self.insight,
            "uncertainty": self.uncertainty,
            "next_question": self.next_question,
            "confidence": self.confidence,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReflectionNote":
        return cls(
            id=data.get("id", f"reflection_{uuid.uuid4().hex[:10]}"),
            episode_id=data.get("episode_id", ""),
            outcome=data.get("outcome", ""),
            insight=data.get("insight", ""),
            uncertainty=data.get("uncertainty", ""),
            next_question=data.get("next_question", ""),
            confidence=float(data.get("confidence", 0.5)),
            created_at=float(data.get("created_at", _now())),
        )


@dataclass
class CognitiveFrame:
    """The inner state that can be injected into an agent prompt."""

    identity: str
    drives: List[Dict[str, Any]]
    beliefs: List[Dict[str, Any]]
    open_questions: List[str]
    action_expectations: List[Dict[str, Any]]
    risk_flags: List[str]
    created_at: float = field(default_factory=_now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "identity": self.identity,
            "drives": self.drives,
            "beliefs": self.beliefs,
            "open_questions": self.open_questions,
            "action_expectations": self.action_expectations,
            "risk_flags": self.risk_flags,
            "created_at": self.created_at,
        }

    def to_prompt_context(self) -> str:
        lines = ["Cognitive State:", f"- Identity: {self.identity}"]
        if self.drives:
            lines.append("- Drives:")
            for drive in self.drives[:5]:
                lines.append(
                    "  - {name}: value={value:.2f}, urgency={urgency:.2f}, target={target:.2f}".format(
                        name=drive.get("name", ""),
                        value=float(drive.get("value", 0.0)),
                        urgency=float(drive.get("urgency", 0.0)),
                        target=float(drive.get("target", 0.0)),
                    )
                )
        if self.beliefs:
            lines.append("- World beliefs:")
            for belief in self.beliefs[:5]:
                lines.append(
                    "  - {subject} {predicate} {object} (confidence={confidence:.2f})".format(
                        subject=belief.get("subject", ""),
                        predicate=belief.get("predicate", ""),
                        object=belief.get("object", ""),
                        confidence=float(belief.get("confidence", 0.0)),
                    )
                )
        if self.action_expectations:
            lines.append("- Action priors:")
            for expectation in self.action_expectations[:5]:
                lines.append(
                    "  - {action}: success~{success:.2f}, cost~{cost:.2f}".format(
                        action=expectation.get("action_name", ""),
                        success=float(expectation.get("expected_success", 0.0)),
                        cost=float(expectation.get("expected_cost", 0.0)),
                    )
                )
        if self.open_questions:
            lines.append("- Open questions:")
            for question in self.open_questions[:4]:
                lines.append(f"  - {question}")
        if self.risk_flags:
            lines.append("- Risk flags:")
            for risk in self.risk_flags[:4]:
                lines.append(f"  - {risk}")
        return "\n".join(lines)


class CognitiveState:
    """A durable self-model, world-model, and reflection loop."""

    def __init__(
        self,
        identity_name: str = "memory-driven agent",
        mission: str = "Turn observations, actions, and feedback into durable adaptive intelligence.",
        capabilities: Optional[List[str]] = None,
        limitations: Optional[List[str]] = None,
        drives: Optional[Dict[str, DriveState]] = None,
        entities: Optional[Dict[str, WorldEntity]] = None,
        beliefs: Optional[List[WorldBelief]] = None,
        reflections: Optional[List[ReflectionNote]] = None,
        open_questions: Optional[List[str]] = None,
        tool_stats: Optional[Dict[str, Dict[str, float]]] = None,
        interaction_count: int = 0,
        last_updated: Optional[float] = None,
    ):
        self.identity_name = identity_name
        self.mission = mission
        self.capabilities = capabilities or [
            "retrieve and consolidate long-term memories",
            "focus attention through active goals",
            "use registered tools as a digital body",
            "learn from action consequences",
        ]
        self.limitations = limitations or [
            "no direct physical senses unless tools or sensors provide observations",
            "reasoning depth depends on the attached planner or language model",
            "world beliefs are probabilistic and must be revised by feedback",
        ]
        self.drives = drives or self._default_drives()
        self.entities = entities or {}
        self.beliefs = beliefs or []
        self.reflections = reflections or []
        self.open_questions = open_questions or []
        self.tool_stats = tool_stats or {}
        self.interaction_count = interaction_count
        self.last_updated = last_updated or _now()

    def observe(self, observation: Any):
        content = getattr(observation, "content", str(observation))
        source = getattr(observation, "source", "environment")
        metadata = getattr(observation, "metadata", {}) or {}

        self.interaction_count += 1
        self._observe_entity(source, "actor", f"Provided observation: {_shorten(content)}")
        self._upsert_belief(
            subject=source,
            predicate="said",
            object=_shorten(content),
            confidence=0.55,
            source="observation",
            evidence=content,
        )

        for person in metadata.get("persons", []) or []:
            self._observe_entity(str(person), "person", f"Mentioned by {source}")
        if metadata.get("location"):
            self._observe_entity(str(metadata["location"]), "place", f"Location metadata from {source}")
        for topic in metadata.get("topics", []) or []:
            self._observe_entity(str(topic), "topic", f"Topic metadata from {source}")

        if self._looks_uncertain(content):
            self._add_open_question(_shorten(content, 140))
            self._adjust_drive("coherence", -0.03)
            self._adjust_drive("curiosity", -0.04)
        else:
            self._adjust_drive("coherence", 0.01)

        self.last_updated = _now()
        self._trim()

    def build_frame(
        self,
        query: str = "",
        workspace: Optional[Any] = None,
        tools: Optional[Any] = None,
    ) -> CognitiveFrame:
        query_tokens = _tokenize(query)
        ranked_beliefs = sorted(
            self.beliefs,
            key=lambda belief: (
                _overlap(query_tokens, belief.tokens()),
                belief.confidence,
                belief.updated_at,
            ),
            reverse=True,
        )
        selected_beliefs = ranked_beliefs[:5]

        drive_items = sorted(
            (drive.to_dict() for drive in self.drives.values()),
            key=lambda item: item["urgency"],
            reverse=True,
        )
        expectations = self._tool_expectations(tools)
        risks = self._risk_flags(workspace, tools)

        return CognitiveFrame(
            identity=self.identity_summary(),
            drives=drive_items,
            beliefs=[belief.to_dict() for belief in selected_beliefs],
            open_questions=self.open_questions[-6:],
            action_expectations=[expectation.to_dict() for expectation in expectations],
            risk_flags=risks,
        )

    def predict_action(self, action: Any, tools: Optional[Any] = None) -> ActionExpectation:
        action_name = getattr(action, "name", str(action))
        tool = getattr(tools, "tools", {}).get(action_name) if tools is not None else None
        stats = self.tool_stats.get(action_name, {})
        attempts = float(stats.get("attempts", 0.0))
        successes = float(stats.get("successes", 0.0))
        expected_success = (successes + 1.0) / (attempts + 2.0)
        expected_cost = float(stats.get("avg_cost", getattr(tool, "cost", 0.0) if tool else 0.0))
        rationale = "tool has prior outcome history" if attempts else "cold-start prior"
        if tool is None and tools is not None:
            expected_success = 0.05
            rationale = "tool is not currently registered"
        return ActionExpectation(
            action_name=action_name,
            expected_success=_clamp(expected_success),
            expected_cost=max(0.0, expected_cost),
            rationale=rationale,
        )

    def reflect_episode(self, episode: Any) -> ReflectionNote:
        action = getattr(episode, "action", None)
        result = getattr(episode, "result", None)
        observation = getattr(episode, "observation", None)
        action_name = getattr(action, "name", "unknown")
        success = bool(getattr(result, "success", False))
        reward = float(getattr(episode, "reward", 0.0))
        output = getattr(result, "output", "")
        cost = float(getattr(result, "cost", 0.0))
        observation_text = getattr(observation, "content", "")

        self._update_tool_stats(action_name, success, reward, cost)
        outcome = "success" if success else "failure"
        insight = (
            f"Action '{action_name}' produced {outcome} for observation '{_shorten(observation_text, 90)}'."
        )
        uncertainty = "" if success else _shorten(str(output), 120)
        next_question = "" if success else f"What precondition was missing before '{action_name}'?"

        confidence = _clamp(0.55 + reward * 0.35 if success else 0.45)
        note = ReflectionNote(
            episode_id=getattr(episode, "id", ""),
            outcome=outcome,
            insight=insight,
            uncertainty=uncertainty,
            next_question=next_question,
            confidence=confidence,
        )
        self.reflections.append(note)

        self._upsert_belief(
            subject=f"tool:{action_name}",
            predicate="outcome",
            object=outcome,
            confidence=confidence,
            source="reflection",
            evidence=insight,
        )

        if success:
            self._adjust_drive("competence", 0.04 + reward * 0.04)
            self._adjust_drive("usefulness", 0.03 + reward * 0.03)
            self._adjust_drive("coherence", 0.02)
        else:
            self._adjust_drive("competence", -0.08)
            self._adjust_drive("coherence", -0.04)
            self._adjust_drive("curiosity", -0.03)
            self._add_open_question(next_question)

        self.last_updated = _now()
        self._trim()
        return note

    def reinforce_from_feedback(self, accepted: bool, corrected_content: Optional[str] = None):
        if accepted:
            self._adjust_drive("usefulness", 0.05)
            self._adjust_drive("coherence", 0.03)
        else:
            self._adjust_drive("usefulness", -0.06)
            self._adjust_drive("coherence", -0.04)
            if corrected_content:
                self._add_open_question(f"Revise answer toward: {_shorten(corrected_content, 120)}")
        self.last_updated = _now()

    def identity_summary(self) -> str:
        mission = self.mission.rstrip(".")
        caps = "; ".join(self.capabilities[:3])
        limits = "; ".join(self.limitations[:2])
        return f"{self.identity_name}. Mission: {mission}. Capabilities: {caps}. Limits: {limits}."

    def get_summary(self) -> Dict[str, Any]:
        return {
            "identity": {
                "name": self.identity_name,
                "mission": self.mission,
                "capabilities": self.capabilities,
                "limitations": self.limitations,
            },
            "drives": {name: drive.to_dict() for name, drive in self.drives.items()},
            "entities_count": len(self.entities),
            "beliefs_count": len(self.beliefs),
            "open_questions": self.open_questions[-8:],
            "tool_stats": self.tool_stats,
            "recent_reflections": [note.to_dict() for note in self.reflections[-5:]],
            "interaction_count": self.interaction_count,
            "last_updated": self.last_updated,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "identity_name": self.identity_name,
            "mission": self.mission,
            "capabilities": self.capabilities,
            "limitations": self.limitations,
            "drives": {name: drive.to_dict() for name, drive in self.drives.items()},
            "entities": {name: entity.to_dict() for name, entity in self.entities.items()},
            "beliefs": [belief.to_dict() for belief in self.beliefs],
            "reflections": [note.to_dict() for note in self.reflections],
            "open_questions": self.open_questions,
            "tool_stats": self.tool_stats,
            "interaction_count": self.interaction_count,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CognitiveState":
        return cls(
            identity_name=data.get("identity_name", "memory-driven agent"),
            mission=data.get(
                "mission",
                "Turn observations, actions, and feedback into durable adaptive intelligence.",
            ),
            capabilities=list(data.get("capabilities", [])) or None,
            limitations=list(data.get("limitations", [])) or None,
            drives={
                name: DriveState.from_dict(item)
                for name, item in dict(data.get("drives", {})).items()
            }
            or None,
            entities={
                name: WorldEntity.from_dict(item)
                for name, item in dict(data.get("entities", {})).items()
            },
            beliefs=[WorldBelief.from_dict(item) for item in data.get("beliefs", [])],
            reflections=[ReflectionNote.from_dict(item) for item in data.get("reflections", [])],
            open_questions=list(data.get("open_questions", [])),
            tool_stats={
                name: {key: float(value) for key, value in dict(stats).items()}
                for name, stats in dict(data.get("tool_stats", {})).items()
            },
            interaction_count=int(data.get("interaction_count", 0)),
            last_updated=float(data.get("last_updated", _now())),
        )

    def _default_drives(self) -> Dict[str, DriveState]:
        return {
            "coherence": DriveState(
                "coherence",
                value=0.72,
                target=0.86,
                importance=0.85,
                description="Keep beliefs, goals, and actions mutually consistent.",
            ),
            "competence": DriveState(
                "competence",
                value=0.68,
                target=0.84,
                importance=0.8,
                description="Increase the chance that chosen actions work.",
            ),
            "curiosity": DriveState(
                "curiosity",
                value=0.62,
                target=0.78,
                importance=0.65,
                description="Resolve useful uncertainty through observation or experiment.",
            ),
            "usefulness": DriveState(
                "usefulness",
                value=0.7,
                target=0.88,
                importance=0.9,
                description="Help the user make real progress.",
            ),
            "safety": DriveState(
                "safety",
                value=0.82,
                target=0.9,
                importance=0.95,
                description="Prefer reversible, auditable, permission-aware action.",
            ),
        }

    def _tool_expectations(self, tools: Optional[Any]) -> List[ActionExpectation]:
        if tools is None or not hasattr(tools, "describe"):
            return []
        expectations: List[ActionExpectation] = []
        for tool in tools.describe():
            fake_action = type("Action", (), {"name": tool.get("name", "")})()
            expectation = self.predict_action(fake_action, tools)
            if tool.get("cost") is not None:
                expectation.expected_cost = max(expectation.expected_cost, float(tool.get("cost", 0.0)))
            expectations.append(expectation)
        expectations.sort(key=lambda item: item.expected_success, reverse=True)
        return expectations[:6]

    def _risk_flags(self, workspace: Optional[Any], tools: Optional[Any]) -> List[str]:
        risks: List[str] = []
        if tools is not None and hasattr(tools, "tools") and not tools.tools:
            risks.append("No tools are registered; the agent can only reason, not act.")
        if workspace is not None and not getattr(workspace, "memories", []):
            risks.append("No relevant long-term memories entered focus.")
        if workspace is not None and not getattr(workspace, "active_goal", None):
            risks.append("No active goal is shaping attention.")
        if self.drives["safety"].urgency > 0.2:
            risks.append("Safety drive has elevated urgency; prefer reversible actions.")
        return risks

    def _update_tool_stats(self, action_name: str, success: bool, reward: float, cost: float):
        stats = self.tool_stats.setdefault(
            action_name,
            {"attempts": 0.0, "successes": 0.0, "failures": 0.0, "avg_reward": 0.0, "avg_cost": 0.0},
        )
        attempts = stats["attempts"]
        stats["attempts"] = attempts + 1.0
        stats["successes"] += 1.0 if success else 0.0
        stats["failures"] += 0.0 if success else 1.0
        stats["avg_reward"] = ((stats["avg_reward"] * attempts) + reward) / stats["attempts"]
        stats["avg_cost"] = ((stats["avg_cost"] * attempts) + cost) / stats["attempts"]

    def _upsert_belief(
        self,
        subject: str,
        predicate: str,
        object: str,
        confidence: float,
        source: str,
        evidence: str,
    ):
        belief = WorldBelief(
            subject=subject,
            predicate=predicate,
            object=object,
            confidence=_clamp(confidence),
            source=source,
            evidence=[_shorten(evidence, 160)] if evidence else [],
        )
        for existing in self.beliefs:
            if existing.key() == belief.key():
                existing.reinforce(confidence, evidence)
                return
        self.beliefs.append(belief)

    def _observe_entity(self, name: str, kind: str, evidence: str):
        if not name:
            return
        entity = self.entities.get(name)
        if entity is None:
            entity = WorldEntity(name=name, kind=kind)
            self.entities[name] = entity
        elif entity.kind == "unknown":
            entity.kind = kind
        entity.observe(evidence)

    def _adjust_drive(self, name: str, delta: float):
        drive = self.drives.get(name)
        if drive:
            drive.adjust(delta)

    def _add_open_question(self, question: str):
        if question and question not in self.open_questions:
            self.open_questions.append(question)
            self.open_questions = self.open_questions[-20:]

    def _looks_uncertain(self, text: str) -> bool:
        lowered = text.lower()
        markers = ["?", "？", "how", "why", "what", "should", "could", "怎么", "为什么", "吗", "如何"]
        return any(marker in lowered for marker in markers)

    def _trim(self):
        self.beliefs = sorted(self.beliefs, key=lambda item: item.updated_at, reverse=True)[:200]
        self.reflections = self.reflections[-100:]
        self.open_questions = self.open_questions[-20:]
        if len(self.entities) > 200:
            ranked = sorted(self.entities.values(), key=lambda item: item.updated_at, reverse=True)[:200]
            self.entities = {entity.name: entity for entity in ranked}
