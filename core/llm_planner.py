"""LLM-backed planner for CognitiveAgent.

This module is provider-agnostic. Pass any callable that takes a prompt string
and returns text. The returned text should contain a JSON action.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
import json
import re

from core.agent_system import AgentAction, Observation, ToolRegistry
from core.attention_system import FocusWorkspace


LLMCallable = Callable[[str], str]


@dataclass
class LLMPlannerConfig:
    """Configuration for the LLM planner."""

    system_instructions: str = (
        "You are the decision core of a memory-driven agent. "
        "Choose exactly one tool/action for the next step."
    )
    default_action: str = "respond"
    strict_tools: bool = True
    max_prompt_chars: int = 12000


class LLMPlanner:
    """
    Planner adapter that asks an LLM to return an AgentAction JSON object.

    Expected model output:

    {
      "name": "respond",
      "arguments": {"message": "..."},
      "rationale": "..."
    }
    """

    def __init__(
        self,
        llm: LLMCallable,
        config: Optional[LLMPlannerConfig] = None,
    ):
        self.llm = llm
        self.config = config or LLMPlannerConfig()
        self.last_prompt: str = ""
        self.last_output: str = ""

    def __call__(self, observation: Observation, workspace: FocusWorkspace, tools: ToolRegistry) -> AgentAction:
        prompt = self.build_prompt(observation, workspace, tools)
        self.last_prompt = prompt
        output = self.llm(prompt)
        self.last_output = output
        return self.parse_action(output, tools)

    def build_prompt(self, observation: Observation, workspace: FocusWorkspace, tools: ToolRegistry) -> str:
        tool_descriptions = json.dumps(tools.describe(), ensure_ascii=False, indent=2)
        focus_context = workspace.to_prompt_context()
        audit = json.dumps(workspace.audit[:8], ensure_ascii=False, indent=2)

        prompt = f"""
{self.config.system_instructions}

Current focus workspace:
{focus_context or "(empty)"}

Observation:
source={observation.source}
content={observation.content}
metadata={json.dumps(observation.metadata, ensure_ascii=False)}

Available tools:
{tool_descriptions}

Attention audit sample:
{audit}

Return ONLY one JSON object with this shape:
{{
  "name": "tool_name",
  "arguments": {{}},
  "rationale": "short reason"
}}

Rules:
- `name` must be one of the available tool names.
- Use `respond` when no external tool is needed.
- Keep arguments small and explicit.
- Do not include markdown outside the JSON.
""".strip()

        if len(prompt) <= self.config.max_prompt_chars:
            return prompt
        return prompt[: self.config.max_prompt_chars] + "\n...[truncated]"

    def parse_action(self, output: str, tools: ToolRegistry) -> AgentAction:
        try:
            data = json.loads(self._extract_json(output))
        except Exception:
            return self._fallback_action(f"Could not parse planner JSON: {output[:200]}")

        name = data.get("name") or data.get("tool") or self.config.default_action
        arguments = data.get("arguments") or data.get("args") or {}
        rationale = data.get("rationale") or data.get("reason") or ""

        if not isinstance(arguments, dict):
            arguments = {"input": arguments}

        if self.config.strict_tools and name not in tools.tools:
            return self._fallback_action(f"Planner selected unavailable tool '{name}'.")

        return AgentAction(name=name, arguments=arguments, rationale=rationale)

    def _fallback_action(self, reason: str) -> AgentAction:
        return AgentAction(
            name=self.config.default_action,
            arguments={"message": reason},
            rationale="LLMPlanner fallback",
        )

    def _extract_json(self, text: str) -> str:
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
        if fenced:
            return fenced.group(1)

        start = text.find("{")
        if start == -1:
            raise ValueError("No JSON object found")

        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(text)):
            char = text[index]
            if escape:
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]

        raise ValueError("Unclosed JSON object")

    @classmethod
    def from_openai_responses(
        cls,
        client: Any,
        model: str,
        config: Optional[LLMPlannerConfig] = None,
        **response_kwargs: Any,
    ) -> "LLMPlanner":
        """
        Build a planner from an OpenAI Responses-style client without importing openai.

        Example:
            client = OpenAI()
            planner = LLMPlanner.from_openai_responses(client, "gpt-4.1-mini")
        """

        def llm(prompt: str) -> str:
            response = client.responses.create(
                model=model,
                input=prompt,
                **response_kwargs,
            )
            output_text = getattr(response, "output_text", None)
            if output_text is not None:
                return output_text
            return str(response)

        return cls(llm=llm, config=config)
