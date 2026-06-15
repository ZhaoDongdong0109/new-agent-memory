"""LLM-backed planner for CognitiveAgent.

This module is provider-agnostic. Pass any callable that takes a prompt string
and returns text. The returned text should contain a JSON action.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Mapping, Optional
import json
import os
import re
import urllib.error
import urllib.request

from core.agent_system import AgentAction, Observation, ToolRegistry
from core.attention_system import FocusWorkspace


LLMCallable = Callable[[str], str]


def _parse_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_float(value: Optional[str], default: float) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _parse_optional_int(value: Optional[str]) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def load_env_file(path: Optional[str] = ".env") -> Dict[str, str]:
    """Load simple KEY=VALUE pairs without requiring python-dotenv."""
    if not path or not os.path.exists(path):
        return {}

    values: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'\"")
            if key:
                values[key] = value
    return values


@dataclass
class OpenAICompatibleConfig:
    """Configuration for OpenAI-compatible Chat Completions endpoints."""

    model: str = "gpt-4.1-mini"
    api_key: Optional[str] = None
    base_url: str = "https://api.openai.com/v1"
    timeout: float = 60.0
    temperature: float = 0.1
    max_tokens: Optional[int] = 800
    stream: bool = False
    use_env_proxy: bool = False
    extra_headers: Dict[str, str] = field(default_factory=dict)
    extra_body: Dict[str, Any] = field(default_factory=dict)

    @property
    def chat_completions_url(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"

    @classmethod
    def from_env(
        cls,
        env_file: Optional[str] = ".env",
        environ: Optional[Mapping[str, str]] = None,
        **overrides: Any,
    ) -> "OpenAICompatibleConfig":
        """Build config from .env + process env + explicit keyword overrides."""
        merged: Dict[str, str] = {}
        merged.update(load_env_file(env_file))
        merged.update(dict(os.environ if environ is None else environ))

        config = cls(
            model=merged.get("OPENAI_COMPATIBLE_MODEL")
            or merged.get("OPENAI_MODEL")
            or cls.model,
            api_key=merged.get("OPENAI_COMPATIBLE_API_KEY")
            or merged.get("OPENAI_API_KEY"),
            base_url=merged.get("OPENAI_COMPATIBLE_BASE_URL")
            or merged.get("OPENAI_BASE_URL")
            or merged.get("OPENAI_API_BASE")
            or cls.base_url,
            timeout=_parse_float(merged.get("OPENAI_COMPATIBLE_TIMEOUT") or merged.get("OPENAI_TIMEOUT"), cls.timeout),
            temperature=_parse_float(
                merged.get("OPENAI_COMPATIBLE_TEMPERATURE") or merged.get("OPENAI_TEMPERATURE"),
                cls.temperature,
            ),
            max_tokens=_parse_optional_int(
                merged.get("OPENAI_COMPATIBLE_MAX_TOKENS") or merged.get("OPENAI_MAX_TOKENS")
            )
            or cls.max_tokens,
            stream=_parse_bool(merged.get("OPENAI_COMPATIBLE_STREAM") or merged.get("OPENAI_STREAM"), cls.stream),
            use_env_proxy=_parse_bool(
                merged.get("OPENAI_COMPATIBLE_USE_ENV_PROXY") or merged.get("OPENAI_USE_ENV_PROXY"),
                cls.use_env_proxy,
            ),
        )

        clean_overrides = {key: value for key, value in overrides.items() if value is not None}
        if clean_overrides:
            config = replace(config, **clean_overrides)
        return config


class OpenAICompatibleChatClient:
    """Tiny stdlib client for /v1/chat/completions compatible APIs."""

    def __init__(self, config: Optional[OpenAICompatibleConfig] = None):
        self.config = config or OpenAICompatibleConfig.from_env()
        self.last_request: Dict[str, Any] = {}
        self.last_response: Dict[str, Any] = {}

    def __call__(self, prompt: str) -> str:
        if self.config.stream:
            raise RuntimeError("Streaming responses are not supported yet; set OPENAI_STREAM=false.")

        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": "Return only the JSON action requested by the user prompt.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": self.config.temperature,
            "stream": self.config.stream,
        }
        if self.config.max_tokens is not None:
            payload["max_tokens"] = self.config.max_tokens
        payload.update(self.config.extra_body)

        headers = {
            "Content-Type": "application/json",
            **self.config.extra_headers,
        }
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        self.last_request = {"url": self.config.chat_completions_url, "payload": payload, "headers": dict(headers)}
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.config.chat_completions_url,
            data=body,
            headers=headers,
            method="POST",
        )

        opener = (
            urllib.request.build_opener()
            if self.config.use_env_proxy
            else urllib.request.build_opener(urllib.request.ProxyHandler({}))
        )

        try:
            with opener.open(request, timeout=self.config.timeout) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI-compatible API HTTP {exc.code}: {detail[:500]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenAI-compatible API connection failed: {exc}") from exc
        except OSError as exc:
            raise RuntimeError(f"OpenAI-compatible API connection failed: {exc}") from exc

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"OpenAI-compatible API returned invalid JSON: {raw[:500]}") from exc
        self.last_response = data
        return self._extract_text(data)

    def _extract_text(self, data: Dict[str, Any]) -> str:
        choices = data.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            content = message.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in content
                )
        if "output_text" in data:
            return str(data["output_text"])
        raise RuntimeError("OpenAI-compatible API response did not include message content.")


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

    @classmethod
    def from_openai_compatible(
        cls,
        api_config: Optional[OpenAICompatibleConfig] = None,
        planner_config: Optional[LLMPlannerConfig] = None,
        **api_overrides: Any,
    ) -> "LLMPlanner":
        """Build a planner from a /v1/chat/completions compatible endpoint."""
        if api_config is None:
            api_config = OpenAICompatibleConfig.from_env(**api_overrides)
        elif api_overrides:
            api_config = replace(
                api_config,
                **{key: value for key, value in api_overrides.items() if value is not None},
            )
        return cls(llm=OpenAICompatibleChatClient(api_config), config=planner_config)

    @classmethod
    def from_openai_compatible_env(
        cls,
        env_file: Optional[str] = ".env",
        planner_config: Optional[LLMPlannerConfig] = None,
        **api_overrides: Any,
    ) -> "LLMPlanner":
        """Build a planner from environment variables and an optional .env file."""
        api_config = OpenAICompatibleConfig.from_env(env_file=env_file, **api_overrides)
        return cls.from_openai_compatible(api_config=api_config, planner_config=planner_config)
