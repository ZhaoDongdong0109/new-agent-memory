"""
本地 LLM 集成

支持 OpenAI 兼容 API 的本地模型（如 vLLM、LM Studio、Ollama 等）。
"""

import json
import os
import urllib.request
from typing import Callable, Optional


class LocalLLM:
    """
    本地 LLM 客户端

    支持 OpenAI 兼容 API 的本地模型。
    """

    def __init__(
        self,
        api_base: str = "http://localhost:8080/v1",
        api_key: str = "not-needed",
        model: str = "local-model",
        temperature: float = 0.1,
        max_tokens: int = 1000,
        timeout: int = 60,
    ):
        """
        Args:
            api_base: API 基础地址
            api_key: API 密钥（本地模型通常不需要）
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大生成 token 数
            timeout: 超时时间（秒）
        """
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        # 创建不使用代理的 opener
        proxy_handler = urllib.request.ProxyHandler({})
        self._opener = urllib.request.build_opener(proxy_handler)

    def __call__(self, prompt: str) -> str:
        """
        调用 LLM

        Args:
            prompt: 输入提示

        Returns:
            生成的文本
        """
        return self.generate(prompt)

    def generate(self, prompt: str) -> str:
        """
        生成文本

        Args:
            prompt: 输入提示

        Returns:
            生成的文本
        """
        url = f"{self.api_base}/chat/completions"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers)

        try:
            with self._opener.open(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result["choices"][0]["message"]["content"]
        except Exception as e:
            # 抛出类型化错误而不是静默返回空串：
            # 空串与"模型真的返回了空"无法区分，会让上游跳过规则回退路径；
            # LLMError 让 LLMPlanner 走降级路径、让抽取器回退到规则抽取。
            from core.llm_planner import LLMError
            raise LLMError(f"LocalLLM 调用失败 ({self.api_base}): {e}") from e

    @classmethod
    def from_env(cls, env_file: str = ".env") -> "LocalLLM":
        """
        从环境变量创建

        Args:
            env_file: .env 文件路径

        Returns:
            LocalLLM 实例
        """
        # 加载 .env 文件
        if os.path.exists(env_file):
            with open(env_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if "=" in line and not line.startswith("#"):
                        key, value = line.split("=", 1)
                        os.environ.setdefault(key.strip(), value.strip())

        def _safe_float(name, default):
            try:
                return float(os.environ.get(name, default))
            except (TypeError, ValueError):
                return default

        def _safe_int(name, default):
            try:
                return int(os.environ.get(name, default))
            except (TypeError, ValueError):
                return default

        return cls(
            api_base=os.environ.get("LOCAL_LLM_API_BASE", "http://localhost:8080/v1"),
            api_key=os.environ.get("LOCAL_LLM_API_KEY", "not-needed"),
            model=os.environ.get("LOCAL_LLM_MODEL", "local-model"),
            # 环境变量格式错误时使用默认值，而不是在导入期崩溃
            temperature=_safe_float("LOCAL_LLM_TEMPERATURE", 0.1),
            max_tokens=_safe_int("LOCAL_LLM_MAX_TOKENS", 1000),
            timeout=_safe_int("LOCAL_LLM_TIMEOUT", 60),
        )


def create_llm_function(
    api_base: str = None,
    api_key: str = None,
    model: str = None,
) -> Callable[[str], str]:
    """
    创建 LLM 函数

    Args:
        api_base: API 基础地址
        api_key: API 密钥
        model: 模型名称

    Returns:
        LLM 函数
    """
    # 从环境变量读取配置
    if api_base is None:
        api_base = os.environ.get("LOCAL_LLM_API_BASE", "http://localhost:8080/v1")
    if api_key is None:
        api_key = os.environ.get("LOCAL_LLM_API_KEY", "not-needed")
    if model is None:
        model = os.environ.get("LOCAL_LLM_MODEL", "local-model")

    llm = LocalLLM(
        api_base=api_base,
        api_key=api_key,
        model=model,
    )

    return llm


def create_memory_extractor_llm() -> Optional[Callable[[str], str]]:
    """
    创建用于 MemoryExtractor 的 LLM 函数

    Returns:
        LLM 函数，如果未配置则返回 None
    """
    api_base = os.environ.get("LOCAL_LLM_API_BASE")
    if not api_base:
        return None

    return create_llm_function(api_base=api_base)
