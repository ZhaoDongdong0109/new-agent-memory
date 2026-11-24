"""MCP 客户端（stdio 传输）——让 CognitiveAgent 消费外部 MCP 工具服务。

大脑主循环 v2 组件：此前仓库只提供 `mcp_server.py` 对外暴露记忆工具，
Agent 自身无法连接外部工具生态（文件系统、搜索、数据库等）。本模块用
纯标准库实现跨平台（含 Windows）的 MCP stdio 客户端与工具桥接器：

- 使用后台线程 + Condition 读取管道，避免在 Windows 管道上调用
  `select()` 触发 WinError 10038；
- 二进制管道显式 UTF-8 编解码，杜绝 Windows 默认 GBK 编码崩溃；
- 支持 `tools/list` 游标翻页、工具名合法化（满足 OpenAI 64 字符与字符集
  约束）、参数 JSON Schema 透传、单服务故障隔离（MCPToolHub）。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import threading
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

from core.agent_system import ActionResult, CognitiveAgent


class MCPError(RuntimeError):
    """MCP 传输层或协议层的类型化错误。"""


def sanitize_mcp_tool_name(server: str, tool_name: str) -> str:
    """生成合法且不超过 64 字符的工具名：`mcp__{server}__{tool}`。

    OpenAI / 主流函数调用协议要求工具名只含 `[a-zA-Z0-9_-]` 且长度 <= 64。
    超长时附加确定性短哈希后缀，避免同前缀长工具名截断后互相碰撞。
    """
    clean_server = re.sub(r"[^a-zA-Z0-9_-]+", "_", (server or "mcp").strip()).strip("_") or "mcp"
    clean_tool = re.sub(r"[^a-zA-Z0-9_-]+", "_", (tool_name or "tool").strip()).strip("_") or "tool"
    full = f"mcp__{clean_server}__{clean_tool}"
    if len(full) <= 64:
        return full
    digest = hashlib.sha256(f"{server}::{tool_name}".encode("utf-8")).hexdigest()[:8]
    head = full[: 64 - 9].rstrip("_")
    return f"{head}_{digest}"


def mcp_result_to_action_result(
    result: Mapping[str, Any],
    server: str = "",
    tool: str = "",
) -> ActionResult:
    """把 MCP `tools/call` 返回结构转换为 Agent 的 `ActionResult`。"""
    is_error = bool(result.get("isError", False))
    raw_items = result.get("content")
    parts: List[str] = []

    if isinstance(raw_items, list):
        for item in raw_items:
            if not isinstance(item, Mapping):
                parts.append(str(item))
                continue
            item_type = str(item.get("type", "text"))
            if item_type == "text":
                parts.append(str(item.get("text", "")))
            else:
                detail = item.get("mimeType") or item.get("uri") or ""
                tag = f"[{item_type}: {detail}]" if detail else f"[{item_type}]"
                parts.append(tag)
    elif raw_items is not None:
        parts.append(str(raw_items))

    output = "\n".join(p for p in parts if p != "")
    return ActionResult(
        success=not is_error,
        output=output,
        metadata={
            "kind": "mcp_tool",
            "server": server,
            "tool": tool,
            "is_error": is_error,
        },
    )


class MCPStdioClient:
    """基于子进程标准输入/输出的 MCP 客户端（JSON-RPC 2.0 行协议）。"""

    def __init__(
        self,
        command: str,
        args: Sequence[str] = (),
        env: Optional[Mapping[str, str]] = None,
        cwd: Optional[str] = None,
        name: str = "",
        timeout: float = 30.0,
        protocol_version: str = "2024-11-05",
    ):
        self.command = command
        self.args = list(args)
        self.env = dict(env) if env is not None else None
        self.cwd = cwd
        self.name = name or os.path.splitext(os.path.basename(command))[0] or "mcp"
        self.timeout = float(timeout)
        self.protocol_version = protocol_version

        self._proc: Optional[subprocess.Popen] = None
        self._write_lock = threading.Lock()
        self._cond = threading.Condition()
        self._responses: Dict[Any, Dict[str, Any]] = {}
        self._next_id = 0
        self._eof = False
        self._closing = False
        self._stdout_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self.stderr_tail: List[str] = []
        self.server_info: Dict[str, Any] = {}
        self.capabilities: Dict[str, Any] = {}

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None and not self._eof

    def __enter__(self) -> "MCPStdioClient":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def start(self) -> Dict[str, Any]:
        """启动子进程并完成 MCP `initialize` + `notifications/initialized` 握手。"""
        if self.is_running and self.server_info:
            return {"serverInfo": self.server_info, "capabilities": self.capabilities}

        merged_env = dict(os.environ)
        if self.env:
            merged_env.update({str(k): str(v) for k, v in self.env.items()})

        try:
            self._proc = subprocess.Popen(
                [self.command, *self.args],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=merged_env,
                cwd=self.cwd,
            )
        except OSError as exc:
            raise MCPError(f"Failed to start MCP server '{self.name}' ({self.command}): {exc}") from exc

        self._eof = False
        self._closing = False
        self._responses.clear()
        self.stderr_tail.clear()

        self._stdout_thread = threading.Thread(
            target=self._read_stdout_loop,
            name=f"mcp-stdout-{self.name}",
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._read_stderr_loop,
            name=f"mcp-stderr-{self.name}",
            daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread.start()

        try:
            init_result = self.request(
                "initialize",
                {
                    "protocolVersion": self.protocol_version,
                    "capabilities": {},
                    "clientInfo": {"name": "new-agent-memory", "version": "0.2.0"},
                },
            )
            if isinstance(init_result, dict):
                self.server_info = dict(init_result.get("serverInfo") or {})
                self.capabilities = dict(init_result.get("capabilities") or {})
            self.notify("notifications/initialized", {})
            return init_result if isinstance(init_result, dict) else {}
        except Exception:
            self.close()
            raise

    def _read_stdout_loop(self) -> None:
        proc = self._proc
        stream = proc.stdout if proc is not None else None
        if stream is None:
            with self._cond:
                self._eof = True
                self._cond.notify_all()
            return

        try:
            while not self._closing:
                raw = stream.readline()
                if not raw:
                    break
                text = raw.decode("utf-8", errors="replace").strip()
                if not text:
                    continue
                try:
                    msg = json.loads(text)
                except json.JSONDecodeError:
                    # 忽略服务端偶然打到 stdout 的非 JSON 诊断行
                    continue
                if not isinstance(msg, dict):
                    continue
                if "id" in msg and msg["id"] is not None:
                    with self._cond:
                        self._responses[msg["id"]] = msg
                        self._cond.notify_all()
        except Exception:
            pass
        finally:
            with self._cond:
                self._eof = True
                self._cond.notify_all()

    def _read_stderr_loop(self) -> None:
        proc = self._proc
        stream = proc.stderr if proc is not None else None
        if stream is None:
            return
        try:
            while not self._closing:
                raw = stream.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                if line:
                    self.stderr_tail.append(line)
                    if len(self.stderr_tail) > 20:
                        del self.stderr_tail[:-20]
        except Exception:
            pass

    def _send_json(self, payload: Dict[str, Any]) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None or proc.poll() is not None:
            raise MCPError(f"MCP server '{self.name}' is not running.")
        encoded = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        with self._write_lock:
            try:
                proc.stdin.write(encoded)
                proc.stdin.flush()
            except OSError as exc:
                raise MCPError(f"Failed to write to MCP server '{self.name}': {exc}") from exc

    def notify(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        """发送无需响应的 JSON-RPC 通知。"""
        payload: Dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        self._send_json(payload)

    def request(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        """发送 JSON-RPC 请求并按 id 等待响应。"""
        if self._proc is None:
            raise MCPError(f"MCP server '{self.name}' has not been started.")

        wait_timeout = self.timeout if timeout is None else float(timeout)
        with self._cond:
            self._next_id += 1
            req_id = self._next_id

        payload: Dict[str, Any] = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params is not None:
            payload["params"] = params
        self._send_json(payload)

        deadline = time.monotonic() + wait_timeout
        with self._cond:
            while req_id not in self._responses:
                if self._eof:
                    detail = f" stderr={self.stderr_tail[-3:]}" if self.stderr_tail else ""
                    raise MCPError(
                        f"MCP server '{self.name}' closed connection while waiting for '{method}'.{detail}"
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise MCPError(
                        f"MCP server '{self.name}' timed out after {wait_timeout:.2f}s on '{method}'."
                    )
                self._cond.wait(timeout=remaining)

            resp = self._responses.pop(req_id)

        if "error" in resp and resp["error"] is not None:
            err = resp["error"]
            if isinstance(err, Mapping):
                code = err.get("code")
                message = err.get("message", "Unknown MCP error")
                raise MCPError(f"MCP error {code} on '{method}': {message}")
            raise MCPError(f"MCP error on '{method}': {err}")

        return resp.get("result")

    def list_tools(self) -> List[Dict[str, Any]]:
        """列出外部服务暴露的全部工具（自动处理 nextCursor 翻页）。"""
        tools: List[Dict[str, Any]] = []
        cursor: Optional[str] = None
        seen_cursors: Set[str] = set()

        while True:
            params: Dict[str, Any] = {"cursor": cursor} if cursor else {}
            result = self.request("tools/list", params if params else None)
            if not isinstance(result, Mapping):
                break
            batch = result.get("tools") or []
            if isinstance(batch, list):
                for item in batch:
                    if isinstance(item, Mapping) and item.get("name"):
                        tools.append(dict(item))
            next_cursor = result.get("nextCursor")
            if not next_cursor or not isinstance(next_cursor, str) or next_cursor in seen_cursors:
                break
            seen_cursors.add(next_cursor)
            cursor = next_cursor

        return tools

    def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """调用远端工具并返回原始 MCP result 字典。"""
        result = self.request(
            "tools/call",
            {"name": name, "arguments": dict(arguments or {})},
        )
        if isinstance(result, dict):
            return result
        return {"content": [{"type": "text", "text": str(result or "")}], "isError": False}

    def close(self) -> None:
        """优雅关闭子进程并释放全部管道句柄（Windows 文件锁安全）。"""
        self._closing = True
        proc = self._proc
        self._proc = None
        if proc is None:
            return

        for pipe in (proc.stdin,):
            if pipe is not None:
                try:
                    pipe.close()
                except Exception:
                    pass

        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=2.0)
            except Exception:
                try:
                    proc.kill()
                    proc.wait(timeout=2.0)
                except Exception:
                    pass

        for pipe in (proc.stdout, proc.stderr):
            if pipe is not None:
                try:
                    pipe.close()
                except Exception:
                    pass

        with self._cond:
            self._eof = True
            self._cond.notify_all()

        for thread in (self._stdout_thread, self._stderr_thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout=1.0)


def register_mcp_tools(
    agent: CognitiveAgent,
    client: MCPStdioClient,
    prefix: Optional[str] = None,
    include: Optional[Set[str]] = None,
    cost: float = 0.05,
) -> List[str]:
    """把一个 MCPStdioClient 的工具注册进 CognitiveAgent 的 ToolRegistry。

    返回实际新注册的工具名列表；若与已有工具重名则跳过（绝不覆盖内置工具）。
    """
    if not client.is_running:
        client.start()

    server_label = (prefix or client.name or "mcp").strip()
    registered: List[str] = []

    for spec in client.list_tools():
        raw_name = str(spec.get("name", "")).strip()
        if not raw_name:
            continue
        if include is not None and raw_name not in include:
            continue

        reg_name = sanitize_mcp_tool_name(server_label, raw_name)
        if reg_name in agent.tools.tools:
            continue

        raw_desc = str(spec.get("description") or f"Call MCP tool '{raw_name}'.").strip()
        description = f"[MCP:{server_label}] {raw_desc}"
        schema = spec.get("inputSchema")
        parameters = dict(schema) if isinstance(schema, Mapping) else {"type": "object", "properties": {}}

        def _make_handler(tool_name: str):
            def _handler(arguments: Dict[str, Any]) -> ActionResult:
                try:
                    mcp_res = client.call_tool(tool_name, arguments)
                except MCPError as exc:
                    return ActionResult(
                        False,
                        f"MCP tool '{tool_name}' error: {exc}",
                        metadata={
                            "kind": "mcp_tool",
                            "server": server_label,
                            "tool": tool_name,
                            "mcp_error": str(exc),
                        },
                    )
                return mcp_result_to_action_result(mcp_res, server=server_label, tool=tool_name)

            return _handler

        agent.add_tool(
            name=reg_name,
            description=description,
            handler=_make_handler(raw_name),
            cost=cost,
            parameters=parameters,
        )
        registered.append(reg_name)

    return registered


def load_mcp_config(path: str) -> Dict[str, Dict[str, Any]]:
    """读取标准 `.mcp.json` / `mcp_config.json` 中的 stdio 服务配置。

    仅保留包含 `command` 且未设置 `disabled: true` 的条目；纯 HTTP/SSE 条目会被跳过。
    """
    if not path or not os.path.exists(path):
        raise MCPError(f"MCP config file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise MCPError(f"Failed to load MCP config '{path}': {exc}") from exc

    raw_servers = data.get("mcpServers") if isinstance(data, Mapping) else None
    if not isinstance(raw_servers, Mapping):
        return {}

    servers: Dict[str, Dict[str, Any]] = {}
    for name, entry in raw_servers.items():
        if not isinstance(entry, Mapping):
            continue
        if entry.get("disabled") is True:
            continue
        command = entry.get("command")
        if not command or not isinstance(command, str):
            continue
        raw_args = entry.get("args") or []
        args = [str(a) for a in raw_args] if isinstance(raw_args, list) else []
        raw_env = entry.get("env")
        env = {str(k): str(v) for k, v in raw_env.items()} if isinstance(raw_env, Mapping) else None
        cwd = str(entry["cwd"]) if isinstance(entry.get("cwd"), str) else None
        servers[str(name)] = {
            "command": command,
            "args": args,
            "env": env,
            "cwd": cwd,
        }

    return servers


class MCPToolHub:
    """管理多个外部 MCP 服务并把工具批量挂载到 CognitiveAgent。

    单个外部服务启动失败时记录到 `self.errors`，不会阻断其余健康服务。
    """

    def __init__(self, specs: Mapping[str, Mapping[str, Any]], timeout: float = 30.0):
        self.specs = {str(k): dict(v) for k, v in specs.items()}
        self.timeout = float(timeout)
        self.clients: Dict[str, MCPStdioClient] = {}
        self.errors: Dict[str, str] = {}

    @classmethod
    def from_config(cls, path: str, timeout: float = 30.0) -> "MCPToolHub":
        return cls(load_mcp_config(path), timeout=timeout)

    def __enter__(self) -> "MCPToolHub":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def attach(self, agent: CognitiveAgent) -> List[str]:
        """启动所有配置的 MCP 服务并将可用工具注册进 `agent`。"""
        all_registered: List[str] = []
        for name, cfg in self.specs.items():
            client = MCPStdioClient(
                command=str(cfg["command"]),
                args=cfg.get("args") or [],
                env=cfg.get("env"),
                cwd=cfg.get("cwd"),
                name=name,
                timeout=self.timeout,
            )
            try:
                client.start()
                names = register_mcp_tools(agent, client, prefix=name)
            except Exception as exc:
                self.errors[name] = f"{type(exc).__name__}: {exc}"
                client.close()
                continue
            self.clients[name] = client
            all_registered.extend(names)
        return all_registered

    def close(self) -> None:
        for client in list(self.clients.values()):
            client.close()
        self.clients.clear()
