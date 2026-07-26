#!/usr/bin/env python3
"""
MCP Server - 让 Claude Code 直接使用记忆系统

启动后，Claude Code 可以通过 MCP 协议：
- 检索记忆
- 添加记忆
- 查看统计

使用方式：
    # 在 Claude Code 的 MCP 配置中添加：
    {
        "mcpServers": {
            "memory": {
                "command": "python",
                "args": ["mcp_server.py"]
            }
        }
    }
"""

import contextlib
import json
import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 加载 .env 文件
def load_dotenv():
    """加载 .env 文件"""
    env_file = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_file):
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())

load_dotenv()

from main import HumanLikeMemorySystem  # noqa: E402  必须在 sys.path 设置之后导入


def _probe_llm_endpoint(base_url: str, timeout: float = 2.0) -> bool:
    """探测本地 LLM 端点是否可达（短超时，绕过代理）

    只要端点返回了任意 HTTP 响应（包括 4xx/5xx）就认为可达；
    连接失败 / 超时则认为不可达。
    """
    import urllib.error
    import urllib.request

    # 本地端点不走代理，避免代理环境变量干扰探测结果
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        resp = opener.open(f"{base_url}/models", timeout=timeout)
        resp.close()
        return True
    except urllib.error.HTTPError:
        # 收到 HTTP 错误响应，说明端点本身可达
        return True
    except Exception:
        return False


def create_llm_fn():
    """创建 LLM 函数（使用本地 LM Studio）

    启动时先用短超时探测端点；不可达时返回 None，
    让下游走规则抽取路径，避免每次调用挂起 60 秒。
    """
    # 本地 LM Studio 配置（WSL 内部访问 Windows 主机）
    base_url = os.getenv("LOCAL_LLM_API_BASE", "http://172.28.32.1:1255/v1")
    model = os.getenv("LOCAL_LLM_MODEL", "google/gemma-4-e4b")

    if not _probe_llm_endpoint(base_url):
        print(f"[MCP] 本地 LLM 不可达: {base_url}，使用规则抽取模式", file=sys.stderr)
        return None

    try:
        import urllib.request

        def llm_fn(prompt: str) -> str:
            """调用本地 LLM API（绕过代理）"""
            # 绕过代理
            import os
            old_proxy = {}
            for key in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
                old_proxy[key] = os.environ.pop(key, None)

            try:
                headers = {
                    "Content-Type": "application/json",
                }

                payload = json.dumps({
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 1000,
                }).encode("utf-8")

                req = urllib.request.Request(
                    f"{base_url}/chat/completions",
                    data=payload,
                    headers=headers,
                    method="POST",
                )

                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    message = data["choices"][0]["message"]
                    # 优先返回 content，如果没有则返回 reasoning_content
                    return message.get("content") or message.get("reasoning_content", "")
            finally:
                # 恢复代理设置
                for key, value in old_proxy.items():
                    if value is not None:
                        os.environ[key] = value

        # 诊断信息一律写 stderr，stdout 只承载 JSON-RPC 消息
        print(f"[MCP] 本地 LLM 已启用: {base_url} / {model}", file=sys.stderr)
        return llm_fn

    except Exception as e:
        print(f"[MCP] 本地 LLM 初始化失败: {e}，使用规则抽取模式", file=sys.stderr)
        return None


class MemoryMCPServer:
    """MCP Server for memory system"""

    def __init__(self):
        # 尝试创建 LLM 函数
        llm_fn = create_llm_fn()

        # 数据目录：优先 MEMORY_DATA_DIR 环境变量，
        # 否则锚定到本文件所在目录，避免随客户端 cwd 变化导致记忆分散
        data_dir = os.getenv("MEMORY_DATA_DIR")
        if not data_dir:
            data_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "memory_data"
            )

        self.memory = HumanLikeMemorySystem(
            data_dir=data_dir,
            store_backend="sqlite",
            llm_fn=llm_fn,
        )

        self.memory.load()

    def handle_request(self, request: dict) -> dict:
        """处理 MCP 请求，返回完整的 JSON-RPC 响应对象"""
        method = request.get("method")
        params = request.get("params", {})
        request_id = request.get("id")

        if method == "initialize":
            result = self._initialize()
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = self._list_tools()
        elif method == "tools/call":
            tool_name = params.get("name")
            if tool_name not in ("memory_search", "memory_add", "memory_stats"):
                # 未知工具：按 JSON-RPC 规范返回顶层 error（Invalid params）
                return self._error_response(
                    request_id, -32602, f"Unknown tool: {tool_name}"
                )
            result = self._call_tool(params)
        elif method == "resources/list":
            result = {"resources": []}
        elif method == "prompts/list":
            result = {"prompts": []}
        else:
            # 未知方法：按 JSON-RPC 规范返回顶层 error（Method not found）
            return self._error_response(
                request_id, -32601, f"Method not found: {method}"
            )

        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    @staticmethod
    def _error_response(request_id, code: int, message: str) -> dict:
        """构造顶层 JSON-RPC 错误响应"""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        }

    def _initialize(self) -> dict:
        """初始化"""
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {}
            },
            "serverInfo": {
                "name": "new-agent-memory",
                "version": "0.1.0"
            }
        }

    def _list_tools(self) -> dict:
        """列出可用工具"""
        return {
            "tools": [
                {
                    "name": "memory_search",
                    "description": "Search through long-term memories. Use this when you need to recall past conversations, facts, or experiences.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "What to search for in memories"
                            }
                        },
                        "required": ["query"]
                    }
                },
                {
                    "name": "memory_add",
                    "description": (
                        "Save something to long-term memory. Use this to remember "
                        "important facts, conversations, or experiences. Note: when "
                        "local LLM extraction is enabled, the LLM may infer its own "
                        "importance; an explicitly provided 'importance' argument "
                        "always takes precedence and is applied to the stored memory."
                    ),
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "What to remember"
                            },
                            "importance": {
                                "type": "number",
                                "description": "How important (0.0-1.0, default 0.5). If explicitly provided, this value overrides any LLM-inferred importance.",
                                "default": 0.5
                            },
                            "topics": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Related topics"
                            }
                        },
                        "required": ["content"]
                    }
                },
                {
                    "name": "memory_stats",
                    "description": "Get memory system statistics",
                    "inputSchema": {
                        "type": "object",
                        "properties": {}
                    }
                }
            ]
        }

    def _call_tool(self, params: dict) -> dict:
        """调用工具

        工具执行失败时按 MCP 规范返回 result.content + isError=true，
        而不是协议级错误。
        """
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        try:
            if tool_name == "memory_search":
                return self._search(arguments)
            elif tool_name == "memory_add":
                return self._add(arguments)
            elif tool_name == "memory_stats":
                return self._stats()
            else:
                return {
                    "content": [
                        {"type": "text", "text": f"Unknown tool: {tool_name}"}
                    ],
                    "isError": True,
                }
        except Exception as e:
            return {
                "content": [
                    {"type": "text", "text": f"Tool execution failed: {e}"}
                ],
                "isError": True,
            }

    def _search(self, args: dict) -> dict:
        """检索记忆"""
        query = args.get("query", "")
        result = self.memory.retrieve(query, allow_forgotten=True)

        if result.success:
            memories = []
            for chunk in result.chunks[:5]:
                memories.append({
                    "content": chunk.content[:200],
                    "importance": chunk.importance,
                    "topics": list(chunk.topics)[:5],
                })
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Found {len(result.chunks)} memories:\n\n" +
                                "\n---\n".join(
                                    f"[{m['importance']:.1f}] {m['content']}"
                                    for m in memories
                                )
                    }
                ]
            }
        else:
            return {
                "content": [{"type": "text", "text": "No relevant memories found."}]
            }

    def _add(self, args: dict) -> dict:
        """添加记忆（支持 LLM 增强抽取）"""
        content = args.get("content", "")
        importance = args.get("importance", 0.5)
        topics = args.get("topics", [])

        # 使用 add_raw_memory 进行自动抽取
        # 如果有 LLM，会使用语义抽取；否则使用规则抽取
        memory_id = self.memory.add_raw_memory(
            text=content,
            importance=importance,
            check_duplicate=True,
        )

        # 用户显式指定的 topics / importance 需要写回记忆，
        # 并通过核心层重新入库以更新倒排索引
        # （直接 _store.put 会绕过索引，导致按主题检索漏掉该记忆）
        chunk = self.memory.core.get(memory_id)
        if chunk:
            changed = False
            if topics:
                chunk.topics.update(topics)
                changed = True
            # 显式传入的 importance 优先于 LLM 抽取推断的值
            if "importance" in args and isinstance(importance, (int, float)):
                chunk.importance = float(importance)
                changed = True
            if changed:
                self.memory.core.add(chunk)

        return {
            "content": [{"type": "text", "text": f"Memory saved: {memory_id}"}]
        }

    def _stats(self) -> dict:
        """获取统计"""
        stats = self.memory.get_memory_stats()
        return {
            "content": [{
                "type": "text",
                "text": json.dumps(stats, indent=2, ensure_ascii=False)
            }]
        }


def main():
    """主循环 - 读取 stdin，写入 stdout

    stdout 只承载 JSON-RPC 消息；底层库通过 print 输出的诊断信息
    全部重定向到 stderr，避免污染 stdio 协议通道。
    """
    stdout = sys.stdout

    def write_message(message: dict):
        """向真实 stdout 写一条 JSON-RPC 消息"""
        stdout.write(json.dumps(message) + "\n")
        stdout.flush()

    with contextlib.redirect_stdout(sys.stderr):
        # 初始化响应只在收到客户端的 initialize 请求后发送，
        # 不主动发送未经请求的消息
        server = MemoryMCPServer()

        # 处理请求
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            # 每轮迭代重置 request_id，避免异常响应关联到上一条请求的 id
            request_id = None
            try:
                try:
                    request = json.loads(line)
                except json.JSONDecodeError:
                    write_message({
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": -32700, "message": "Parse error"},
                    })
                    continue

                request_id = request.get("id")

                # 处理 notifications（没有 id 的消息不需要响应）
                if request_id is None:
                    continue

                write_message(server.handle_request(request))

            except Exception as e:
                write_message({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32603, "message": str(e)},
                })


if __name__ == "__main__":
    main()
