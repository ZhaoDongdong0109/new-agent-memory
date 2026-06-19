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

from main import HumanLikeMemorySystem


def create_llm_fn():
    """创建 LLM 函数（使用本地 LM Studio）"""
    # 本地 LM Studio 配置（WSL 内部访问 Windows 主机）
    base_url = os.getenv("LOCAL_LLM_API_BASE", "http://172.28.32.1:1255/v1")
    model = os.getenv("LOCAL_LLM_MODEL", "google/gemma-4-e4b")

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

        print(f"[MCP] 本地 LLM 已启用: {base_url} / {model}")
        return llm_fn

    except Exception as e:
        print(f"[MCP] 本地 LLM 初始化失败: {e}，使用规则抽取模式")
        return None


class MemoryMCPServer:
    """MCP Server for memory system"""

    def __init__(self):
        # 尝试创建 LLM 函数
        llm_fn = create_llm_fn()

        self.memory = HumanLikeMemorySystem(
            data_dir="./memory_data",
            store_backend="sqlite",
            llm_fn=llm_fn,
        )

        self.memory.load()

    def handle_request(self, request: dict) -> dict:
        """处理 MCP 请求"""
        method = request.get("method")
        params = request.get("params", {})

        if method == "initialize":
            return self._initialize()
        elif method == "tools/list":
            return self._list_tools()
        elif method == "tools/call":
            return self._call_tool(params)
        else:
            return {"error": f"Unknown method: {method}"}

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
                    "description": "Save something to long-term memory. Use this to remember important facts, conversations, or experiences.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "What to remember"
                            },
                            "importance": {
                                "type": "number",
                                "description": "How important (0.0-1.0, default 0.5)",
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
        """调用工具"""
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
                return {"error": f"Unknown tool: {tool_name}"}
        except Exception as e:
            return {"error": str(e)}

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

        # 如果用户指定了 topics，手动添加到记忆
        if topics:
            chunk = self.memory.core._store.get(memory_id)
            if chunk:
                chunk.topics.update(topics)
                self.memory.core._store.put(chunk)

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
    """主循环 - 读取 stdin，写入 stdout"""
    server = MemoryMCPServer()

    # 发送初始化响应
    response = server._initialize()
    print(json.dumps({"jsonrpc": "2.0", "id": 0, "result": response}))
    sys.stdout.flush()

    # 处理请求
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
            request_id = request.get("id")

            # 处理 notifications（没有 id）
            if request_id is None:
                method = request.get("method")
                if method == "notifications/initialized":
                    continue
                elif method == "notifications/cancelled":
                    continue
                # 其他 notification 忽略
                continue

            result = server.handle_request(request)
            response = {"jsonrpc": "2.0", "id": request_id, "result": result}
            print(json.dumps(response))
            sys.stdout.flush()

        except json.JSONDecodeError:
            continue
        except Exception as e:
            error_response = {
                "jsonrpc": "2.0",
                "id": request_id if 'request_id' in dir() else 0,
                "error": {"code": -1, "message": str(e)}
            }
            print(json.dumps(error_response))
            sys.stdout.flush()


if __name__ == "__main__":
    main()
