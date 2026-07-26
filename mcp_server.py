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
import time

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

        # 数据目录优先级：
        # 1. MEMORY_DATA_DIR 环境变量（显式指定）
        # 2. 当前工作目录已存在的 ./memory_data（历史部署的存量记忆，
        #    直接切走会造成"静默失忆"）
        # 3. 本文件所在目录（新部署的默认值，避免随客户端 cwd 分散）
        data_dir = os.getenv("MEMORY_DATA_DIR")
        if not data_dir:
            legacy_dir = os.path.join(os.getcwd(), "memory_data")
            if os.path.isdir(legacy_dir) and os.listdir(legacy_dir):
                data_dir = legacy_dir
                print(f"[MCP] 使用现有记忆目录: {legacy_dir}", file=sys.stderr)
            else:
                data_dir = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "memory_data"
                )

        self.memory = HumanLikeMemorySystem(
            data_dir=data_dir,
            store_backend="sqlite",
            llm_fn=llm_fn,
        )

        self.memory.load()

        # 工具名 -> 处理器。唯一事实来源：handle_request 的白名单与
        # _call_tool 的分发都从这里读。狗粮期第二轮实测：白名单曾
        # 硬编码 3 个工具名，7 个认知工具在真实 JSON-RPC 路径上全部
        # "Unknown tool"——而狗粮脚本直接调 _call_tool 绕过了它。
        self._tool_handlers = {
            "memory_search": self._search,
            "memory_add": self._add,
            "memory_stats": lambda args: self._stats(),
            "memory_explain": self._explain,
            "memory_history": self._history,
            "memory_sleep": lambda args: self._sleep(),
            "memory_focus": self._focus,
            "memory_feedback": self._feedback,
            "memory_maintain": lambda args: self._maintain(),
        }

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
            if tool_name not in self._tool_handlers:
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
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Max memories to return (1-20, default 5). Weaker tail matches beyond the limit are omitted and counted in the note.",
                                "default": 5
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
                            },
                            "memory_type": {
                                "type": "string",
                                "enum": ["interaction", "fact", "preference", "story", "idea", "procedure"],
                                "description": (
                                    "Memory type. 'fact'/'preference'/'procedure' get "
                                    "bi-temporal supersession (new values replace "
                                    "old, history preserved) and slower decay; "
                                    "'procedure' (how-to/workflow knowledge) decays "
                                    "slowest, then 'story'; default 'interaction' "
                                    "decays fastest. Use 'fact' for durable facts."
                                ),
                                "default": "interaction"
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
                },
                {
                    "name": "memory_explain",
                    "description": (
                        "Explain WHY a memory ranks where it does: full ACT-R "
                        "activation breakdown (retention, per-factor weights, "
                        "recall feedback bias), layer/lifecycle state, "
                        "supersession links, and strongest associations. "
                        "Use the memory id returned by memory_search."
                    ),
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "memory_id": {
                                "type": "string",
                                "description": "Memory id (mem_...)"
                            }
                        },
                        "required": ["memory_id"]
                    }
                },
                {
                    "name": "memory_history",
                    "description": (
                        "Show the bi-temporal supersession chain of a fact: "
                        "current value plus every superseded historical value "
                        "with validity periods ('used to live in Lisbon')."
                    ),
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "memory_id": {
                                "type": "string",
                                "description": "Any memory id on the chain"
                            }
                        },
                        "required": ["memory_id"]
                    }
                },
                {
                    "name": "memory_sleep",
                    "description": (
                        "Run one deterministic sleep-consolidation cycle: "
                        "related episodic memories are abstracted into a "
                        "slow-decaying gist (sources archived but cue-wakeable). "
                        "Returns a full audit report."
                    ),
                    "inputSchema": {
                        "type": "object",
                        "properties": {}
                    }
                },
                {
                    "name": "memory_focus",
                    "description": (
                        "Goal-driven attention workspace: given the current "
                        "query/task, returns what the system should be "
                        "thinking about right now (relevant memories, "
                        "triggered procedures, active goal), with explainable "
                        "scores. Different from memory_search: answers "
                        "'what should I think about', not 'what do I remember'."
                    ),
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Current task or question"
                            }
                        },
                        "required": ["query"]
                    }
                },
                {
                    "name": "memory_feedback",
                    "description": (
                        "Tell the memory system whether its last recall for a "
                        "query was right or wrong. Confirmed memories gain "
                        "persistent weight; corrected ones sink toward "
                        "forgetting. This is how the system learns from use."
                    ),
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The query that was answered"
                            },
                            "accepted": {
                                "type": "boolean",
                                "description": "true = recall was correct"
                            }
                        },
                        "required": ["query", "accepted"]
                    }
                },
                {
                    "name": "memory_maintain",
                    "description": (
                        "Run memory maintenance: sleep-consolidation if due, "
                        "demote decayed memories to the pseudo-forgotten "
                        "layer, clean up. Normally automatic; call to force."
                    ),
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
            handler = self._tool_handlers.get(tool_name)
            if handler is None:
                return {
                    "content": [
                        {"type": "text", "text": f"Unknown tool: {tool_name}"}
                    ],
                    "isError": True,
                }
            return handler(arguments)
        except Exception as e:
            return {
                "content": [
                    {"type": "text", "text": f"Tool execution failed: {e}"}
                ],
                "isError": True,
            }

    def _search(self, args: dict) -> dict:
        """检索记忆（返回 id 供 memory_explain / memory_history 溯源）"""
        query = args.get("query", "")
        try:
            limit = int(args.get("limit", 5))
        except (TypeError, ValueError):
            limit = 5
        limit = max(1, min(20, limit))
        result = self.memory.retrieve(query, allow_forgotten=True, limit=limit)

        if result.success:
            lines = [
                f"Found {len(result.chunks)} memories "
                f"(path={result.retrieval_path}, confidence={result.confidence:.2f})"
            ]
            if result.review_note:
                lines.append(f"note: {result.review_note}")
            lines.append("")
            for chunk in result.chunks:
                topics = ",".join(list(chunk.topics)[:4])
                lines.append(
                    f"[{chunk.id}] (imp={chunk.importance:.1f}"
                    + (f", topics={topics}" if topics else "")
                    + f") {chunk.content[:200]}"
                )
            return {"content": [{"type": "text", "text": "\n".join(lines)}]}
        else:
            # 可审计弃答：把拒答理由原样带给客户端
            text = "No relevant memories found."
            if result.review_note:
                text += f" ({result.review_note})"
            return {"content": [{"type": "text", "text": text}]}

    def _find_chunk(self, memory_id: str):
        """核心层优先，其次伪遗忘层"""
        chunk = self.memory.core.get(memory_id)
        if chunk is not None:
            return chunk, "core"
        chunk = self.memory.forgotten.get(memory_id)
        if chunk is not None:
            return chunk, "forgotten"
        return None, None

    def _explain(self, args: dict) -> dict:
        """可解释性：这条记忆为什么排在这里"""
        memory_id = args.get("memory_id", "")
        chunk, layer = self._find_chunk(memory_id)
        if chunk is None:
            return {
                "content": [{"type": "text", "text": f"Memory not found: {memory_id}"}],
                "isError": True,
            }

        from core.weight_system import actr_decay
        wf = self.memory.core.calc_weight(chunk)

        def _ts(value):
            return time.strftime("%Y-%m-%d %H:%M", time.localtime(value)) if value else "-"

        report = {
            "id": chunk.id,
            "layer": layer,
            "memory_type": getattr(chunk.memory_type, "value", str(chunk.memory_type)),
            "content_preview": chunk.content[:120],
            "lifecycle": {
                "created_at": _ts(chunk.created_at),
                "last_accessed": _ts(chunk.last_accessed),
                "access_count": chunk.access_count,
                "successful_recalls": chunk.successful_recall_count,
                "valid_at": _ts(chunk.valid_at),
                "invalid_at": _ts(chunk.invalid_at),
                "superseded_by": chunk.metadata.get("superseded_by"),
                "supersedes": chunk.parent_id,
                "consolidated_into": chunk.metadata.get("consolidated_into"),
            },
            "actr_activation": {
                "B": round(wf.activation, 3),
                "retention_P": round(wf.retention, 4),
                "decay_d_for_type": actr_decay(chunk.memory_type),
                # Pavlik 间隔效应：最近各次使用事件的衰减速率
                # （复习时激活越高越接近上限，越低越接近类型基线）
                "event_decays_recent": [
                    round(dj, 3) if dj is not None else None
                    for dj in chunk.access_decays[-5:]
                ],
            },
            "weight_factors": {
                "emotion_boost": round(wf.emotion_boost, 4),
                "association_density": round(wf.association_density, 4),
                "importance": round(wf.importance_base, 4),
                "connection": round(wf.connection_boost, 4),
                "recall_bias": round(wf.recall_bias, 4),
                "final_weight": round(wf.final, 4),
            },
            "encoding_surprise": chunk.metadata.get("encoding_surprise"),
            "top_associations": [
                {"id": aid, "strength": round(s, 3)}
                for aid, s in sorted(
                    chunk.associations.items(), key=lambda x: -x[1]
                )[:5]
            ],
        }
        return {
            "content": [{
                "type": "text",
                "text": json.dumps(report, indent=2, ensure_ascii=False),
            }]
        }

    def _history(self, args: dict) -> dict:
        """双时态取代链：一个事实的完整版本历史"""
        memory_id = args.get("memory_id", "")
        chunk, _layer = self._find_chunk(memory_id)
        if chunk is None:
            return {
                "content": [{"type": "text", "text": f"Memory not found: {memory_id}"}],
                "isError": True,
            }

        # 走到链头（最新版本）
        head = chunk
        seen = {head.id}
        while head.metadata.get("superseded_by"):
            nxt, _ = self._find_chunk(head.metadata["superseded_by"])
            if nxt is None or nxt.id in seen:
                break
            head = nxt
            seen.add(head.id)

        # 从链头沿 parent_id 回溯全部历史
        def _ts(value):
            return time.strftime("%Y-%m-%d %H:%M", time.localtime(value)) if value else "?"

        lines = []
        node = head
        seen = set()
        while node is not None and node.id not in seen:
            seen.add(node.id)
            if node.invalid_at is None:
                status = "CURRENT"
                period = f"since {_ts(node.valid_at or node.created_at)}"
            else:
                status = "superseded"
                period = f"{_ts(node.valid_at or node.created_at)} -> {_ts(node.invalid_at)}"
            lines.append(f"[{status}] ({period}) [{node.id}] {node.content[:120]}")
            node = self._find_chunk(node.parent_id)[0] if node.parent_id else None

        if len(lines) == 1:
            lines.append("(no supersession history — this fact was never updated)")
        return {"content": [{"type": "text", "text": "\n".join(lines)}]}

    def _sleep(self) -> dict:
        """执行一次睡眠巩固，返回审计报告"""
        report = self.memory.sleep()
        lines = [f"Sleep cycle done: {report.summary()}"]
        for detail in report.details:
            lines.append(
                f"\ngist {detail['gist_id']} <- {detail['cluster_size']} episodes "
                f"({', '.join(detail['source_ids'])})"
            )
        if not report.gists_created:
            lines.append("(no clusters large enough to consolidate)")
        return {"content": [{"type": "text", "text": "\n".join(lines)}]}

    def _focus(self, args: dict) -> dict:
        """注意力工作区：现在该想什么"""
        query = args.get("query", "")
        workspace = self.memory.focus(query)
        return {
            "content": [{"type": "text", "text": workspace.to_prompt_context()}]
        }

    def _feedback(self, args: dict) -> dict:
        """回忆反馈：确认/纠正上一次检索"""
        query = args.get("query", "")
        accepted = bool(args.get("accepted", True))
        self.memory.retrieval.feedback(query, accepted=accepted)
        verdict = "confirmed (weights raised)" if accepted else "corrected (weights lowered)"
        return {
            "content": [{"type": "text", "text": f"Feedback recorded: {verdict}."}]
        }

    def _maintain(self) -> dict:
        """强制维护：睡眠巩固（如到期）+ 降级 + 清理"""
        core_before = len(self.memory.core)
        forgotten_before = len(self.memory.forgotten)
        self.memory.maintain()
        core_after = len(self.memory.core)
        forgotten_after = len(self.memory.forgotten)
        return {
            "content": [{
                "type": "text",
                "text": (
                    f"Maintenance done. core: {core_before} -> {core_after}, "
                    f"forgotten: {forgotten_before} -> {forgotten_after}"
                ),
            }]
        }

    def _add(self, args: dict) -> dict:
        """添加记忆（支持 LLM 增强抽取与记忆类型）"""
        content = args.get("content", "")
        importance = args.get("importance", 0.5)
        topics = args.get("topics", [])
        memory_type_str = args.get("memory_type", "interaction")

        if memory_type_str != "interaction":
            # 事实/偏好/故事/想法：抽取锚点后走 add_memory ——
            # 这样才能进入取代决策表（FACT/PREFERENCE 的双时态管理）
            # 并按类型使用正确的衰减速率。狗粮期实测：MCP 存的
            # 事实类知识全按最快衰减的 interaction 处理是真实缺陷。
            from core.entity_extractor import EntityExtractor
            from core.weight_system import MemoryType

            extractor = EntityExtractor()
            persons = extractor.extract_persons(content)
            location = extractor.extract_location(content)
            t_abs, t_rel, t_ctx = extractor.extract_time(content)
            keywords = extractor.extract_keywords(content)
            merged_topics = extractor.extract_topics(content) | set(topics)

            memory_id = self.memory.add_memory(
                content=content,
                memory_type=MemoryType(memory_type_str),
                persons=list(persons),
                location=location,
                time_absolute=t_abs,
                time_relative=t_rel,
                time_context=t_ctx,
                topics=list(merged_topics),
                keywords=list(keywords),
                importance=importance,
                source="system_extract",
            )
            return {
                "content": [{
                    "type": "text",
                    "text": f"Memory saved: {memory_id} (type={memory_type_str})",
                }]
            }

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
