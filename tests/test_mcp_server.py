"""
MCP Server 回归测试

覆盖：
- stdio 协议纯净性：首个请求之前 stdout 必须无任何输出
- initialize / tools/list / tools/call 的 JSON-RPC 响应格式与 id 关联
- 未知方法返回顶层 -32601 错误；未知工具返回 -32602；非法 JSON 返回 -32700
- 工具执行失败返回 result.content + isError=true（MCP 规范）
- 本地 LLM 端点探测（不可达时禁用，走规则抽取）
- memory_add 的 topics 写回核心层倒排索引
- 显式 importance 优先于 LLM 推断值
- 数据目录：MEMORY_DATA_DIR 环境变量优先，默认锚定模块目录而非 cwd
"""

import json
import os
import select
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER_PATH = os.path.join(REPO_ROOT, "mcp_server.py")

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import mcp_server  # noqa: E402

# 不可达端点：端口 9（discard）通常关闭，连接立即被拒绝，探测快速失败
UNREACHABLE_LLM = "http://127.0.0.1:9/v1"


# ============ 子进程工具 ============


class LineReader:
    """基于 fd + select 的行读取器，带超时，避免缓冲导致的阻塞/漏读"""

    def __init__(self, fileobj):
        self.fd = fileobj.fileno()
        self.buf = b""

    def read_line(self, timeout):
        """读取一行（不含换行符）；超时或 EOF 返回 None"""
        deadline = time.time() + timeout
        while b"\n" not in self.buf:
            remaining = deadline - time.time()
            if remaining <= 0:
                return None
            ready, _, _ = select.select([self.fd], [], [], remaining)
            if not ready:
                return None
            chunk = os.read(self.fd, 65536)
            if not chunk:
                return None
            self.buf += chunk
        line, self.buf = self.buf.split(b"\n", 1)
        return line.decode("utf-8")


@pytest.fixture
def start_server(tmp_path):
    """启动 mcp_server 子进程；cwd 指向临时目录以验证数据目录不依赖 cwd"""
    started = []

    def _start():
        env = dict(os.environ)
        env["MEMORY_DATA_DIR"] = str(tmp_path / "memdata")
        env["LOCAL_LLM_API_BASE"] = UNREACHABLE_LLM
        stderr_file = open(tmp_path / "stderr.log", "ab")
        proc = subprocess.Popen(
            [sys.executable, SERVER_PATH],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr_file,
            env=env,
            cwd=str(tmp_path),
        )
        started.append((proc, stderr_file))
        return proc, LineReader(proc.stdout)

    yield _start

    for proc, stderr_file in started:
        proc.kill()
        proc.wait(timeout=5)
        stderr_file.close()


def send(proc, obj):
    proc.stdin.write((json.dumps(obj) + "\n").encode("utf-8"))
    proc.stdin.flush()


def send_raw(proc, text):
    proc.stdin.write((text + "\n").encode("utf-8"))
    proc.stdin.flush()


def read_response(reader, expect_id, timeout=15.0):
    """读取一条响应，校验 JSON-RPC 格式与 id 关联"""
    line = reader.read_line(timeout)
    assert line is not None, "等待服务器响应超时"
    resp = json.loads(line)
    assert resp.get("jsonrpc") == "2.0"
    assert resp.get("id") == expect_id
    return resp


# ============ stdio 协议纯净性 ============


def test_stdout_silent_until_first_request(start_server):
    """首个请求之前 stdout 必须无任何输出；initialize 后恰好一条匹配 id 的响应"""
    proc, reader = start_server()

    # 启动后等待 3 秒：不允许有未经请求的输出（诊断信息、id=0 响应等）
    early = reader.read_line(3.0)
    assert early is None, f"首个请求前 stdout 出现输出: {early!r}"

    send(proc, {
        "jsonrpc": "2.0", "id": 11, "method": "initialize",
        "params": {"protocolVersion": "2024-11-05", "capabilities": {}},
    })
    resp = read_response(reader, expect_id=11)
    assert "error" not in resp
    assert resp["result"]["protocolVersion"]
    assert resp["result"]["serverInfo"]["name"]

    # initialize 之后也不应有多余输出
    extra = reader.read_line(0.5)
    assert extra is None, f"initialize 后出现多余输出: {extra!r}"


def test_full_session_all_lines_are_jsonrpc(start_server, tmp_path):
    """完整会话：每条 stdout 都是 id 正确关联的 JSON-RPC 响应"""
    proc, reader = start_server()

    send(proc, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    read_response(reader, expect_id=1)

    # notification 不应有响应
    send(proc, {"jsonrpc": "2.0", "method": "notifications/initialized"})

    send(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    resp = read_response(reader, expect_id=2)
    names = {t["name"] for t in resp["result"]["tools"]}
    assert {"memory_search", "memory_add", "memory_stats"} <= names

    send(proc, {
        "jsonrpc": "2.0", "id": 3, "method": "tools/call",
        "params": {
            "name": "memory_add",
            "arguments": {
                "content": "Discussed the zebra migration project with the team",
                "importance": 0.9,
                "topics": ["zebra"],
            },
        },
    })
    resp = read_response(reader, expect_id=3)
    assert "Memory saved" in resp["result"]["content"][0]["text"]
    assert not resp["result"].get("isError")

    send(proc, {
        "jsonrpc": "2.0", "id": 4, "method": "tools/call",
        "params": {"name": "memory_search", "arguments": {"query": "zebra"}},
    })
    resp = read_response(reader, expect_id=4)
    assert "content" in resp["result"]

    # 标准探活类请求需要优雅响应
    send(proc, {"jsonrpc": "2.0", "id": 5, "method": "ping"})
    resp = read_response(reader, expect_id=5)
    assert resp["result"] == {}

    send(proc, {"jsonrpc": "2.0", "id": 6, "method": "resources/list"})
    resp = read_response(reader, expect_id=6)
    assert resp["result"]["resources"] == []

    send(proc, {"jsonrpc": "2.0", "id": 7, "method": "prompts/list"})
    resp = read_response(reader, expect_id=7)
    assert resp["result"]["prompts"] == []

    # 数据目录写入 MEMORY_DATA_DIR，而不是 cwd 下的 ./memory_data
    assert (tmp_path / "memdata").is_dir()
    assert not (tmp_path / "memory_data").exists()


def test_protocol_errors(start_server):
    """未知方法 -32601；未知工具 -32602；非法 JSON -32700（id 为 null）"""
    proc, reader = start_server()

    send(proc, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    read_response(reader, expect_id=1)

    # 未知方法：顶层 error，不是包在 result 里
    send(proc, {"jsonrpc": "2.0", "id": 42, "method": "bogus/method"})
    resp = read_response(reader, expect_id=42)
    assert "result" not in resp
    assert resp["error"]["code"] == -32601

    # 未知工具
    send(proc, {
        "jsonrpc": "2.0", "id": 43, "method": "tools/call",
        "params": {"name": "nonexistent_tool", "arguments": {}},
    })
    resp = read_response(reader, expect_id=43)
    assert "result" not in resp
    assert resp["error"]["code"] == -32602

    # 非法 JSON：-32700，id 为 null
    send_raw(proc, "this is not json {{{")
    line = reader.read_line(10.0)
    assert line is not None
    resp = json.loads(line)
    assert resp["id"] is None
    assert resp["error"]["code"] == -32700

    # 之后服务器仍然可用
    send(proc, {"jsonrpc": "2.0", "id": 44, "method": "ping"})
    resp = read_response(reader, expect_id=44)
    assert resp["result"] == {}


# ============ 本地 LLM 端点探测 ============


def test_create_llm_fn_unreachable_returns_none(monkeypatch):
    """端点不可达时 create_llm_fn 返回 None（下游走规则抽取）"""
    monkeypatch.setenv("LOCAL_LLM_API_BASE", UNREACHABLE_LLM)
    assert mcp_server.create_llm_fn() is None


def test_create_llm_fn_reachable_returns_callable(monkeypatch):
    """端点可达时 create_llm_fn 返回可调用对象"""

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = b'{"data": []}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        port = httpd.server_address[1]
        monkeypatch.setenv("LOCAL_LLM_API_BASE", f"http://127.0.0.1:{port}/v1")
        fn = mcp_server.create_llm_fn()
        assert callable(fn)
    finally:
        httpd.shutdown()
        httpd.server_close()


# ============ 进程内行为测试 ============


def _make_server(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORY_DATA_DIR", str(tmp_path / "memdata"))
    monkeypatch.setenv("LOCAL_LLM_API_BASE", UNREACHABLE_LLM)
    return mcp_server.MemoryMCPServer()


def _extract_memory_id(result):
    text = result["content"][0]["text"]
    assert "Memory saved: " in text
    return text.split("Memory saved: ", 1)[1].strip()


def test_memory_add_topics_update_inverted_index(monkeypatch, tmp_path):
    """用户指定的 topics 必须进入核心层倒排索引，否则按主题检索会漏掉"""
    server = _make_server(monkeypatch, tmp_path)
    result = server._call_tool({
        "name": "memory_add",
        "arguments": {
            "content": "planned the quarterly budget with the finance team",
            "topics": ["custom_zebra_topic"],
        },
    })
    memory_id = _extract_memory_id(result)

    chunk = server.memory.core.get(memory_id)
    assert chunk is not None
    assert "custom_zebra_topic" in chunk.topics
    # 关键断言：倒排索引里能找到该记忆（修复前被 _store.put 绕过）
    assert memory_id in server.memory.core.topic_index.get("custom_zebra_topic", set())


def test_explicit_importance_overrides_llm(monkeypatch, tmp_path):
    """显式传入的 importance 优先于 LLM 抽取推断的值"""
    server = _make_server(monkeypatch, tmp_path)

    # 伪造 LLM：返回 importance=0.2，试图覆盖用户显式指定的 0.9
    def fake_llm(prompt):
        return json.dumps({
            "persons": [],
            "location": None,
            "time": None,
            "topics": ["fake_topic"],
            "keywords": ["purple", "elephant", "memo"],
            "emotion_valence": 0.0,
            "emotion_intensity": 0.0,
            "importance": 0.2,
        })

    server.memory.llm_fn = fake_llm

    result = server._call_tool({
        "name": "memory_add",
        "arguments": {"content": "unique purple elephant memo", "importance": 0.9},
    })
    memory_id = _extract_memory_id(result)
    chunk = server.memory.core.get(memory_id)
    assert chunk.importance == pytest.approx(0.9)


def test_tool_failure_returns_is_error(monkeypatch, tmp_path):
    """工具执行失败：result.content + isError=true，而不是协议错误"""
    server = _make_server(monkeypatch, tmp_path)

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(server.memory, "retrieve", boom)

    resp = server.handle_request({
        "jsonrpc": "2.0", "id": 7, "method": "tools/call",
        "params": {"name": "memory_search", "arguments": {"query": "x"}},
    })
    assert resp["id"] == 7
    assert "error" not in resp
    assert resp["result"]["isError"] is True
    assert "boom" in resp["result"]["content"][0]["text"]


def test_data_dir_env_override(monkeypatch, tmp_path):
    """MEMORY_DATA_DIR 环境变量优先"""
    server = _make_server(monkeypatch, tmp_path)
    assert server.memory.data_dir == str(tmp_path / "memdata")


def test_data_dir_default_anchored_to_module(monkeypatch, tmp_path):
    """未设置环境变量时，默认目录锚定模块所在目录，而不是 cwd"""
    monkeypatch.delenv("MEMORY_DATA_DIR", raising=False)
    monkeypatch.setenv("LOCAL_LLM_API_BASE", UNREACHABLE_LLM)

    module_dir = tmp_path / "moduledir"
    module_dir.mkdir()
    cwd_dir = tmp_path / "cwd"
    cwd_dir.mkdir()

    monkeypatch.setattr(mcp_server, "__file__", str(module_dir / "mcp_server.py"))
    monkeypatch.chdir(cwd_dir)

    server = mcp_server.MemoryMCPServer()
    assert os.path.abspath(server.memory.data_dir) == str(module_dir / "memory_data")
    assert not (cwd_dir / "memory_data").exists()
