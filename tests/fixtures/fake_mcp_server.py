"""测试用轻量级假 MCP stdio 服务端（覆盖翻页、错误、非 JSON 噪声与慢响应）。"""

from __future__ import annotations

import json
import sys
import time


TOOLS_PAGE_1 = [
    {
        "name": "echo",
        "description": "Echo text back to caller.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
]

TOOLS_PAGE_2 = [
    {
        "name": "add_numbers",
        "description": "Add two numbers.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["a", "b"],
        },
    },
    {
        "name": "slow_op",
        "description": "Sleep before replying to test client timeout.",
        "inputSchema": {
            "type": "object",
            "properties": {"seconds": {"type": "number"}},
        },
    },
]


def _write(obj: dict) -> None:
    sys.stdout.buffer.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
    sys.stdout.buffer.flush()


def main() -> int:
    # 故意向 stdout 打一行非 JSON 横幅、向 stderr 打诊断信息，验证客户端容错
    sys.stdout.buffer.write(b"[fake-mcp] booting diagnostics line\n")
    sys.stdout.buffer.flush()
    sys.stderr.buffer.write(b"[fake-mcp] stderr ready\n")
    sys.stderr.buffer.flush()

    while True:
        raw = sys.stdin.buffer.readline()
        if not raw:
            break
        line = raw.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        req = json.loads(line)
        method = req.get("method", "")
        req_id = req.get("id")
        params = req.get("params") or {}

        if req_id is None:
            # 通知（例如 notifications/initialized）不返回响应
            continue

        if method == "initialize":
            _write({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "fake-mcp", "version": "1.0.0"},
                },
            })
            # 夹带一条服务端通知，客户端应忽略而不串号
            _write({"jsonrpc": "2.0", "method": "notifications/message", "params": {"ok": True}})
        elif method == "tools/list":
            cursor = params.get("cursor")
            if not cursor:
                _write({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"tools": TOOLS_PAGE_1, "nextCursor": "page2"},
                })
            else:
                _write({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"tools": TOOLS_PAGE_2},
                })
        elif method == "tools/call":
            name = params.get("name")
            args = params.get("arguments") or {}
            if name == "echo":
                _write({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": f"ECHO:{args.get('text', '')}"}],
                        "isError": False,
                    },
                })
            elif name == "add_numbers":
                total = float(args.get("a", 0)) + float(args.get("b", 0))
                _write({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": str(int(total) if total.is_integer() else total)}],
                        "isError": False,
                    },
                })
            elif name == "slow_op":
                time.sleep(float(args.get("seconds", 2.0)))
                _write({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"content": [{"type": "text", "text": "done"}], "isError": False},
                })
            else:
                _write({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": f"Unknown tool: {name}"}],
                        "isError": True,
                    },
                })
        else:
            _write({
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            })

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
