from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import threading

from new_agent_memory.cli import main


def test_cli_ask_runs_one_turn_against_openai_compatible_endpoint(tmp_path, capsys):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["Content-Length"])
            self.rfile.read(length)
            response = {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "name": "respond",
                                    "arguments": {"message": "cli agent online"},
                                    "rationale": "cli smoke test",
                                }
                            )
                        }
                    }
                ]
            }
            body = json.dumps(response).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        code = main(
            [
                "ask",
                "--data-dir",
                str(tmp_path / "memory_data"),
                "--env-file",
                "",
                "--api-key",
                "test",
                "--base-url",
                f"http://127.0.0.1:{server.server_port}/v1",
                "--model",
                "fake-chat-model",
                "hello",
            ]
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)

    captured = capsys.readouterr()
    assert code == 0
    assert "cli agent online" in captured.out
