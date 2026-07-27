from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import threading
from types import SimpleNamespace

from new_agent_memory.cli import main
import new_agent_memory.cli as cli


class RecordingRunner:
    """Fake agent runner that records every message routed to the agent."""

    def __init__(self):
        self.messages = []

    def run_turn(self, message):
        self.messages.append(message)
        return SimpleNamespace(result=SimpleNamespace(output=f"echo:{message}", success=True))


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


# ============ --fresh must not overwrite on-disk memory ============


def _seed_data_dir(data_dir):
    seeded = cli.HumanLikeMemorySystem(data_dir=str(data_dir))
    seeded.add_memory(content="seeded precious memory", importance=0.9)
    seeded.save()
    text = (data_dir / "core.json").read_text(encoding="utf-8")
    assert "seeded precious memory" in text


def test_cli_fresh_ask_does_not_overwrite_existing_memory(tmp_path, monkeypatch, capsys):
    data_dir = tmp_path / "memory_data"
    _seed_data_dir(data_dir)

    monkeypatch.setattr(
        cli.HumanLikeMemorySystem,
        "create_openai_agent",
        lambda self, **kwargs: RecordingRunner(),
    )
    code = main(["ask", "--data-dir", str(data_dir), "--fresh", "hello"])

    captured = capsys.readouterr()
    assert code == 0
    # save is suppressed and the user is told why on stderr
    assert "--fresh" in captured.err
    assert "seeded precious memory" in (data_dir / "core.json").read_text(encoding="utf-8")


def test_cli_fresh_ask_with_explicit_save_flag_persists_fresh_state(tmp_path, monkeypatch, capsys):
    data_dir = tmp_path / "memory_data"
    _seed_data_dir(data_dir)

    monkeypatch.setattr(
        cli.HumanLikeMemorySystem,
        "create_openai_agent",
        lambda self, **kwargs: RecordingRunner(),
    )
    code = main(["ask", "--data-dir", str(data_dir), "--fresh", "--save", "hello"])

    captured = capsys.readouterr()
    assert code == 0
    # explicit opt-in: fresh state overwrites disk, no suppression notice
    assert "--fresh run not saved" not in captured.err
    assert "seeded precious memory" not in (data_dir / "core.json").read_text(encoding="utf-8")


# ============ chat REPL command handling ============


def _make_fake_memory(runner):
    class FakeMemory:
        def __init__(self):
            self.save_calls = 0
            self.maintain_calls = 0

        def load(self):
            return False

        def save(self):
            self.save_calls += 1

        def start_goal(self, goal):
            return None

        def create_openai_agent(self, **kwargs):
            return runner

        def get_cognitive_summary(self):
            return {"ok": True}

        def auto_maintain_if_needed(self):
            # 6 小时限速的维护入口：chat 每轮调用（限速在真实实现里）
            self.maintain_calls += 1

    return FakeMemory()


def test_cli_chat_bare_save_and_summary_go_to_agent(monkeypatch, capsys):
    runner = RecordingRunner()
    fake_memory = _make_fake_memory(runner)
    monkeypatch.setattr(cli, "HumanLikeMemorySystem", lambda data_dir: fake_memory)
    inputs = iter(["save", "summary", "quit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

    code = main(["chat", "--no-save"])

    captured = capsys.readouterr()
    assert code == 0
    # bare words reach the agent instead of being hijacked as commands
    assert runner.messages == ["save", "summary"]
    assert fake_memory.save_calls == 0
    assert '"ok": true' not in captured.out
    assert "echo:save" in captured.out
    assert "echo:summary" in captured.out
    # 每个对话轮后调用限速维护入口（睡眠/衰减在长聊里真实可达）
    assert fake_memory.maintain_calls == 2
    assert "maintain skipped" not in captured.out


def test_cli_chat_colon_commands_and_bare_quit_still_work(monkeypatch, capsys):
    runner = RecordingRunner()
    fake_memory = _make_fake_memory(runner)
    monkeypatch.setattr(cli, "HumanLikeMemorySystem", lambda data_dir: fake_memory)
    inputs = iter([":save", ":summary", "quit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

    code = main(["chat", "--no-save"])

    captured = capsys.readouterr()
    assert code == 0
    assert runner.messages == []
    assert fake_memory.save_calls == 1
    assert "saved" in captured.out
    assert '"ok": true' in captured.out


# ============ add / search / stats subcommands (no LLM required) ============


def test_cli_add_prints_chunk_id_and_persists(tmp_path, capsys):
    data_dir = str(tmp_path / "mem")
    code = main(
        [
            "add",
            "duck dinner with the client in Beijing",
            "--data-dir",
            data_dir,
            "--type",
            "story",
            "--topics",
            "food,travel",
            "--keywords",
            "duck,dinner",
            "--importance",
            "0.8",
            "--location",
            "Beijing",
            "--persons",
            "client",
        ]
    )

    chunk_id = capsys.readouterr().out.strip()
    assert code == 0
    assert chunk_id
    text = (tmp_path / "mem" / "core.json").read_text(encoding="utf-8")
    assert "duck dinner with the client in Beijing" in text
    assert chunk_id in text


def test_cli_search_prints_status_line_and_chunk_contents(tmp_path, capsys):
    data_dir = str(tmp_path / "mem")
    main(["add", "duck dinner with the client in Beijing", "--data-dir", data_dir, "--keywords", "duck,dinner"])
    capsys.readouterr()

    code = main(["search", "duck dinner", "--data-dir", data_dir])

    out = capsys.readouterr().out
    assert code == 0
    lines = out.strip().splitlines()
    assert lines[0].startswith("status=")
    assert "path=" in lines[0]
    assert "confidence=" in lines[0]
    assert any("duck dinner with the client in Beijing" in line for line in lines[1:])


def test_cli_search_limit_caps_printed_chunks(tmp_path, capsys):
    data_dir = str(tmp_path / "mem")
    main(["add", "duck dinner first note", "--data-dir", data_dir, "--keywords", "duck"])
    main(["add", "duck dinner second note", "--data-dir", data_dir, "--keywords", "duck"])
    capsys.readouterr()

    code = main(["search", "duck dinner", "--data-dir", data_dir, "--limit", "1"])

    out = capsys.readouterr().out
    assert code == 0
    lines = out.strip().splitlines()
    assert lines[0].startswith("status=")
    assert len(lines[1:]) <= 1


def test_cli_stats_prints_counts_and_backend(tmp_path, capsys):
    data_dir = str(tmp_path / "mem")
    main(["add", "duck dinner with the client in Beijing", "--data-dir", data_dir])
    capsys.readouterr()

    code = main(["stats", "--data-dir", data_dir])

    out = capsys.readouterr().out
    assert code == 0
    payload = json.loads(out)
    assert payload["core_chunks"] == 1
    assert payload["forgotten_chunks"] == 0
    assert payload["backend"] == "json"
    assert payload["data_dir"] == data_dir
    assert payload["attention_goals"] == 0
    assert payload["attention_procedures"] == 0
