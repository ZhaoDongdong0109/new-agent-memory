"""Command line entry points for new-agent-memory."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict

from main import HumanLikeMemorySystem


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="new-agent-memory",
        description="Run a memory-driven agent with an OpenAI-compatible API.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask = subparsers.add_parser("ask", help="Run one agent turn.")
    _add_common_args(ask)
    ask.add_argument("message", help="User message for the agent.")
    ask.set_defaults(func=run_ask)

    chat = subparsers.add_parser("chat", help="Start an interactive agent session.")
    _add_common_args(chat)
    chat.set_defaults(func=run_chat)

    return parser


def _add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument("--data-dir", default="./memory_data", help="Directory for memory JSON files.")
    parser.add_argument("--env-file", default=".env", help="Optional .env file with OpenAI-compatible settings.")
    parser.add_argument("--name", default="openai-compatible-agent", help="Agent name.")
    parser.add_argument("--goal", default="", help="Optional active goal to push before running.")
    parser.add_argument("--fresh", action="store_true", help="Do not load existing memory before running.")
    parser.add_argument("--no-save", action="store_true", help="Do not save memory after running.")
    parser.add_argument("--show-action", action="store_true", help="Print selected action metadata.")
    parser.add_argument("--show-summary", action="store_true", help="Print cognitive summary after the turn.")
    parser.add_argument("--runtime", action="store_true", help="Use the multi-step CognitiveRuntime loop.")
    parser.add_argument("--max-steps", type=int, default=4, help="Maximum CognitiveRuntime steps.")

    parser.add_argument("--api-key", default=None, help="Override OPENAI_API_KEY.")
    parser.add_argument("--base-url", default=None, help="Override OPENAI_BASE_URL, e.g. http://localhost:1234/v1.")
    parser.add_argument("--model", default=None, help="Override OPENAI_MODEL.")
    parser.add_argument("--temperature", type=float, default=None, help="Override model temperature.")
    parser.add_argument("--timeout", type=float, default=None, help="Override request timeout seconds.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Override max_tokens.")


def run_ask(args: argparse.Namespace) -> int:
    memory, runner = _build_memory_and_runner(args)
    if args.runtime:
        run = runner.run(args.message)
        _print_run(run, args, memory)
        ok = run.result.success
    else:
        episode = runner.run_turn(args.message)
        _print_episode(episode, args, memory)
        ok = episode.result.success
    _save_if_needed(memory, args)
    return 0 if ok else 1


def run_chat(args: argparse.Namespace) -> int:
    memory, runner = _build_memory_and_runner(args)
    print("new-agent-memory chat. Type :quit to exit, :save to persist, :summary to inspect state.")

    while True:
        try:
            message = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not message:
            continue
        command = message.lower()
        if command in {":q", ":quit", ":exit", "q", "quit", "exit"}:
            break
        if command in {":save", "save"}:
            memory.save()
            print("saved")
            continue
        if command in {":summary", "summary"}:
            print(json.dumps(memory.get_cognitive_summary(), ensure_ascii=False, indent=2))
            continue

        if args.runtime:
            run = runner.run(message)
            _print_run(run, args, memory)
        else:
            episode = runner.run_turn(message)
            _print_episode(episode, args, memory)

    _save_if_needed(memory, args)
    return 0


def _build_memory_and_runner(args: argparse.Namespace):
    memory = HumanLikeMemorySystem(data_dir=args.data_dir)
    if not args.fresh:
        memory.load()
    if args.goal:
        memory.start_goal(args.goal)

    if args.runtime:
        runner = memory.create_openai_runtime(
            name=args.name,
            env_file=args.env_file,
            max_steps=args.max_steps,
            **_api_overrides(args),
        )
    else:
        runner = memory.create_openai_agent(
            name=args.name,
            env_file=args.env_file,
            **_api_overrides(args),
        )
    return memory, runner


def _api_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "api_key": args.api_key,
        "base_url": args.base_url,
        "model": args.model,
        "temperature": args.temperature,
        "timeout": args.timeout,
        "max_tokens": args.max_tokens,
    }


def _print_episode(episode: Any, args: argparse.Namespace, memory: HumanLikeMemorySystem):
    if episode.result.output:
        print(episode.result.output)
    if args.show_action:
        print(
            json.dumps(
                {
                    "action": episode.action.to_dict(),
                    "result": episode.result.to_dict(),
                    "reward": episode.reward,
                    "lesson": episode.lesson,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    if args.show_summary:
        print(json.dumps(memory.get_cognitive_summary(), ensure_ascii=False, indent=2))


def _print_run(run: Any, args: argparse.Namespace, memory: HumanLikeMemorySystem):
    if run.result.output:
        print(run.result.output)
    if args.show_action:
        print(
            json.dumps(
                {
                    "runtime": {
                        "id": run.id,
                        "goal": run.goal,
                        "completed": run.completed,
                        "stop_reason": run.stop_reason,
                    },
                    "result": run.result.to_dict(),
                    "steps": [
                        {
                            "index": step.index,
                            "action": step.action.to_dict(),
                            "result": step.result.to_dict(),
                            "reward": step.reward,
                        }
                        for step in run.steps
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    if args.show_summary:
        print(json.dumps(memory.get_cognitive_summary(), ensure_ascii=False, indent=2))


def _save_if_needed(memory: HumanLikeMemorySystem, args: argparse.Namespace):
    if not args.no_save:
        memory.save()


def _configure_stdio():
    """Keep Windows terminals from crashing on model output outside GBK."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def main(argv: Any = None) -> int:
    _configure_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
