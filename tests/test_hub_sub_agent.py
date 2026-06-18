"""Tests for the Hub Sub-Agent task identification logic."""

import sys
sys.path.insert(0, ".")

from adapters.hub_sub_agent import is_task_message, process_message


def test_task_detection():
    """Test that task messages are correctly identified."""
    # Task messages with keywords
    assert is_task_message("帮我写一个函数") == True
    assert is_task_message("执行这个命令") == True
    assert is_task_message("做一个测试") == True
    assert is_task_message("写一个文档") == True
    assert is_task_message("改一下这个bug") == True
    assert is_task_message("查一下日志") == True
    assert is_task_message("分析这个问题") == True
    assert is_task_message("重构这段代码") == True
    assert is_task_message("部署到生产环境") == True

    # Task messages with @Claude Code
    assert is_task_message("@Claude Code 帮我看看") == True
    assert is_task_message("@claudecode 做个事情") == True
    assert is_task_message("@claude-code 执行任务") == True

    # Non-task messages (chat)
    assert is_task_message("你好") == False
    assert is_task_message("今天天气怎么样？") == False
    assert is_task_message("谢谢") == False
    assert is_task_message("好的") == False


def test_process_message():
    """Test message processing returns correct replies."""
    # Task message
    task_msg = {"content": "帮我写一个Python脚本", "sender_name": "user1"}
    reply = process_message(task_msg)
    assert reply == "收到，我来处理这个任务 🫡"

    # Chat message
    chat_msg = {"content": "你好啊", "sender_name": "user2"}
    reply = process_message(chat_msg)
    assert reply is not None
    assert len(reply) > 0


if __name__ == "__main__":
    test_task_detection()
    test_process_message()
    print("All tests passed!")
