"""CLI entry point for the job_hunt agent with HITL interrupt handling.

When the agent tries to use a tool that requires approval, execution pauses
and the user is prompted in the terminal to approve or reject with a reason.
"""

import sqlite3
import json
import sys
from collections.abc import Iterable

from langgraph.types import Command

from job_hunt.paths import CHECKPOINT_DB_PATH, WORKSPACE_ROOT

DEFAULT_THREAD_ID = 'cli-session'
MAX_TOOL_OUTPUT_CHARS = 800


def _format_action(action: dict) -> str:
    """Format an action request for display."""
    return (
        f"  工具：{action['name']}\n"
        f"  参数：{action.get('args', {})}"
    )


def _handle_interrupt(interrupts: list) -> dict:
    """Ask the user to decide on each pending tool call.

    Returns a resume dict suitable for Command(resume=...).
    """
    decisions = []

    for interrupt in interrupts:
        value = interrupt.value
        for action in value['action_requests']:
            name = action['name']

            print(f"\n{'=' * 50}")
            print('是否批准这次工具调用？')
            print(_format_action(action))
            print(f"{'=' * 50}")

            while True:
                choice = input('  [a]批准  [r]拒绝：').strip().lower()

                if choice in ('a', 'approve', 'y', 'yes', '批准', '同意'):
                    decisions.append({'type': 'approve'})
                    break
                elif choice in ('r', 'reject', 'n', 'no', '拒绝', '否'):
                    reason = input(
                        '  拒绝原因，希望 Agent 该怎么做：'
                    ).strip()
                    message = (
                        f'用户拒绝调用工具 `{name}`。\n'
                        f'原因：{reason}'
                    )
                    decisions.append({'type': 'reject', 'message': message})
                    break
                else:
                    print("  无效输入，请输入 'a' 或 'r'。")

    return {'decisions': decisions}


def _print_error(message: str, error: Exception) -> None:
    """Print a concise error without exposing provider details or a traceback."""
    print(f'\n[错误] {message}（{type(error).__name__}）\n')


def _configure_output_encoding() -> None:
    """Use UTF-8 output so decorative terminal text does not crash on Windows."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')


def _print_help() -> None:
    """Print available CLI commands."""
    print(
        '\n可用命令：\n'
        '  /help   查看命令说明\n'
        '  /exit   退出程序\n'
        '  /new    开始新的会话\n'
        '  /debug  开启或关闭工具调用显示\n'
    )


def _shorten(value: object, max_chars: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    """Return a readable, bounded preview for terminal output."""
    text = str(value)
    if len(text) <= max_chars:
        return text
    return f'{text[:max_chars]}\n...[已截断，原长度 {len(text)} 字符]'


def _message_text(message: object) -> str:
    """Extract plain text from a streamed message chunk."""
    content = getattr(message, 'content', '')
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get('type') in ('text', 'text_delta'):
                parts.append(block.get('text', ''))
        return ''.join(parts)
    return ''


def _iter_update_messages(update: object) -> Iterable[object]:
    """Yield messages from a LangGraph updates payload."""
    if not isinstance(update, dict):
        return

    for value in update.values():
        if isinstance(value, dict):
            messages = value.get('messages', ())
            if isinstance(messages, list):
                yield from messages


def _print_tool_call(name: str, args: object) -> None:
    """Print a compact tool-call trace."""
    try:
        args_text = json.dumps(args, ensure_ascii=False)
    except TypeError:
        args_text = str(args)
    print(f'\n[工具调用] {name}')
    print(f'参数：{_shorten(args_text)}')


def _print_tool_result(message: object) -> None:
    """Print a compact tool-result trace."""
    name = getattr(message, 'name', None) or 'unknown'
    print(f'\n[工具结果] {name}')
    print(_shorten(_message_text(message)))


def _print_tool_updates(update: object) -> None:
    """Print tool calls and results from one stream update."""
    for message in _iter_update_messages(update):
        for tool_call in getattr(message, 'tool_calls', ()) or ():
            _print_tool_call(tool_call.get('name', 'unknown'), tool_call.get('args', {}))

        if getattr(message, 'type', None) == 'tool':
            _print_tool_result(message)


def _has_checkpoint_history(thread_id: str) -> bool:
    """Return whether a previous checkpoint exists for this CLI thread."""
    if not CHECKPOINT_DB_PATH.exists():
        return False

    try:
        conn = sqlite3.connect(f'file:{CHECKPOINT_DB_PATH}?mode=ro', uri=True)
        with conn:
            row = conn.execute(
                'SELECT 1 FROM checkpoints WHERE thread_id = ? LIMIT 1',
                (thread_id,),
            ).fetchone()
    except sqlite3.Error:
        return False
    finally:
        if 'conn' in locals():
            conn.close()

    return row is not None


def _run_agent(agent, request: object, config: dict, show_tools: bool) -> dict:
    """Run the agent with streamed assistant output and optional tool traces."""
    stream_modes = ['messages', 'values']
    if show_tools:
        stream_modes.append('updates')

    final_state = None
    streamed_answer = False

    for mode, payload in agent.stream(
        request,
        config=config,
        stream_mode=stream_modes,
    ):
        if mode == 'messages':
            message, metadata = payload
            if metadata.get('langgraph_node') != 'model':
                continue
            if getattr(message, 'tool_calls', None):
                continue

            text = _message_text(message)
            if text:
                if not streamed_answer:
                    print()
                print(text, end='', flush=True)
                streamed_answer = True
        elif mode == 'updates' and show_tools:
            _print_tool_updates(payload)
        elif mode == 'values':
            final_state = payload

    if streamed_answer:
        print('\n')
    elif isinstance(final_state, dict) and '__interrupt__' not in final_state:
        messages = final_state.get('messages', [])
        if messages:
            print(f'\n{messages[-1].content}\n')

    return final_state or {}


def chat():
    """Run an interactive chat loop with interrupt handling."""
    _configure_output_encoding()
    has_history = _has_checkpoint_history(DEFAULT_THREAD_ID)
    from job_hunt.agent import agent

    config = {'configurable': {'thread_id': DEFAULT_THREAD_ID}}
    show_tools = False
    print('Ciallo~(∠・ω< )⌒★ job_hunt 已启动。输入 /help 查看命令 ❛‿˂̵ \n')
    if has_history:
        print(f'已恢复历史会话啦 (｀・ω・´)：{WORKSPACE_ROOT}\n')
    else:
        print(f'已新建会话啦 (｡･∀･)ﾉﾞ：{WORKSPACE_ROOT}\n')

    while True:
        try:
            user_input = input('请输入喵> ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\n再见啦 ( ´･･)ﾉ(._.`)')
            break

        if not user_input:
            continue
        if user_input == '/help':
            _print_help()
            continue
        if user_input == '/exit':
            print('再见啦 ( ´･･)ﾉ(._.`)')
            break
        if user_input == '/new':
            config = {'configurable': {'thread_id': f'cli-{id(config)}'}}
            print('已开始新会话 (ง •_•)ง\n')
            continue
        if user_input == '/debug':
            show_tools = not show_tools
            status = '开启' if show_tools else '关闭'
            print(f'已{status}工具调用显示 (๑•̀ㅂ•́)و✧\n')
            continue

        # Run the initial request. A failed request should not terminate the CLI.
        try:
            result = _run_agent(
                agent,
                {'messages': [{'role': 'user', 'content': user_input}]},
                config,
                show_tools,
            )
        except (EOFError, KeyboardInterrupt):
            print('\n再见啦 ( ´･･)ﾉ(._.`)')
            break
        except Exception as error:
            _print_error(
                '模型或工具调用失败，请检查网络连接和 API 配置。',
                error,
            )
            continue

        # Handle any interrupts
        try:
            while '__interrupt__' in result:
                resume = _handle_interrupt(result['__interrupt__'])
                result = _run_agent(agent, Command(resume=resume), config, show_tools)
        except (EOFError, KeyboardInterrupt):
            print('\n再见啦 ( ´･･)ﾉ(._.`)')
            break
        except Exception as error:
            _print_error(
                '工具审批或会话恢复失败，当前请求已停止。',
                error,
            )
            continue


if __name__ == '__main__':
    chat()
