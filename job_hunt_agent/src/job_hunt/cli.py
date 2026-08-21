"""CLI entry point for the job_hunt agent with HITL interrupt handling.

When the agent tries to use a tool that requires approval, execution pauses
and the user is prompted in the terminal to approve or reject with a reason.
"""

from langgraph.types import Command

from job_hunt.agent import agent


def _format_action(action: dict) -> str:
    """Format an action request for display."""
    return (
        f"  Tool: {action['name']}\n"
        f"  Args: {action.get('args', {})}"
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
            print("Approve this tool call?")
            print(_format_action(action))
            print(f"{'=' * 50}")

            while True:
                choice = input("  [a]pprove  [r]eject: ").strip().lower()

                if choice in ('a', 'approve'):
                    decisions.append({'type': 'approve'})
                    break
                elif choice in ('r', 'reject'):
                    reason = input(
                        '  Why reject + what should the agent do instead: '
                    ).strip()
                    message = (
                        f'User rejected the tool call to `{name}`.\n'
                        f'Reason: {reason}'
                    )
                    decisions.append({'type': 'reject', 'message': message})
                    break
                else:
                    print("  Invalid choice. Try 'a' or 'r'.")

    return {'decisions': decisions}


def chat():
    """Run an interactive chat loop with interrupt handling."""
    config = {'configurable': {'thread_id': 'cli-session'}}
    print("job_hunt agent ready. Commands: /exit  /new\n")

    while True:
        try:
            user_input = input('> ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\nGoodbye.')
            break

        if not user_input:
            continue
        if user_input == '/exit':
            print('Goodbye.')
            break
        if user_input == '/new':
            config = {'configurable': {'thread_id': f'cli-{id(config)}'}}
            print('New session started.\n')
            continue

        # Run the agent, handling interrupts along the way
        result = agent.invoke(
            {'messages': [{'role': 'user', 'content': user_input}]},
            config=config,
        )

        # Handle any interrupts
        while '__interrupt__' in result:
            resume = _handle_interrupt(result['__interrupt__'])
            result = agent.invoke(
                Command(resume=resume),
                config=config,
            )

        # Print agent's final response
        last_msg = result['messages'][-1]
        print(f'\n{last_msg.content}\n')


if __name__ == '__main__':
    chat()
