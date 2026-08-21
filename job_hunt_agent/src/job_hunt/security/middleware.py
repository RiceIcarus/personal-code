from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
)


def create_security_middleware() -> list:
    """Assemble and return security middleware for the agent.

    Middleware order:
        1. ModelCallLimitMiddleware — max 10 model calls per invoke
        2. HumanInTheLoopMiddleware — any tool listed as True requires
           human approval before execution; False means auto-approved
    """
    return [
        ModelCallLimitMiddleware(run_limit=10, exit_behavior='end'),
        HumanInTheLoopMiddleware(
            interrupt_on={
                # Any tool NOT listed here passes through without approval.
                # Add new tools here to require manual approval.
                'read_file': True,        # could read sensitive local files
                'tavily_search': False,   # read-only external, safe to auto-approve
            },
            description_prefix='Tool call needs approval',
        ),
    ]
