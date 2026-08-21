import sqlite3
from pathlib import Path

from langchain.agents import create_agent
from langgraph.checkpoint.sqlite import SqliteSaver

from job_hunt.model import model
from job_hunt.tools import ALL_TOOLS
from job_hunt.security.middleware import create_security_middleware

DB_DIR = Path('checkpoints')
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = str(DB_DIR / 'agent_checkpoints.db')

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(conn)

agent = create_agent(
    model,
    tools=ALL_TOOLS,
    checkpointer=checkpointer,
    middleware=create_security_middleware(),
)
