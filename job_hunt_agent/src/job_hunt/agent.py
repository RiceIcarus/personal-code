import sqlite3

from langchain.agents import create_agent
from langgraph.checkpoint.sqlite import SqliteSaver

from job_hunt.model import model
from job_hunt.paths import CHECKPOINT_DB_PATH, CHECKPOINT_DIR, SYSTEM_PROMPT_PATH
from job_hunt.tools import ALL_TOOLS
from job_hunt.security.middleware import create_security_middleware

SYSTEM_PROMPT = SYSTEM_PROMPT_PATH.read_text(encoding='utf-8').strip()
if not SYSTEM_PROMPT:
    raise ValueError(f'System prompt is empty: {SYSTEM_PROMPT_PATH}')

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = str(CHECKPOINT_DB_PATH)

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(conn)

agent = create_agent(
    model,
    tools=ALL_TOOLS,
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
    middleware=create_security_middleware(),
)
