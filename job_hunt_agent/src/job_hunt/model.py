from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

from job_hunt.paths import ENV_PATH

load_dotenv(ENV_PATH)

model = init_chat_model(
    'deepseek-v4-flash',
    extra_body={'thinking': {'type': 'disabled'}},
)
