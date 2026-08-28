from dotenv import load_dotenv
from langchain_tavily import TavilySearch

from job_hunt.paths import ENV_PATH

load_dotenv(ENV_PATH)

web_search = TavilySearch(max_results=5)
