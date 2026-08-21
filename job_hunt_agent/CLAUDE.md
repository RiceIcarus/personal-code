# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## User Preferences

- **LLM**: DeepSeek, `base_url=https://api.deepseek.com`
- **Framework**: LangChain, built from scratch, adding features incrementally
- **Interaction**: Write code step by step, ensure the user understands; DIY-first approach

## Project Overview

A "job-hunt" experimentation project using the OpenAI Python SDK to interact with LLM APIs (currently DeepSeek) via Jupyter notebooks. Python 3.13+, managed by `uv`.

## Architecture

- **Notebook-driven development**: All code lives in Jupyter notebooks (`.ipynb`). There is no `src/` package layout — notebooks are the primary and only development medium.
- **LLM client pattern**: The OpenAI SDK client is configured to point at DeepSeek's API (`https://api.deepseek.com`) via `base_url`. API keys are loaded from `.env` using `dotenv.load_dotenv()`. To switch to a different OpenAI-compatible provider, change `base_url` and the corresponding `*_API_KEY` env var.
- **Environment**: `.env` holds secrets (e.g., `DEEPSEEK_API_KEY`). Never commit `.env` — though it's currently not in `.gitignore`, do not add it to version control.

## Dependencies

| Package | Purpose |
|---------|---------|
| `openai` (≥2.45.0) | OpenAI-compatible SDK client |
| `dotenv` (≥0.9.9) | Load `.env` into `os.environ` |
| `notebook` (≥7.6.0) | Jupyter Notebook interface |

The PyPI index is set to Tsinghua mirror (`https://pypi.tuna.tsinghua.edu.cn/simple`).
