# job_hunt

一个基于 LangChain/LangGraph 的交互式求职助手实验项目。当前使用 DeepSeek 模型，并提供本地文本文件读取和 Tavily 网络搜索工具。

## 环境要求

- Python 3.13+
- `uv`
- DeepSeek API 密钥
- Tavily API 密钥（使用网络搜索时需要）

## 配置和运行

在 `job_hunt` 项目根目录创建 `.env`，配置：

```dotenv
DEEPSEEK_API_KEY=your_deepseek_api_key
TAVILY_API_KEY=your_tavily_api_key
```

然后安装依赖并启动命令行 Agent：

```powershell
uv sync
uv run job-hunt
```

如果希望在其他工作目录中直接启动 Agent，推荐在 `job_hunt` 项目根目录执行一次：

```powershell
uv tool install --editable .
```

之后可以在任意目录执行：

```powershell
job-hunt
```

Agent 会把启动命令所在目录固定为本次运行的文件工具工作区，但 `.env` 和 system prompt 仍固定使用 `job_hunt` 项目目录下的文件；checkpoint 会按工作区分别保存到 `job_hunt/checkpoints/<工作区名>/agent_checkpoints.db`。
启动时，CLI 会提示当前工作区是已恢复历史会话还是已新建会话。

跨目录使用时推荐直接运行 `job-hunt`；`uv run job-hunt` 会优先按当前目录发现项目，不适合作为全局启动方式。

CLI 内置命令：

- `/help`：查看命令说明
- `/exit`：退出程序
- `/new`：开始新的会话
- `/debug`：开启或关闭简洁工具调用显示

## 当前功能

- 使用 DeepSeek-v4-flash 模型进行对话
- 进行安全的基础数学计算
- 浏览当前工作目录内的可见文件和目录
- 读取当前工作目录内的文本文件
- 通过 Tavily 搜索网络信息
- 使用 SQLite 保存 Agent 会话 checkpoint
- 读取本地文件前进行人工审批
- 默认流式输出回答

详细的项目结构、安全边界、修改约定和验证方式见 [`AGENTS.md`](AGENTS.md)。
