# job_hunt 项目协作说明

本文档是当前仓库的主要协作说明，描述实际结构和开发约定。修改代码前应以源码和 `pyproject.toml` 为准。

## 项目概览

这是一个使用 LangChain/LangGraph 构建的交互式求职助手实验项目，当前通过 DeepSeek 模型回答问题，并提供基础计算、本地文件读取和 Tavily 网络搜索工具。

- Python 版本：`>=3.13`，当前版本由 `.python-version` 指定为 `3.13`
- 包管理和运行工具：`uv`
- 包布局：`src/job_hunt/`
- 命令行入口：`job-hunt = job_hunt.cli:chat`
- 当前没有 `tests/` 测试目录，验证以编译检查和必要的手动运行检查为主

## 目录结构

```text
src/job_hunt/
├── agent.py                 # 创建 Agent、注册 checkpoint
├── cli.py                   # 交互式命令行和人工审批流程
├── model.py                 # 加载环境变量并初始化 DeepSeek 模型
├── paths.py                 # 固定项目配置、启动工作区和 checkpoint 路径
├── security/
│   ├── middleware.py        # 模型调用上限和工具审批策略
│   └── sandbox.py            # 本地文件路径和敏感文件校验
└── tools/
    ├── file_lister.py        # 受限的工作区文件列表工具
    ├── file_reader.py       # 受限的本地文本文件读取工具
    ├── calculator.py        # 安全的基础数学计算工具
    └── web_search.py        # Tavily 搜索工具
```

## 核心运行流程

1. `paths.py` 定位 `job_hunt` 项目根目录，并在启动时固定一次当前工作目录作为本次运行的工作区。
2. `model.py` 从项目根目录 `.env` 加载环境变量，通过 `init_chat_model('deepseek-v4-flash')` 初始化模型，并关闭 thinking。
3. `tools/__init__.py` 将 `calculate`、`list_files`、`read_file` 和 `web_search` 汇总为 `ALL_TOOLS`。
4. `agent.py` 使用 `create_agent()` 组装模型、工具、SQLite checkpointer 和安全 middleware。
5. `cli.py` 以 `thread_id='cli-session'` 启动会话，启动时提示当前工作区是恢复历史会话还是新建会话；调用工具时处理 LangGraph interrupt，并在终端请求人工审批。

## 环境变量和密钥

运行前在 `job_hunt` 项目根目录的 `.env` 中配置当前代码需要的密钥：

- `DEEPSEEK_API_KEY`：DeepSeek 模型调用
- `TAVILY_API_KEY`：Tavily 网络搜索

`.env` 及其中的密钥不得读取后输出、提交到 Git 或写入日志。提交前仍应检查 `git status` 和暂存区。`DASHSCOPE_BASE_URL`、`DASHSCOPE_API_KEY` 虽然存在于本地配置中，但当前源码没有使用它们；除非同步修改模型初始化逻辑，否则不要把它们描述为当前运行链路的一部分。

## 运行和构建

在仓库根目录执行：

```powershell
uv sync
uv run job-hunt
```

CLI 支持以下内置命令：

- `/help`：查看命令说明
- `/exit`：退出程序
- `/new`：切换到新的会话线程
- `/debug`：开启或关闭简洁工具调用显示

Agent checkpoint 固定写入 `job_hunt` 项目根目录下的 `checkpoints/<工作区名>/agent_checkpoints.db`。工作区名由启动时的当前目录生成，例如 `D:\0\VSCode_Projects\Python\leetcode` 会映射为 `d--0-VSCode-Projects-Python-leetcode`。已有工作区目录会复用，不存在则自动创建；CLI 会根据该数据库中是否已有 `cli-session` checkpoint 提示恢复历史会话或新建会话。`checkpoints/`、`dist/`、`.venv/` 和构建产物已在 `.gitignore` 中排除，不要将这些生成文件作为源码提交。

如需构建发行包：

```powershell
uv build
```

## 工具和安全边界

### 本地文件读取

`list_files` 和 `read_file` 是两个职责独立的工具：前者只发现启动时固定工作区内可见的文件和目录，后者读取指定文本文件内容。`list_files` 默认不递归，并过滤敏感条目和生成目录；`read_file` 仍需人工审批。

`read_file` 只能读取启动时固定工作区以内的文件。路径会先解析，再拒绝目录外路径和敏感文件；敏感模式包括 `.env`、密钥/证书、凭据、密码、token，以及部分 SSH、云服务和系统认证文件。

它只返回文本内容：会检测二进制文件，尝试 `utf-8`、`gbk` 和 `latin-1` 编码，并将超过 100 KB 的内容截断。修改路径校验或文件读取策略时，应同时考虑路径穿越、敏感文件泄露、二进制文件和超大文件场景。

### 基础计算

`calculate` 使用 Python AST 白名单解析基础数学表达式，不使用 `eval`，支持四则运算、整除、取模、幂运算、`pi`、`e` 和部分常用数学函数。它不访问文件、网络或环境变量，因此可以自动批准。

### 工具审批和调用限制

`create_security_middleware()` 当前配置为：

- 每次 Agent 调用最多进行 10 次模型调用，超出后结束本次运行
- `calculate`：只进行本地数学计算，自动批准
- `list_files`：只返回非敏感的工作区条目，自动批准
- `read_file`：需要人工审批
- `tavily_search`：自动批准

新增工具时，需要同时检查 `tools/__init__.py` 的注册列表，以及 `security/middleware.py` 中的审批策略。涉及写文件、执行命令、发送外部请求或其他副作用的工具，默认应要求人工审批，并明确展示参数。

## 修改约定

- 优先做局部、清晰的修改，保持现有 `src` 包布局和 LangChain/LangGraph 组装方式。
- 必填配置直接使用，不要用 `.get()` 静默隐藏配置错误。
- 依赖变更通过 `uv add`、`uv remove` 或直接修改 `pyproject.toml` 后运行 `uv lock`，保持 `pyproject.toml` 与 `uv.lock` 同步。
- 遵循 Ruff 配置：行宽 100，单引号格式，当前 lint 规则为 `E` 和 `F`。
- 不要为了修复无关问题改写项目说明、生成文件或用户已有的工作区变更。
- 手动编辑使用 `apply_patch`，避免提交密钥、checkpoint 数据库和构建产物。

## 验证方式

默认对改动文件执行 Python 编译检查，例如：

```powershell
uv run python -m py_compile src/job_hunt/agent.py src/job_hunt/cli.py
```

涉及依赖、入口点或打包配置时，再执行 `uv sync` 或 `uv build`。需要验证完整行为时，使用已配置密钥运行 `uv run job-hunt`，但不要在输出中暴露密钥或敏感文件内容。
