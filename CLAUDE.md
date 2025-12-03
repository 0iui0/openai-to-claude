# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个高性能的 RESTful API 代理服务，将 Anthropic Claude API 格式转换为 OpenAI API 兼容格式，允许开发者使用现有的 Anthropic 客户端代码来调用 OpenAI 模型。

主要功能：
- ✅ 请求/响应格式转换（Anthropic ↔ OpenAI）
- ✅ 流式响应支持
- ✅ 错误处理映射
- ✅ 配置管理（支持热重载）
- ✅ 结构化日志
- ✅ 健康检查端点
- ✅ 模型智能路由（基于请求内容自动选择合适模型）
- ✅ Token 计数和缓存

## 项目结构

```
openai-to-claude/
├── src/
│   ├── api/             # API 端点和中间件
│   │   ├── handlers.py          # 核心消息处理
│   │   ├── middleware/          # 认证、计时等中间件
│   │   └── routes.py            # 健康检查路由
│   ├── config/          # 配置管理（含热重载监听）
│   ├── core/            # 核心业务逻辑
│   │   ├── clients/     # HTTP 客户端（OpenAI 服务客户端）
│   │   └── converters/  # 数据格式转换器（Anthropic ↔ OpenAI）
│   ├── models/          # Pydantic 数据模型（Anthropic 和 OpenAI 格式）
│   └── common/          # 公共工具（日志、token 计数、缓存等）
├── tests/               # 测试套件
│   └── integration/    # 集成测试（单元测试直接放在 tests/ 目录下）
├── config/              # 配置文件模板
└── pyproject.toml       # 项目依赖和配置（使用 uv 管理）
```

## 核心架构

### 数据流向
1. 客户端发送 Anthropic 格式的请求到 `/v1/messages` 端点
2. `MessagesHandler` 接收请求并使用 `AnthropicToOpenAIConverter` 转换为 OpenAI 格式
3. `OpenAIServiceClient` 将请求发送到实际的 OpenAI API 服务
4. 收到 OpenAI 响应后，使用 `OpenAIToAnthropicConverter` 转换回 Anthropic 格式
5. 返回给客户端

### 主要组件
- **src/api/handlers.py**: 核心消息处理逻辑，处理 `/v1/messages` 端点
- **src/api/middleware/**: 中间件（API 密钥认证、请求计时等）
- **src/core/converters/**: 请求/响应格式转换器（`AnthropicToOpenAIConverter`, `OpenAIToAnthropicConverter`）
- **src/core/clients/openai_client.py**: OpenAI API 客户端，负责与 OpenAI 服务通信
- **src/models/**: 数据模型定义（Anthropic 和 OpenAI 格式）
- **src/config/settings.py**: 配置管理，支持 JSON 配置文件和环境变量
- **src/config/watcher.py**: 配置文件热重载监听器
- **src/common/token_counter.py**: Token 计数和缓存
- **src/common/logging.py**: 结构化日志记录，支持请求 ID 跟踪

## 开发命令

### 安装依赖
```bash
# 使用 uv 安装依赖（推荐）
uv sync

# 或使用 pip（需先安装 uv 或 pip）
pip install -e .
```

### 运行服务
```bash
# 开发模式（使用默认配置文件）
uv run main.py

# 指定配置文件
uv run main.py --config config/settings.json

# 通过环境变量指定配置文件
CONFIG_PATH=config/settings.json uv run main.py

# 直接使用 uvicorn（不推荐，会跳过配置加载）
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker 运行
```bash
# 构建镜像
docker compose build

# 启动服务（前台）
docker compose up

# 启动服务（后台）
docker compose up -d

# 查看日志
docker compose logs -f

# 停止服务
docker compose down

# 构建并启动
docker compose up --build
```

### 运行测试
```bash
# 运行所有测试
pytest

# 运行集成测试
pytest tests/integration/

# 运行特定测试文件
pytest tests/integration/test_messages_endpoint.py

# 带覆盖率报告
pytest --cov=src --cov-report=html

# 运行测试并生成覆盖率 XML（用于 CI）
pytest --cov=src --cov-report=xml
```

### 代码质量检查
```bash
# 代码格式化
black .

# 代码 linting 检查
ruff check .

# 自动修复可修复的 linting 问题
ruff check . --fix

# 类型检查
mypy src

# 安全检查（bandit）
pip install bandit && bandit -r src/
```

## 配置说明

### 环境变量
- `CONFIG_PATH`: 配置文件路径，默认 `config/settings.json`
- `LOG_LEVEL`: 日志级别，默认 `INFO`，可选 `DEBUG`、`INFO`、`WARNING`、`ERROR`、`CRITICAL`

### 配置文件 (config/settings.json)
```json
{
  "openai": {
    "api_key": "your-openai-api-key-here",
    "base_url": "https://api.openai.com/v1"
  },
  "server": {
    "host": "0.0.0.0",
    "port": 8000
  },
  "api_key": "your-proxy-api-key-here",
  "logging": {
    "level": "INFO"
  },
  "models": {
    "default": "Qwen/Qwen3-Coder",
    "small": "deepseek-ai/DeepSeek-V3-0324",
    "think": "deepseek-ai/DeepSeek-R1-0528",
    "long_context": "gemini-2.5-pro",
    "web_search": "gemini-2.5-flash"
  },
  "parameter_overrides": {
    "max_tokens": null,
    "temperature": null,
    "top_p": null,
    "top_k": null
  }
}
```

### 配置字段说明
- **openai**: OpenAI API 配置
  - `api_key`: OpenAI API 密钥，用于访问 OpenAI 服务
  - `base_url`: OpenAI API 基础 URL，默认为 `https://api.openai.com/v1`
- **server**: 服务端配置
  - `host`: 服务监听地址，默认 `0.0.0.0`（监听所有网络接口）
  - `port`: 服务监听端口，默认 `8000`
- **api_key**: 代理服务的 API 密钥，用于验证对 `/v1/messages` 端点的请求
- **logging**: 日志配置
  - `level`: 日志级别，控制日志输出详细程度
- **models**: 模型配置，定义不同使用场景下的模型选择
  - `default`: 默认通用模型，用于一般请求
  - `small`: 轻量级模型，用于简单任务
  - `think`: 深度思考模型，用于复杂推理任务
  - `long_context`: 长上下文处理模型，用于处理长文本
  - `web_search`: 网页搜索模型，支持网络搜索（目前支持 Gemini）
- **parameter_overrides**: 参数覆盖配置，允许管理员在配置文件中设置模型参数覆盖值
  - `max_tokens`: 最大 token 数覆盖，设置后会覆盖客户端请求中的 max_tokens 参数
  - `temperature`: temperature 参数覆盖，控制输出的随机性，范围 0.0-2.0
  - `top_p`: top_p 采样参数覆盖，控制候选词的概率阈值，范围 0.0-1.0
  - `top_k`: top_k 采样参数覆盖，控制候选词的数量，范围 ≥0

配置文件支持热重载，修改配置文件后服务会自动重新加载配置，无需重启。

## Claude Code 集成

此代理服务可与 Claude Code 配合使用，将 Claude Code 的请求转发到 OpenAI 模型。配置示例：

1. 在 Claude Code 设置文件 `.claude/settings.json` 中添加：
```json
{
  "env": {
    "ANTHROPIC_API_KEY": "your-proxy-api-key-here",
    "ANTHROPIC_BASE_URL": "http://localhost:8000/v1",
    "DISABLE_TELEMETRY": "1",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"
  },
  "apiKeyHelper": "echo 'your-proxy-api-key-here'"
}
```

2. 确保 `ANTHROPIC_API_KEY` 与配置文件中的 `api_key` 一致
3. `ANTHROPIC_BASE_URL` 指向代理服务的地址（默认 `http://localhost:8000/v1`）

## API 端点

### 主要端点
- `POST /v1/messages` - Anthropic 消息 API（兼容 OpenAI 聊天 API），接收 Anthropic 格式的请求，返回 Anthropic 格式的响应
- `GET /health` - 健康检查端点，返回服务状态和 OpenAI API 连接状态

### 其他端点
- `GET /` - 欢迎页面，返回服务基本信息
- `GET /docs` - 自动生成的 API 文档（Swagger UI）
- `GET /openapi.json` - OpenAPI 规范文件

### 使用示例
```bash
# 健康检查
curl http://localhost:8000/health

# 发送消息请求
curl -X POST http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-proxy-api-key-here" \
  -d '{
    "model": "gpt-4o",
    "messages": [
      {"role": "user", "content": "Hello, world!"}
    ],
    "max_tokens": 1024
  }'
```

**注意**：所有请求到 `/v1/messages` 端点都需要在 `x-api-key` 头部提供配置文件中设置的 `api_key`。