# OpenAI-to-Claude API 代理服务

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

将 OpenAI API 转为 Anthropic API 兼容格式的高性能代理服务。允许开发者使用现有的 Anthropic 客户端代码无缝调用 OpenAI 模型。

## 🌟 核心特性

- ✅ **无缝兼容**: 使用标准 Anthropic 客户端调用 OpenAI 模型
- ✅ **完整功能**: 支持文本、工具调用、流式响应等功能
- ✅ **智能路由**: 根据请求内容自动选择最适合的 OpenAI 模型
- ✅ **热重载**: 配置文件修改后自动重载，无需重启服务
- ✅ **结构化日志**: 详细的请求/响应日志，便于调试和监控
- ✅ **错误映射**: 完善的错误处理和映射机制

## 🚀 快速开始

### 环境要求

- Python 3.11+
- uv (推荐的包管理器)

### 安装依赖

```bash
# 使用 uv 安装依赖（推荐）
uv sync
```

### 配置

1. 复制示例配置文件：
```bash
cp config/example.json config/settings.json
```

2. 编辑 `config/settings.json`：
```json
{
  "openai": {
    "api_key": "your-openai-api-key-here",  // 替换为你的 OpenAI API 密钥
    "base_url": "https://api.openai.com/v1"  // OpenAI API 地址
  },
  "api_key": "your-proxy-api-key-here",  // 代理服务的 API 密钥
  // 其他配置...
}
```

### 启动服务

```bash
# 开发模式
uv run main.py --config config/settings.json

# 生产模式
uv run main.py
```

### 使用 Docker 启动

```bash
# 构建并启动服务
docker-compose up --build

# 后台运行
docker-compose up --build -d

# 停止服务
docker-compose down
```

服务将在 `http://localhost:8000` 启动。

## 🎯 模型选择

代理服务提供两种模型选择方式：

### 1. 直接指定模型（客户端控制）

客户端可以在请求中直接指定任何模型名称。代理会使用您指定的确切模型：

```python
from anthropic import Anthropic

client = Anthropic(
    base_url="http://localhost:8000/v1",
    api_key="your-proxy-api-key-here"
)

# 直接指定模型 - 将精确使用您指定的模型
response = client.messages.create(
    model="gpt-4o",  # 将直接使用 gpt-4o
    messages=[{"role": "user", "content": "你好！"}],
    max_tokens=1024
)

# 另一个例子 - 指定自定义模型
response = client.messages.create(
    model="deepseek-ai/DeepSeek-V3",  # 将使用这个确切的模型
    messages=[{"role": "user", "content": "你好！"}],
    max_tokens=1024
)
```

### 2. 智能路由（自动选择）

如果您使用通用的 Claude 模型名称（如 `claude-3-5-sonnet`、`claude-haiku`），代理将根据您的请求特征自动路由到最合适的模型：

```python
# 使用通用的 Claude 模型名称触发智能路由
response = client.messages.create(
    model="claude-3-5-sonnet",  # 将根据请求内容进行路由
    messages=[{"role": "user", "content": "你好！"}],
    max_tokens=1024
)
```

#### 路由规则

智能路由考虑以下因素（按优先级排序）：

1. **网页搜索工具**：如果请求包含 `web_search` 工具 → 使用 `web_search` 模型（仅 Gemini）
2. **长上下文**：如果总 token 数 > 100,000 → 使用 `long_context` 模型
3. **思考模式**：如果启用了 `thinking` → 使用 `think` 模型（用于推理）
4. **模型名称模式**：
   - 包含 "haiku" → 使用 `small` 模型
   - 包含 "sonnet" → 使用 `default` 模型
   - 包含 "opus" → 使用 `default` 模型

#### 示例：智能路由

```python
# 示例 1：基于思考模式的自动路由
response = client.messages.create(
    model="claude-3-5-sonnet",
    messages=[{"role": "user", "content": "解决这个复杂问题..."}],
    thinking={"type": "enabled"},  # 将路由到 "think" 模型
    max_tokens=1024
)

# 示例 2：长上下文自动路由
long_messages = [{"role": "user", "content": "..." * 50000}]  # 非常长的内容
response = client.messages.create(
    model="claude-3-5-sonnet",
    messages=long_messages,  # 将路由到 "long_context" 模型
    max_tokens=1024
)

# 示例 3：网页搜索自动路由
response = client.messages.create(
    model="claude-3-5-sonnet",
    messages=[{"role": "user", "content": "搜索..."}],
    tools=[{"type": "web_search", "name": "web_search"}],  # 将路由到 "web_search" 模型
    max_tokens=1024
)
```

### 模型配置

在 `config/settings.json` 中配置智能路由使用的模型：

```json
{
  "models": {
    "default": "gpt-4o",                // 用于 sonnet 类请求
    "small": "gpt-4o-mini",             // 用于 haiku 类请求
    "think": "deepseek-ai/DeepSeek-R1", // 用于思考/推理任务
    "long_context": "gemini-2.5-pro",   // 用于长上下文（>100k tokens）
    "web_search": "gemini-2.5-flash"    // 用于网页搜索工具
  }
}
```

### 选择直接指定 vs 智能路由

| 使用场景 | 推荐方式 | 示例 |
|---------|---------|------|
| 您明确知道要使用哪个模型 | 直接指定 | `model="gpt-4o"` |
| 您希望代理优化模型选择 | 智能路由 | `model="claude-3-5-sonnet"` |
| 您正在从 Anthropic Claude 迁移 | 智能路由 | 继续使用 `claude-*` 模型名称 |
| 您需要一致的模型行为 | 直接指定 | 指定确切的模型名称 |
| 您希望成本/性能优化 | 智能路由 | 让代理选择最佳模型 |

## 🛠️ 使用方法

### Claude Code 使用方法

本项目可以与 [Claude Code](https://claude.ai/code) 一起使用进行开发和测试。要配置 Claude Code 以使用此代理服务，请创建一个 `.claude/settings.json` 文件，配置如下：

```json
{
    "env": {
        "ANTHROPIC_API_KEY": "your-api-key",
        "ANTHROPIC_BASE_URL": "http://127.0.0.1:8000",
        "DISABLE_TELEMETRY": "1",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"
    },
    "apiKeyHelper": "echo 'your-api-key'",
    "permissions": {
        "allow": [],
        "deny": []
    }
}
```

配置说明：
- 将 `ANTHROPIC_API_KEY` 替换为您配置的 API 密钥，在 `config/settings.json` 中
- 将 `ANTHROPIC_BASE_URL` 替换为此代理服务实际运行的 URL
- `apiKeyHelper` 字段也应更新为您的 API 密钥

### 使用 Anthropic Python 客户端

```python
from anthropic import Anthropic

# 初始化客户端，指向代理服务
client = Anthropic(
    base_url="http://localhost:8000/v1",
    api_key="your-proxy-api-key-here"  # 使用配置文件中的 api_key
)

# 发送消息请求
response = client.messages.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "你好，GPT！"}
    ],
    max_tokens=1024
)

print(response.content[0].text)
```

### 流式响应

```python
# 流式响应
stream = client.messages.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "给我讲一个关于 AI 的故事"}
    ],
    max_tokens=1024,
    stream=True
)

for chunk in stream:
    if chunk.type == "content_block_delta":
        print(chunk.delta.text, end="", flush=True)
```

### 工具调用

```python
# 工具调用
tools = [
    {
        "name": "get_current_weather",
        "description": "获取指定城市的当前天气",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称"
                }
            },
            "required": ["city"]
        }
    }
]

response = client.messages.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "北京现在的天气怎么样？"}
    ],
    tools=tools,
    tool_choice={"type": "auto"}
)
```

## 📁 项目结构

```
openai-to-claude/
├── src/
│   ├── api/             # API 端点和中间件
│   ├── config/          # 配置管理
│   ├── core/            # 核心业务逻辑
│   │   ├── clients/     # HTTP 客户端
│   │   └── converters/  # 数据格式转换器
│   ├── models/          # Pydantic 数据模型
│   └── common/          # 公共工（日志、token计数等）
├── config/              # 配置文件
├── tests/               # 测试套件
├── CLAUDE.md           # Claude Code 项目指导
└── pyproject.toml      # 项目依赖和配置
```

## 🤖 Claude Code 使用方法

本项目可以与 [Claude Code](https://claude.ai/code) 一起使用进行开发和测试。要配置 Claude Code 以使用此代理服务，请创建一个 `.claude/settings.json` 文件，配置如下：

### 示例配置文件

```json
{
    "env": {
        "ANTHROPIC_API_KEY": "sk-chen0v0...",
        "ANTHROPIC_BASE_URL": "http://127.0.0.1:8100",
        "DISABLE_TELEMETRY": "1",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"
    },
    "apiKeyHelper": "echo 'sk-chen0v0...'",
    "permissions": {
        "allow": [],
        "deny": []
    }
}
```

### 配置说明

- 将 `ANTHROPIC_API_KEY` 替换为您的实际 Anthropic API 密钥
- 将 `ANTHROPIC_BASE_URL` 替换为此代理服务实际运行的 URL
- `apiKeyHelper` 字段也应更新为您的实际 API 密钥

## 🔧 配置说明

### 环境变量

- `CONFIG_PATH`: 配置文件路径 (默认: `config/settings.json`)
- `LOG_LEVEL`: 日志级别 (默认: `INFO`)

### 配置文件 (`config/settings.json`)

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

#### 配置项说明

- **openai**: OpenAI API 配置
  - `api_key`: OpenAI API 密钥，用于访问 OpenAI 服务
  - `base_url`: OpenAI API 基础 URL，默认为 `https://api.openai.com/v1`

- **server**: 服务器配置
  - `host`: 服务监听主机地址，默认为 `0.0.0.0`（监听所有网络接口）
  - `port`: 服务监听端口，默认为 `8000`

- **api_key**: 代理服务的 API 密钥，用于验证访问 `/v1/messages` 端点的请求

- **logging**: 日志配置
  - `level`: 日志级别，可选值为 `DEBUG`、`INFO`、`WARNING`、`ERROR`、`CRITICAL`，默认为 `INFO`

- **models**: 模型配置，定义不同使用场景下的模型选择
  - `default`: 默认通用模型，用于一般请求
  - `small`: 轻量级模型，用于简单任务
  - `think`: 深度思考模型，用于复杂推理任务
  - `long_context`: 长上下文处理模型，用于处理长文本
  - `web_search`: 网络搜索模型，用于网络搜索，目前支持geimini

- **parameter_overrides**: 参数覆盖配置，允许管理员在配置文件中设置模型参数的覆盖值
  - `max_tokens`: 最大 token 数覆盖，设置后会覆盖客户端请求中的 max_tokens 参数
  - `temperature`: 温度参数覆盖，控制输出的随机程度，范围为 0.0-2.0
  - `top_p`: top_p 采样参数覆盖，控制候选词汇的概率阈值，范围为 0.0-1.0
  - `top_k`: top_k 采样参数覆盖，控制候选词汇的数量，范围为 >=0

## 🧪 测试

```bash
# 运行所有测试
pytest

# 运行单元测试
pytest tests/unit

# 运行集成测试
pytest tests/integration

# 生成覆盖率报告
pytest --cov=src --cov-report=html
```

## 📊 API 端点

- `POST /v1/messages` - Anthropic 消息 API
- `GET /health` - 健康检查端点
- `GET /` - 欢迎页面

## 🛡️ 安全性

- API 密钥验证
- 请求频率限制（计划中）
- 输入验证和清理
- 结构化日志记录

## 📈 性能监控

- 请求/响应时间监控
- 内存使用情况跟踪
- 错误率统计

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [claude-code-router](https://github.com/musistudio/claude-code-router) - 很好的项目，本项目很多地方参考了这个项目
- [FastAPI](https://fastapi.tiangolo.com/) - 现代高性能 Web 框架
- [Anthropic](https://www.anthropic.com/) - Claude AI 模型
- [OpenAI](https://openai.com/) - OpenAI API 规范