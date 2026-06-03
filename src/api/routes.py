from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, Request

from src.config.settings import Config
from src.core.clients.openai_client import OpenAIServiceClient

router = APIRouter()


async def get_openai_client() -> OpenAIServiceClient:
    """获取OpenAI客户端实例"""
    config = await Config.from_file()
    # 获取有效的API keys列表（支持单key和多key配置）
    effective_keys = config.openai.get_effective_keys()
    if not effective_keys:
        raise ValueError("配置中没有有效的OpenAI API密钥")

    # 使用第一个有效的API key
    first_key = effective_keys[0]
    return OpenAIServiceClient(
        api_key=first_key["api_key"],
        base_url=first_key["base_url"],
    )


@router.get("/health", tags=["health"])
async def health_check(
    client: OpenAIServiceClient = Depends(get_openai_client),
) -> dict[str, Any]:
    """健康检查端点 - 验证OpenAI连通性"""

    health_status = {
        "status": "healthy",
        "service": "openai-to-claude",
        "timestamp": datetime.now().isoformat(),
        "checks": {},
    }

    try:
        # 检查OpenAI服务可用性
        openai_health = await client.health_check()
        health_status["checks"]["openai"] = openai_health

        # 如果任何一个检查失败，状态设为降级
        if not all(openai_health.values()):
            health_status["status"] = "degraded"

    except Exception as e:
        # 如果无法创建客户端或者检查抛出异常
        health_status["status"] = "unhealthy"
        health_status["checks"]["openai"] = {
            "openai_service": False,
            "api_accessible": False,
            "error": str(e),
        }

    return health_status


@router.get("/api-keys/status", tags=["monitoring"])
async def get_api_keys_status(request: Request) -> dict[str, Any]:
    """获取所有 API Key 的状态和使用统计

    返回信息包括：
    - 当前使用的策略
    - 每个 key 的可用性状态
    - 使用次数、失败次数、token 使用量
    - 当前使用的 key
    """
    # 从应用状态获取消息处理器（已由main.py在启动时初始化）
    handler = request.app.state.messages_handler

    if handler.key_rotator is None:
        return {
            "strategy": "single_key",
            "total_keys": 1,
            "keys": [
                {
                    "name": "default",
                    "is_current": True,
                    "is_available": True,
                }
            ],
        }

    # 获取轮换器的状态
    status = handler.key_rotator.get_status()

    # 添加额外的时间戳信息
    status["timestamp"] = datetime.now(timezone.utc).isoformat()

    return status


@router.post("/api-keys/reset", tags=["monitoring"])
async def reset_api_keys(request: Request) -> dict[str, Any]:
    """重置所有 API Key 的状态（清除配额耗尽和限流标记）

    用于手动恢复因临时错误被错误标记为不可用的 key
    """
    handler = request.app.state.messages_handler

    if handler.key_rotator is None:
        return {"status": "ok", "message": "单key模式，无需重置"}

    handler.key_rotator.reset_all_keys()

    # 更新客户端凭证到重置后的当前key
    current_key = handler.key_rotator.get_current_key()
    handler.client.update_credentials(current_key.api_key, current_key.base_url)

    return {
        "status": "ok",
        "message": "所有API Key状态已重置",
        "current_key": current_key.name,
    }
