from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends

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
