from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from src.models.errors import get_error_response


class APIKeyMiddleware(BaseHTTPMiddleware):
    """中间件：验证API密钥"""

    def __init__(self, app, api_key: str):
        super().__init__(app)
        self.api_key = api_key

    async def dispatch(self, request: Request, call_next):
        # 检查是否为需要密钥验证的路径
        if not self._requires_auth(request.url.path):
            # 跳过认证，直接处理请求
            response = await call_next(request)
            return response

        # 尝试从多个位置获取API密钥
        token = self._extract_api_key(request)

        if token != self.api_key:
            error_response = get_error_response(401, message="API密钥无效")

            # 直接返回401响应，而不是抛出异常
            from fastapi.responses import JSONResponse

            return JSONResponse(status_code=401, content=error_response.dict())

        response = await call_next(request)
        return response

    def _extract_api_key(self, request: Request) -> str:
        """从请求中提取API密钥
        
        支持多种格式：
        1. x-api-key 请求头
        2. Authorization: Bearer <token> 请求头
        3. Authorization: <token> 请求头
        4. ANTHROPIC_AUTH_TOKEN 环境变量（fallback）
        """
        # 尝试方式1: x-api-key 请求头
        token = request.headers.get("x-api-key")
        if token:
            return token
        
        # 尝试方式2: Authorization 请求头
        auth_header = request.headers.get("authorization", "")
        if auth_header:
            # 移除 "Bearer " 前缀（如果存在）
            if auth_header.lower().startswith("bearer "):
                token = auth_header[7:]  # 移除 "Bearer " (7个字符)
            else:
                token = auth_header
            if token:
                return token
        
        # 如果没有找到，返回空字符串（会导致401）
        return ""
    
    def _requires_auth(self, path: str) -> bool:
        """检查路径是否需要API密钥验证"""
        # 只有 /v1/messages 需要密钥验证
        auth_required_paths = ["/v1/messages"]
        return path in auth_required_paths
