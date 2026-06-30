"""
Anthropic /v1/messages 端点处理程序

实现Anthropic native messages API与OpenAI API的转换和代理
"""

import asyncio
import json
import time
from collections.abc import AsyncGenerator
from contextlib import aclosing

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from src.core.api_key_rotator import APIKeyInfo, APIKeyRotator
from src.core.clients.openai_client import OpenAIClientError, OpenAIServiceClient
from src.core.converters.request_converter import (
    AnthropicToOpenAIConverter,
)
from src.core.converters.response_converter import OpenAIToAnthropicConverter
from src.models.anthropic import (
    AnthropicMessageResponse,
    AnthropicRequest,
)
from src.models.errors import get_error_response
from loguru import logger

router = APIRouter(prefix="/v1", tags=["messages"])


# ---- 模型回退辅助：上下文窗口匹配 ----

def _match_context_window(model_name: str, context_windows: dict[str, int]) -> int | None:
    """通过最长前缀匹配获取模型的上下文窗口大小。"""
    best: int | None = None
    best_len = 0
    for prefix, window in context_windows.items():
        if model_name.startswith(prefix) and len(prefix) > best_len:
            best = window
            best_len = len(prefix)
    return best


# ---- 超时断路器（跨请求共享） ----

_model_timeout_log: dict[str, list[float]] = {}
_model_timeout_lock = asyncio.Lock()


async def _record_model_timeout(model: str, window: float = 120.0) -> None:
    """记录一次模型超时，用于断路器状态跟踪。"""
    async with _model_timeout_lock:
        now = time.time()
        entries = _model_timeout_log.setdefault(model, [])
        entries.append(now)
        # 清理过期记录
        _model_timeout_log[model] = [t for t in entries if now - t < window]


async def _is_model_circuit_open(model: str, window: float = 120.0, max_timeouts: int = 2) -> bool:
    """检查模型是否处于断路器打开状态（近期连续超时 >= max_timeouts 次）。"""
    async with _model_timeout_lock:
        now = time.time()
        recent = [t for t in _model_timeout_log.get(model, []) if now - t < window]
        return len(recent) >= max_timeouts


class MessagesHandler:
    """处理Anthropic /v1/messages 端点请求"""

    def __init__(self, config):
        self.request_converter = AnthropicToOpenAIConverter()
        self.response_converter = OpenAIToAnthropicConverter()
        self.config = config
        self._config = None

        # 初始化 API Key 轮换器和客户端池
        api_keys_config = config.openai.get_effective_keys()
        if not api_keys_config:
            raise ValueError("未配置任何 API Key")

        if len(api_keys_config) == 1:
            # 单 key：不需要轮换器，直接使用
            self.key_rotator = None
            self._clients: dict[int, OpenAIServiceClient] = {
                0: OpenAIServiceClient(
                    api_key=api_keys_config[0]["api_key"],
                    base_url=api_keys_config[0]["base_url"],
                )
            }
        else:
            # 多 keys：使用 round_robin 策略 + per-key 客户端池
            # round_robin 在 asyncio.Lock 保护下循环分配 key，并发请求自动分散
            # per-key 客户端池确保每个 key 有独立的 HTTP 连接，无 credential 冲突
            self.key_rotator = APIKeyRotator(api_keys_config, strategy="round_robin")
            # 为每个 key 创建独立的 OpenAIServiceClient，避免并发时共享 client 的 credential 冲突
            self._clients: dict[int, OpenAIServiceClient] = {}
            for idx, key_config in enumerate(api_keys_config):
                self._clients[idx] = OpenAIServiceClient(
                    api_key=key_config["api_key"],
                    base_url=key_config.get("base_url"),
                )

    @classmethod
    async def create(cls, config=None):
        """异步工厂方法创建 MessagesHandler 实例"""
        if config is None:
            from src.config.settings import get_config

            config = await get_config()

        instance = cls.__new__(cls)
        instance.request_converter = AnthropicToOpenAIConverter()
        instance.response_converter = OpenAIToAnthropicConverter()
        instance.config = config
        instance._config = config

        # 初始化 API Key 轮换器和客户端池
        api_keys_config = config.openai.get_effective_keys()
        if not api_keys_config:
            raise ValueError("未配置任何 API Key")

        if len(api_keys_config) == 1:
            instance.key_rotator = None
            instance._clients = {
                0: OpenAIServiceClient(
                    api_key=api_keys_config[0]["api_key"],
                    base_url=api_keys_config[0]["base_url"],
                )
            }
        else:
            instance.key_rotator = APIKeyRotator(api_keys_config, strategy="balanced")
            instance._clients = {}
            for idx, key_config in enumerate(api_keys_config):
                instance._clients[idx] = OpenAIServiceClient(
                    api_key=key_config["api_key"],
                    base_url=key_config.get("base_url"),
                )

        return instance

    async def _get_client_for_session(self, session_id: str | None = None, model_name: str | None = None) -> tuple[OpenAIServiceClient, APIKeyInfo | None]:
        """根据 session 或模型名获取对应的 client 和 key 信息（并发安全）

        当传入 model_name 且存在多个 key 时，优先选择 model 字段匹配的 key。
        匹配规则：
          1. 先通过 config.models.model_aliases 解析别名（如 claude-sonnet-4-6 → gemma4-agentic）
          2. 再匹配 key.model（精确字符串包含匹配）
        若无匹配则回退到 session 策略。

        Args:
            session_id: 会话标识符（用于 session_affinity 策略）
            model_name: 请求的模型名（用于模型路由）

        Returns:
            (client, key_info) 元组。key_info 在无轮换器时为 None
        """
        if self.key_rotator is None:
            return self._clients[0], None

        # 模型名路由：当有多个 key 且指定了模型名时，按 model 字段匹配
        if model_name and len(self.key_rotator.api_keys) > 1:
            # 解析模型别名（claude-sonnet-4-6 → gemma4-agentic）
            resolved_model = model_name
            if self.config and self.config.models and self.config.models.model_aliases:
                if model_name in self.config.models.model_aliases:
                    resolved_model = self.config.models.model_aliases[model_name]

            for key in self.key_rotator.api_keys:
                if key.model and key.model in resolved_model:
                    client = self._clients[key.index]
                    async with self.key_rotator._lock:
                        self.key_rotator.current_key_index = key.index
                        if session_id:
                            self.key_rotator.session_key_mapping[session_id] = key.index
                    logger.info(f"模型路由: {model_name}(->{resolved_model}) -> Key [{key.name}] (model={key.model})")
                    return client, key

            # 精确匹配未命中，尝试反向匹配（resolved_model 包含在 key.model 中）
            for key in self.key_rotator.api_keys:
                if key.model and resolved_model in key.model:
                    client = self._clients[key.index]
                    async with self.key_rotator._lock:
                        self.key_rotator.current_key_index = key.index
                        if session_id:
                            self.key_rotator.session_key_mapping[session_id] = key.index
                    logger.info(f"模型路由(反向): {model_name}(->{resolved_model}) -> Key [{key.name}] (model={key.model})")
                    return client, key

        # 回退到 session 策略
        key_info = await self.key_rotator.get_current_key(session_id=session_id)
        client = self._clients[key_info.index]
        return client, key_info

    async def _handle_client_error_with_retry(
        self,
        error: OpenAIClientError,
        request_id: str | None = None,
        session_id: str | None = None,
    ):
        """处理客户端错误并根据需要切换 API key

        Args:
            error: OpenAI 客户端错误
            request_id: 请求 ID
            session_id: 会话 ID（用于 session_affinity 策略的错误处理）

        Raises:
            HTTPException: 如果无法恢复或所有 keys 都不可用
        """
        from src.common.logging import get_logger_with_request_id

        bound_logger = get_logger_with_request_id(request_id)

        if self.key_rotator is None:
            # 没有轮换器，直接抛出错误
            bound_logger.warning("未启用 API Key 轮换，直接返回错误")
            raise HTTPException(
                status_code=error.status_code or 500,
                detail=error.error_response.model_dump(exclude_none=True),
            )

        # 使用轮换器处理错误，传递 session_id 以便 session_affinity 策略正确处理
        status_code = error.status_code or 500
        error_message = error.error_message or str(error.error_response.error)

        await self.key_rotator.handle_error(status_code, error_message, session_id=session_id)

        # 获取该 session 切换后的新 key 和对应的独立 client
        new_key = await self.key_rotator.get_current_key(session_id=session_id)
        bound_logger.info(f"已切换到新的 API Key [{new_key.name}] for session {session_id[:8] if session_id else 'global'}...")

    async def _send_request_with_retry(
        self,
        openai_request,
        request_id: str | None = None,
        session_id: str | None = None,
        max_retries: int = 3,
        estimated_input_tokens: int | None = None,
    ):
        """发送请求并在配额用尽时自动重试，支持403权限错误的模型回退

        Args:
            openai_request: OpenAI 请求对象
            request_id: 请求 ID
            session_id: 会话 ID（用于 session_affinity 策略）
            max_retries: 最大重试次数
            estimated_input_tokens: 预估输入 token 数，用于跳过窗口过小的模型

        Returns:
            OpenAI 响应

        Raises:
            HTTPException: 如果所有重试都失败
        """
        from src.common.logging import get_logger_with_request_id
        from src.config.settings import get_config

        bound_logger = get_logger_with_request_id(request_id)
        last_error = None

        # 获取当前 session 对应的 client 和 key（按模型名路由）
        client, current_key = await self._get_client_for_session(session_id, openai_request.model)

        while True:
            config = await get_config()
            fallback_models = config.models.fallback_models
            context_windows = config.models.context_windows

            # 检查当前 key 是否有模型覆盖（如本地 vLLM 固定模型）
            if current_key and current_key.model:
                models_to_try = [current_key.model]
                openai_request.model = current_key.model
            else:
                current_model = openai_request.model
                models_to_try = [current_model] + [m for m in fallback_models if m != current_model]

            # 按上下文窗口和断路器过滤模型
            if estimated_input_tokens and context_windows:
                filtered = []
                for model in models_to_try:
                    window = _match_context_window(model, context_windows)
                    if window and estimated_input_tokens > window:
                        bound_logger.warning(
                            f"跳过模型 {model}：输入 {estimated_input_tokens} tokens "
                            f"超过窗口上限 {window} tokens"
                        )
                        continue
                    if await _is_model_circuit_open(model):
                        bound_logger.warning(
                            f"跳过模型 {model}：断路器已打开（近期多次超时），跳过"
                        )
                        continue
                    filtered.append(model)
                if not filtered:
                    bound_logger.error(
                        f"所有模型均被跳过（窗口不足或断路器打开），原始列表: {models_to_try}"
                    )
                    raise HTTPException(
                        status_code=503,
                        detail={
                            "error": {
                                "message": (
                                    f"所有模型均不可用（输入 {estimated_input_tokens} tokens "
                                    f"超过上下文窗口或断路器打开）"
                                )
                            }
                        },
                    )
                if len(filtered) < len(models_to_try):
                    bound_logger.warning(
                        f"模型列表已过滤: {models_to_try} -> {filtered}"
                    )
                models_to_try = filtered

            key_switched_with_override = False

            for model_idx, model_to_try in enumerate(models_to_try):
                openai_request.model = model_to_try

                # 追踪当前模型已尝试过403的key索引（用于换key而非直接跳模型）
                tried_key_indices: set[int] = set()

                for attempt in range(max_retries):
                    # 对 fallback 模型使用更短的超时（30s），快速失败
                    request_timeout: float | None = None
                    if model_idx > 0:
                        request_timeout = min(client.timeout, 30.0)

                    try:
                        if model_idx > 0:
                            bound_logger.info(
                                f"尝试回退模型 {model_idx}/{len(models_to_try)}: {openai_request.model} -> {model_to_try}"
                            )

                        # 使用 session 对应的独立 client 发送请求
                        response = await client.send_request(
                            openai_request, request_id=request_id, timeout=request_timeout
                        )

                        if model_idx > 0:
                            bound_logger.info(f"回退模型 {model_to_try} 成功响应")

                        return response

                    except OpenAIClientError as e:
                        last_error = e
                        is_permission_error = (e.status_code == 403)
                        is_client_error = (400 <= e.status_code < 500) and e.status_code not in (403, 429)

                        bound_logger.warning(
                            f"请求失败 (模型: {model_to_try}, 尝试 {attempt + 1}/{max_retries}) - Status: {e.status_code}"
                        )

                        if is_permission_error:
                            key_name = current_key.name if current_key else "unknown"
                            if current_key is not None:
                                tried_key_indices.add(current_key.index)

                            # 先尝试换一个key，所有key都无权使用时才换模型
                            if self.key_rotator and current_key is not None:
                                total_keys = len(self.key_rotator.api_keys)
                                found_new_key = False
                                for offset in range(1, total_keys):
                                    next_idx = (current_key.index + offset) % total_keys
                                    next_key_info = self.key_rotator.api_keys[next_idx]
                                    if next_key_info.is_available() and next_idx not in tried_key_indices:
                                        async with self.key_rotator._lock:
                                            self.key_rotator.current_key_index = next_idx
                                            if session_id:
                                                self.key_rotator.session_key_mapping[session_id] = next_idx
                                        client = self._clients[next_idx]
                                        current_key = next_key_info
                                        found_new_key = True
                                        bound_logger.warning(
                                            f"检测到403权限错误，模型 {model_to_try} 无权限 (Key: [{key_name}])，"
                                            f"切换到 Key: [{current_key.name}] 重试同一模型"
                                        )
                                        break

                                if found_new_key:
                                    continue  # 用新key重试同一模型

                            # 所有key都无权使用此模型，尝试下一个模型
                            bound_logger.warning(
                                f"检测到403权限错误，模型 {model_to_try} 所有Key均无权限，尝试下一个模型"
                            )
                            break

                        if is_client_error:
                            bound_logger.warning(f"客户端错误 {e.status_code}，不重试")
                            break

                        # 504超时：记录断路器，最多重试1次，退避与超时值成比例
                        is_timeout = (e.status_code == 504)
                        if is_timeout:
                            await _record_model_timeout(model_to_try)
                            if attempt < 1:
                                effective_timeout = request_timeout or client.timeout
                                backoff = min(30.0, effective_timeout / 2)
                                bound_logger.warning(
                                    f"请求超时，{backoff:.0f}s 后退避重试 (模型: {model_to_try}, "
                                    f"尝试 {attempt + 1}/{max_retries})"
                                )
                                await asyncio.sleep(backoff)
                                continue
                            else:
                                bound_logger.warning(
                                    f"请求超时重试已用尽 (模型: {model_to_try})，尝试下一个模型"
                                )
                                break

                        if self.key_rotator and attempt < max_retries - 1:
                            await self._handle_client_error_with_retry(
                                e, request_id=request_id or "default", session_id=session_id
                            )
                            # 错误处理后重新获取 session 对应的 client 和 key
                            client, current_key = await self._get_client_for_session(session_id, openai_request.model)
                            if current_key and current_key.model:
                                openai_request.model = current_key.model
                                bound_logger.info(
                                    f"新 key [{current_key.name}] 有固定模型，切换至: {current_key.model}"
                                )
                                key_switched_with_override = True
                                break
                            continue
                        else:
                            break

                if key_switched_with_override:
                    break

            if not key_switched_with_override:
                break

        error_msg = f"所有模型尝试均失败: {', '.join(models_to_try)}"
        bound_logger.error(error_msg)
        raise HTTPException(
            status_code=last_error.status_code if last_error else 500,
            detail=last_error.error_response.model_dump(exclude_none=True)
            if last_error
            else {"error": {"message": error_msg}},
        )

    async def process_message(
        self, request: AnthropicRequest, request_id: str = None, session_id: str | None = None
    ) -> AnthropicMessageResponse:
        """处理非流式消息请求"""
        # 获取绑定了请求ID的logger
        from src.common.logging import get_logger_with_request_id

        bound_logger = get_logger_with_request_id(request_id)

        # 获取当前 session 对应的 client 和 key（按模型名路由）
        client, current_key = await self._get_client_for_session(session_id, request.model)

        try:
            bound_logger.debug("处理非流式请求")
            # 验证请求
            # await validate_anthropic_request(request, request_id)
            # 将 Anthropic 请求转换为 OpenAI 格式（异步）
            openai_request = await self.request_converter.convert_anthropic_to_openai(
                request, request_id
            )

            # 获取缓存的输入 token 数，用于跳过窗口过小的 fallback 模型
            from src.common.token_cache import get_cached_tokens
            estimated_input_tokens = get_cached_tokens(request_id, delete=False)

            # 发送到 OpenAI（带重试机制），传递 session_id
            openai_response = await self._send_request_with_retry(
                openai_request, request_id=request_id, session_id=session_id,
                estimated_input_tokens=estimated_input_tokens,
            )
            bound_logger.debug(
                f"OpenAI 响应: {json.dumps(openai_response, ensure_ascii=False)}"
            )

            # 将 OpenAI 响应转回 Anthropic 格式
            anthropic_response = await self.response_converter.convert_response(
                openai_response, request.model, request_id
            )
            # 安全地提取响应文本
            response_text = "empty"
            if (
                anthropic_response.content
                and len(anthropic_response.content) > 0
                and hasattr(anthropic_response.content[0], "text")
                and anthropic_response.content[0].text
            ):
                response_text = anthropic_response.content[0].text
            bound_logger.info(
                f"Anthropic 响应生成完成 - Text: {response_text[:100]}..., Usage: {anthropic_response.usage}"
            )

            # 标记 API key 使用成功，并统计 token 使用量（使用 session 对应的 key）
            if self.key_rotator and current_key and anthropic_response.usage:
                total_tokens = (
                    (anthropic_response.usage.input_tokens or 0)
                    + (anthropic_response.usage.output_tokens or 0)
                )
                current_key.mark_success(tokens_used=total_tokens)

            return anthropic_response

        except ValidationError as e:
            bound_logger.warning(f"Validation error - Errors: {e.errors()}")
            error_response = get_error_response(
                422, details={"validation_errors": e.errors(), "request_id": request_id}
            )
            raise HTTPException(status_code=422, detail=error_response.model_dump())

        except json.JSONDecodeError as e:
            # 专门处理JSON解析错误，这通常发生在OpenAI响应解析时
            bound_logger.exception(
                f"JSON解析错误 - Error: {str(e)}, Position: {e.pos if hasattr(e, 'pos') else 'unknown'}"
            )
            error_response = get_error_response(
                502,
                message="上游服务返回无效JSON格式",
                details={"json_error": str(e), "request_id": request_id},
            )
            raise HTTPException(status_code=502, detail=error_response.model_dump())
        except HTTPException as e:
            bound_logger.exception(
                f"处理非流式消息请求错误 - Type: {type(e).__name__}, Error: {str(e)}"
            )
            error_response = get_error_response(
                e.status_code, message=str(e.detail), details={"request_id": request_id}
            )
            raise HTTPException(
                status_code=e.status_code,
                detail=error_response.model_dump(exclude_none=True),
            )

        except Exception as e:
            bound_logger.exception(
                f"处理非流式消息请求错误 - Type: {type(e).__name__}, Error: {str(e)}"
            )
            error_response = get_error_response(
                500, message=str(e), details={"request_id": request_id}
            )
            raise HTTPException(
                status_code=500, detail=error_response.model_dump(exclude_none=True)
            )

    async def process_stream_message(
        self, request: AnthropicRequest, request_id: str = None, session_id: str | None = None
    ) -> AsyncGenerator[str, None]:
        """处理流式消息请求，支持模型回退和key轮换重试"""
        if not request.stream:
            raise ValueError("流式响应参数必须为true")

        from src.common.logging import get_logger_with_request_id
        from src.config.settings import get_config

        bound_logger = get_logger_with_request_id(request_id)
        total_tokens_used = 0

        # 获取当前 session 对应的 client 和 key（按模型名路由）
        stream_client, current_key = await self._get_client_for_session(session_id, request.model)

        try:
            # 转换请求（只做一次，重试时复用）
            openai_request = await self.request_converter.convert_anthropic_to_openai(
                request, request_id
            )

            # 获取缓存的输入 token 数，用于跳过窗口过小的 fallback 模型
            from src.common.token_cache import get_cached_tokens
            estimated_input_tokens = get_cached_tokens(request_id, delete=False)

            # 确定要尝试的模型列表（与 _send_request_with_retry 一致）
            config = await get_config()
            if current_key and current_key.model:
                models_to_try = [current_key.model]
                openai_request.model = current_key.model
                bound_logger.info(
                    f"流式请求应用 key [{current_key.name}] 的固定模型: {current_key.model}"
                )
            else:
                current_model = openai_request.model
                fallback_models = config.models.fallback_models if config.models else []
                models_to_try = [current_model] + [
                    m for m in fallback_models if m != current_model
                ]

            # 按上下文窗口和断路器过滤模型
            context_windows = config.models.context_windows if config.models else {}
            if estimated_input_tokens and context_windows:
                filtered = []
                for model in models_to_try:
                    window = _match_context_window(model, context_windows)
                    if window and estimated_input_tokens > window:
                        bound_logger.warning(
                            f"跳过模型 {model}：输入 {estimated_input_tokens} tokens "
                            f"超过窗口上限 {window} tokens"
                        )
                        continue
                    if await _is_model_circuit_open(model):
                        bound_logger.warning(
                            f"跳过模型 {model}：断路器已打开（近期多次超时），跳过"
                        )
                        continue
                    filtered.append(model)
                if not filtered:
                    bound_logger.error(
                        f"所有模型均被跳过（窗口不足或断路器打开），原始列表: {models_to_try}"
                    )
                    error_resp = get_error_response(
                        503,
                        message=(
                            f"所有模型均不可用（输入 {estimated_input_tokens} tokens "
                            f"超过上下文窗口或断路器打开）"
                        ),
                    )
                    error_data = error_resp.model_dump()
                    if request_id:
                        error_data["request_id"] = request_id
                    yield (
                        f"event: error\ndata: "
                        f"{json.dumps(error_data, ensure_ascii=False)}\n\n"
                    )
                    return
                if len(filtered) < len(models_to_try):
                    bound_logger.warning(
                        f"模型列表已过滤: {models_to_try} -> {filtered}"
                    )
                models_to_try = filtered

            last_error = None

            for model_idx, model_to_try in enumerate(models_to_try):
                openai_request.model = model_to_try

                # 追踪当前模型已尝试过403的key索引
                tried_key_indices: set[int] = set()

                for attempt in range(2):  # 流式每模型最多2次尝试
                    try:
                        if model_idx > 0 and attempt == 0:
                            bound_logger.info(
                                f"流式请求回退模型: {model_to_try}"
                            )

                        # 对 fallback 模型使用更短的超时（30s），快速失败
                        stream_timeout: float | None = None
                        if model_idx > 0:
                            stream_timeout = min(stream_client.timeout, 30.0)

                        # 为每次尝试创建独立的流数据源
                        async def _make_openai_stream():
                            # 用 aclosing 显式关闭上游生成器，避免依赖 asyncio 的
                            # asyncgen finalizer 触发 "aclose(): asynchronous generator
                            # is already running" 竞态
                            async with aclosing(
                                stream_client.send_streaming_request(
                                    openai_request, request_id=request_id, timeout=stream_timeout
                                )
                            ) as openai_stream:
                                try:
                                    async for chunk in openai_stream:
                                        if chunk is not None:
                                            yield f"{chunk}\n\n"
                                except GeneratorExit:
                                    return
                                except RuntimeError as e:
                                    msg = str(e)
                                    if "aclose" in msg or "already running" in msg:
                                        return
                                    raise

                        # 流式转换并yield事件（aclosing 确保转换器及上游链被有序关闭）
                        async with aclosing(
                            self.response_converter.convert_openai_stream_to_anthropic_stream(
                                _make_openai_stream(),
                                model=request.model,
                                request_id=request_id,
                            )
                        ) as anthropic_stream:
                            async for anthropic_event in anthropic_stream:
                                # 提取 usage 信息
                                try:
                                    event_data = json.loads(
                                        anthropic_event.split("data: ")[1]
                                    )
                                    if "usage" in event_data and event_data["usage"]:
                                        usage = event_data["usage"]
                                        total_tokens_used = (
                                            (usage.get("input_tokens") or 0)
                                            + (usage.get("output_tokens") or 0)
                                        )
                                except (IndexError, json.JSONDecodeError, KeyError):
                                    pass

                                yield anthropic_event

                        # 流式成功完成
                        bound_logger.info("流式转换完成")
                        if current_key and total_tokens_used > 0:
                            current_key.mark_success(tokens_used=total_tokens_used)
                        return

                    except GeneratorExit:
                        raise  # 客户端断开，向上传播
                    except RuntimeError as e:
                        msg = str(e)
                        if "aclose" in msg or "already running" in msg:
                            return  # 客户端断开导致的aclose
                        raise

                    except OpenAIClientError as e:
                        last_error = e
                        is_permission_error = e.status_code == 403
                        is_client_error = (
                            (400 <= (e.status_code or 0) < 500)
                            and e.status_code not in (403, 429)
                        )

                        bound_logger.warning(
                            f"流式请求失败 (模型: {model_to_try}, "
                            f"尝试 {attempt + 1}/2) - Status: {e.status_code}"
                        )

                        if is_permission_error:
                            key_name = current_key.name if current_key else "unknown"
                            if current_key is not None:
                                tried_key_indices.add(current_key.index)

                            # 先尝试换一个key，所有key都无权使用时才换模型
                            if self.key_rotator and current_key is not None:
                                total_keys = len(self.key_rotator.api_keys)
                                found_new_key = False
                                for offset in range(1, total_keys):
                                    next_idx = (current_key.index + offset) % total_keys
                                    next_key_info = self.key_rotator.api_keys[next_idx]
                                    if next_key_info.is_available() and next_idx not in tried_key_indices:
                                        async with self.key_rotator._lock:
                                            self.key_rotator.current_key_index = next_idx
                                            if session_id:
                                                self.key_rotator.session_key_mapping[session_id] = next_idx
                                        stream_client = self._clients[next_idx]
                                        current_key = next_key_info
                                        found_new_key = True
                                        bound_logger.warning(
                                            f"检测到403权限错误，模型 {model_to_try} 无权限 (Key: [{key_name}])，"
                                            f"切换到 Key: [{current_key.name}] 重试同一模型"
                                        )
                                        break

                                if found_new_key:
                                    continue  # 用新key重试同一模型

                            # 所有key都无权使用此模型，尝试下一个模型
                            bound_logger.warning(
                                f"检测到403权限错误，模型 {model_to_try} 所有Key均无权限，尝试下一个模型"
                            )
                            break  # 尝试下一个模型

                        if is_client_error:
                            bound_logger.warning(
                                f"客户端错误 {e.status_code}，不重试"
                            )
                            error_resp = get_error_response(
                                e.status_code or 400,
                                message=str(e.error_response.error),
                            )
                            error_data = error_resp.model_dump()
                            if request_id:
                                error_data["request_id"] = request_id
                            yield (
                                f"event: error\ndata: "
                                f"{json.dumps(error_data, ensure_ascii=False)}\n\n"
                            )
                            return

                        # 504超时：记录断路器，最多重试1次，退避与超时值成比例
                        if e.status_code == 504:
                            await _record_model_timeout(model_to_try)
                            if attempt < 1:
                                effective_timeout = stream_timeout or stream_client.timeout
                                backoff = min(30.0, effective_timeout / 2)
                                bound_logger.warning(
                                    f"流式请求超时，{backoff:.0f}s 后退避重试 (模型: {model_to_try}, "
                                    f"尝试 {attempt + 1}/2)"
                                )
                                await asyncio.sleep(backoff)
                                continue
                            else:
                                bound_logger.warning(
                                    f"流式请求超时重试已用尽 (模型: {model_to_try})，"
                                    f"尝试下一个模型"
                                )
                                break

                        # 其他服务端错误：换key重试
                        if self.key_rotator and attempt < 1:
                            await self._handle_client_error_with_retry(
                                e,
                                request_id=request_id or "default",
                                session_id=session_id,
                            )
                            stream_client, current_key = await self._get_client_for_session(
                                session_id, request.model
                            )
                            if current_key and current_key.model:
                                openai_request.model = current_key.model
                                bound_logger.info(
                                    f"新 key [{current_key.name}] 有固定模型，"
                                    f"切换至: {current_key.model}"
                                )
                                break
                            continue
                        break

                    except (ValidationError, ValueError) as e:
                        error_detail = (
                            e.errors() if hasattr(e, "errors") else str(e)
                        )
                        bound_logger.warning(
                            f"流式请求验证失败 - Errors: {error_detail}"
                        )
                        error_resp = get_error_response(
                            422, message=str(error_detail)
                        )
                        error_data = error_resp.model_dump()
                        if request_id:
                            error_data["request_id"] = request_id
                        yield (
                            f"event: error\ndata: "
                            f"{json.dumps(error_data, ensure_ascii=False)}\n\n"
                        )
                        return

                    except json.JSONDecodeError as e:
                        bound_logger.exception(
                            f"流式模式JSON解析错误 - Error: {str(e)}"
                        )
                        error_resp = get_error_response(
                            502,
                            message="流式响应中发现无效JSON格式",
                            details={
                                "json_error": str(e),
                                "request_id": request_id,
                            },
                        )
                        error_data = error_resp.model_dump()
                        if request_id:
                            error_data["request_id"] = request_id
                        yield (
                            f"event: error\ndata: "
                            f"{json.dumps(error_data, ensure_ascii=False)}\n\n"
                        )
                        return

            # 所有重试已用尽
            error_msg = (
                f"流式请求所有模型尝试均失败: {', '.join(models_to_try)}"
            )
            bound_logger.error(error_msg)
            error_resp = get_error_response(
                last_error.status_code if last_error else 502,
                message=error_msg,
            )
            error_data = error_resp.model_dump()
            if request_id:
                error_data["request_id"] = request_id
            yield (
                f"event: error\ndata: "
                f"{json.dumps(error_data, ensure_ascii=False)}\n\n"
            )

        except GeneratorExit:
            raise  # 客户端断开
        except RuntimeError as e:
            if "aclose" in str(e) or "already running" in str(e):
                return  # 客户端断开导致的aclose
            raise
        except Exception as e:
            bound_logger.exception(
                f"流式请求处理错误 - Type: {type(e).__name__}, Error: {str(e)}"
            )
            error_resp = get_error_response(500, message=str(e))
            error_data = error_resp.model_dump()
            if request_id:
                error_data["request_id"] = request_id
            yield (
                f"event: error\ndata: "
                f"{json.dumps(error_data, ensure_ascii=False)}\n\n"
            )


def _normalize_request_body(body: dict, bound_logger) -> dict:
    """
    预处理请求体，使其符合 Anthropic API 规范。

    处理以下兼容性场景：
    1. 从 messages 数组中提取 role="system" 的消息，合并到顶层 system 参数
    2. (未来可扩展更多规范化逻辑)

    Anthropic API 规范中 system 是顶层参数，不在 messages 数组中。
    但某些客户端（如 Claude Code）可能以 OpenAI 格式发送 system 消息，
    即将其放在 messages 数组里。此函数自动完成格式规范化。
    """
    messages = body.get("messages", [])
    if not messages:
        return body

    # 分离 system 消息和普通消息
    system_messages = []
    normal_messages = []

    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "system":
            system_messages.append(msg)
        else:
            normal_messages.append(msg)

    if not system_messages:
        return body

    # 提取 system 消息的文本内容
    system_texts = []
    for msg in system_messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            system_texts.append(content)
        elif isinstance(content, list):
            # 处理 content 为列表的情况（如 [{"type": "text", "text": "..."}]）
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    system_texts.append(block.get("text", ""))

    extracted_system = "\n".join(system_texts) if system_texts else None

    # 合并到现有的 system 参数
    existing_system = body.get("system")
    if existing_system:
        if isinstance(existing_system, str):
            combined = f"{existing_system}\n{extracted_system}" if extracted_system else existing_system
        elif isinstance(existing_system, list):
            # system 是 AnthropicSystemMessage 列表格式
            if extracted_system:
                combined = list(existing_system) + [{"type": "text", "text": extracted_system}]
            else:
                combined = existing_system
        else:
            combined = extracted_system
    else:
        combined = extracted_system

    # 更新 body
    body = body.copy()
    body["messages"] = normal_messages
    if combined:
        body["system"] = combined

    if system_texts:
        bound_logger.info(
            f"从 messages 数组中提取了 {len(system_messages)} 条 system 消息，"
            f"已合并到顶层 system 参数"
        )

    return body


@router.post("/messages")
async def messages_endpoint(request: Request, background_tasks: BackgroundTasks):
    """
    Anthropic /v1/messages 端点

    这个端点实现了Anthropic原生messages API的主要功能：
    - 接受Anthropic格式的请求
    - 转换为OpenAI格式发送到后端
    - 返回Anthropic格式的响应
    """
    # 从应用状态获取消息处理器（已由main.py在启动时初始化）
    handler: MessagesHandler = request.app.state.messages_handler

    # 获取请求ID（由中间件生成，如果启用的话）
    from src.common.logging import (
        get_logger_with_request_id,
        get_request_id_from_request,
    )

    request_id = get_request_id_from_request(request)
    bound_logger = get_logger_with_request_id(request_id)

    # 记录请求
    client_ip = request.client.host if request.client else "unknown"
    bound_logger.info(
        f"收到Anthropic请求 - Method: {request.method}, URL: {str(request.url)}, IP: {client_ip}"
    )

    try:
        # 解析请求体
        body = await request.json()

        # 预处理：从 messages 数组中提取 system 角色的消息，
        # 并将其合并到顶层的 system 参数中（符合 Anthropic API 规范）
        body = _normalize_request_body(body, bound_logger)

        # 记录请求
        log_body = body.copy()
        log_body["tools"] = []
        bound_logger.debug(
            f"Anthropic请求体 - Model: {body.get('model', 'unknown')}, Messages: {len(body.get('messages', []))}, Stream: {body.get('stream', False)}\n{json.dumps(log_body, ensure_ascii=False, indent=2)}"
        )

        anthropic_request = AnthropicRequest(**body)

        # 使用 request_id 作为 session_id，用于 key 轮转策略的并发分配
        # balanced 策略会优先选择使用次数最少的 key，自动实现多 agent 并行负载均衡
        session_id = request_id or client_ip

        # 记录清理后的请求信息（移除敏感信息）
        # safe_body = sanitize_for_logging(body)
        # logger.debug("请求已清理", request_body=safe_body)

        # 根据请求类型处理响应
        if anthropic_request.stream:
            # 流式响应 - 优化配置确保真正的流式效果
            async def stream_wrapper():
                """包装器确保流式响应的立即传输"""
                try:
                    # aclosing 确保客户端断开或正常结束时，整条生成器链被有序关闭，
                    # 而非依赖 asyncio 的 asyncgen finalizer（后者会触发
                    # "aclose(): asynchronous generator is already running" 竞态）
                    async with aclosing(
                        handler.process_stream_message(
                            anthropic_request, request_id=request_id, session_id=session_id
                        )
                    ) as stream:
                        async for chunk in stream:
                            # 立即传输每个chunk，不缓冲
                            # chunk已经是完整的SSE格式字符串，编码后返回
                            yield chunk.encode("utf-8")
                            # 强制刷新缓冲区（在某些环境中有效）
                            await asyncio.sleep(0)
                except GeneratorExit:
                    # 客户端断开连接，正常清理
                    return
                except RuntimeError as e:
                    msg = str(e)
                    if "aclose" in msg or "already running" in msg:
                        # Python 3.11+: 在运行中的async generator上调用aclose()
                        # 会触发此RuntimeError。客户端断开时可能发生，正常清理。
                        return
                    raise
                except Exception as e:
                    # 如果流式处理出错，记录完整错误并发送错误事件
                    bound_logger.exception(f"流式处理出错 - Error: {str(e)}")
                    error_data = {"error": str(e)}
                    if request_id:
                        error_data["request_id"] = request_id
                    error_event = f"event: error\ndata: {json.dumps(error_data)}\n\n"
                    yield error_event.encode("utf-8")

            return StreamingResponse(
                stream_wrapper(),
                media_type="text/event-stream; charset=utf-8",
                headers={
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",  # 禁用nginx缓冲
                    "X-Content-Type-Options": "nosniff",
                    "Transfer-Encoding": "chunked",
                    "Access-Control-Allow-Origin": "*",  # CORS支持
                    "Access-Control-Allow-Headers": "*",
                    "Access-Control-Allow-Methods": "*",
                    "X-Proxy-Buffering": "no",  # 禁用代理缓冲
                    "Buffering": "no",  # 禁用缓冲
                },
            )
        else:
            # 非流式响应
            response = await handler.process_message(
                anthropic_request, request_id=request_id, session_id=session_id
            )
            json_response = JSONResponse(content=response.model_dump(exclude_none=True))
            if request_id:
                json_response.headers["X-Request-ID"] = request_id
            return json_response

    except ValidationError as e:
        bound_logger.warning(f"请求验证失败 - Errors: {e.errors()}")
        error_response = get_error_response(
            422, details={"validation_errors": e.errors()}
        )
        error_detail = error_response.model_dump()
        error_detail["request_id"] = request_id
        raise HTTPException(status_code=422, detail=error_detail)

    except json.JSONDecodeError as e:
        bound_logger.warning(f"请求中的JSON格式错误 - Error: {str(e)}")
        error_response = get_error_response(400, message="无效的JSON格式")
        error_detail = error_response.model_dump()
        error_detail["request_id"] = request_id
        raise HTTPException(status_code=400, detail=error_detail)

    except Exception as e:
        # 检查是否为HTTPException，避免重复记录已处理的错误
        if isinstance(e, HTTPException):
            # HTTPException已经在内层处理过，直接重新抛出
            raise e

        bound_logger.exception(
            f"在messages端点发生意外错误 - Type: {type(e).__name__}, Error: {str(e)}"
        )
        error_response = get_error_response(500, message=str(e))
        error_detail = error_response.model_dump()
        error_detail["request_id"] = request_id
        raise HTTPException(status_code=500, detail=error_detail)
