"""
OpenAI-to-Claude请求转换器

该模块提供将Anthropic格式请求转换为OpenAI格式的功能。
"""

import json
import re
from typing import Any

from fastapi import HTTPException
from loguru import logger

from src.common.token_counter import TokenCounter
from src.models.anthropic import (
    AnthropicMessage,
    AnthropicRequest,
    AnthropicSystemMessage,
    AnthropicToolDefinition,
)
from src.models.openai import (
    OpenAIMessage,
    OpenAIRequest,
    OpenAITool,
    OpenAIToolFunction,
)

# 全局缓存配置对象


def sanitize_tool_call_id(original_id: str) -> str:
    """清理工具调用ID，使其符合OpenAI API的格式要求

    OpenAI要求tool_call.id匹配模式^[a-zA-Z0-9_-]+
    如果原始ID包含非法字符，则生成一个新的符合要求的ID

    Args:
        original_id: 原始工具调用ID

    Returns:
        清理后的或新生成的工具调用ID
    """
    if not original_id:
        import time
        return f"tool_{int(time.time() * 1000)}"

    # 检查ID是否已经符合格式要求
    import re
    if re.match(r'^[a-zA-Z0-9_-]+$', original_id):
        return original_id

    # 如果不符合格式，生成一个新的ID
    # 使用哈希确保相同原始ID总是生成相同的新ID（保持一致性）
    import hashlib
    import time

    # 使用原始ID的哈希值+时间戳生成唯一但确定的新ID
    hash_part = hashlib.md5(original_id.encode()).hexdigest()[:8]
    return f"tool_{hash_part}"


class AnthropicToOpenAIConverter:
    """将Anthropic请求转换为OpenAI格式"""

    @staticmethod
    async def get_target_model(
        anthropic_request: AnthropicRequest, request_id: str = None
    ) -> str:
        """
        根据请求选择目标模型

        规则：
        1. 如果客户端指定了具体的模型名（非通用 Claude 名称），则直接使用该模型
        2. 如果是通用的 Claude 模型名（如 claude-3-5-sonnet、claude-haiku），则使用智能路由
        3. 智能路由会根据请求内容（工具使用、thinking、token 数量等）选择合适的模型

        Args:
            anthropic_request: Anthropic特定请求对象
            request_id: 请求ID用于日志追踪

        Returns:
            选定的目标模型ID
        """
        original_model = anthropic_request.model

        # 使用全局缓存的配置对象
        from src.config.settings import get_config
        config = await get_config()

        # 获取绑定了请求ID的logger
        from src.common.logging import get_logger_with_request_id
        bound_logger = get_logger_with_request_id(request_id)

        # 模型别名替换：将不支持的模型名映射为可用模型
        if original_model and original_model in config.models.model_aliases:
            alias = config.models.model_aliases[original_model]
            bound_logger.info(
                f"模型别名替换: {original_model} -> {alias}"
            )
            original_model = alias

        # 规范化模型名称：移除日期后缀（如 claude-sonnet-4-5-20250929 -> claude-sonnet-4-5）
        normalized_model = original_model
        if original_model:
            # 匹配模型名称末尾的日期后缀模式：-YYYYMMDD
            date_suffix_pattern = r'-\d{8}$'
            if re.search(date_suffix_pattern, original_model):
                normalized_model = re.sub(date_suffix_pattern, '', original_model)
                bound_logger.info(
                    f"检测到带日期后缀的模型名称: {original_model} -> 规范化为: {normalized_model}"
                )
                # 使用规范化后的模型名称继续处理
                original_model = normalized_model

        # 如果模型包含逗号，直接返回原模型（保留复杂性）
        if original_model and "," in original_model:
            bound_logger.info(f"使用客户端指定的复合模型: {original_model}")
            return original_model

        # 检查是否为通用的 Claude 模型名称（需要智能路由）
        # 通用名称示例: claude-3-5-sonnet, claude-3-haiku, claude-sonnet-4, etc.
        # 注意：我们只检查通用的 Claude API 模型名，不包括具体的版本号（如 claude-sonnet-4-5）
        is_generic_claude_model = original_model and (
            # 匹配类似 claude-3-5-sonnet, claude-3-opus, claude-3-haiku 的模式（带版本号 2/3/4）
            (bool(re.search(r'claude-[234]-\d*-?(sonnet|haiku|opus)', original_model.lower())))
            or
            # 匹配不带具体版本的通用名称，如 claude-sonnet, claude-haiku, claude-opus
            (bool(re.search(r'^claude-(sonnet|haiku|opus)$', original_model.lower())))
        )

        # 如果不是通用 Claude 模型名称，直接使用客户端指定的模型（不进行智能路由）
        if not is_generic_claude_model and original_model:
            bound_logger.info(f"使用客户端直接指定的模型: {original_model}")
            return original_model

        # 使用智能路由选择模型
        if not config.models.default:
            bound_logger.warning(f"配置中未设置默认模型，使用原始模型: {original_model}")
            return original_model

        resolved_model = config.models.default
        routing_reason = "默认模型"

        # 基于模型名称的路由
        if original_model:
            if "haiku" in original_model.lower():
                resolved_model = config.models.small
                routing_reason = "Haiku -> 小型模型"
            elif "sonnet" in original_model.lower():
                resolved_model = config.models.default
                routing_reason = "Sonnet -> 默认模型"

        # 如果有tools定义，使用tool模型
        # if anthropic_request.tools and len(anthropic_request.tools) > 0:
        #     resolved_model = config.models.tool
        #     routing_reason = "工具使用 -> 工具模型"

        # 如果thinking为enabled，使用think模型
        if (
            anthropic_request.thinking is not None
            and isinstance(anthropic_request.thinking, dict)
            and anthropic_request.thinking.get("type") == "enabled"
        ):
            resolved_model = config.models.think
            routing_reason = "思考模式 -> 推理模型"

        # 计算token数量
        token_counter = TokenCounter()
        total_tokens = await token_counter.count_tokens(
            anthropic_request.messages,
            anthropic_request.system,
            anthropic_request.tools,
        )
        if total_tokens > 1000 * 100:
            resolved_model = config.models.long_context
            routing_reason = f"长上下文({total_tokens} tokens) -> 长上下文模型"

        # 缓存token数量用于后续响应处理
        if request_id:
            from src.common.token_cache import cache_tokens
            cache_tokens(request_id, total_tokens)

        # 检查 web_search 工具
        if anthropic_request.tools:
            has_web_search = any(
                tool.type and "web_search" in tool.type
                for tool in anthropic_request.tools
            )
            if has_web_search:
                resolved_model = config.models.web_search
                routing_reason = "网页搜索 -> 搜索模型"
                if "gemini" not in resolved_model:
                    raise HTTPException(
                        status_code=400,
                        detail="Web search is only supported with Gemini models",
                    )

        bound_logger.info(
            f"智能路由: {original_model} -> {resolved_model} (原因: {routing_reason})"
        )
        return resolved_model

    @staticmethod
    async def convert_anthropic_to_openai(
        anthropic_request: AnthropicRequest,
        request_id: str = None,
    ) -> OpenAIRequest:
        """
        将Anthropic请求转换为OpenAI格式请求

        Args:
            anthropic_request: Anthropic格式的请求
            request_id: 请求ID用于日志追踪

        Returns:
            转换后的OpenAI格式请求
        """
        # 获取绑定了请求ID的logger
        from src.common.logging import get_logger_with_request_id

        bound_logger = get_logger_with_request_id(request_id)

        # 动态选择目标模型
        target_model = await AnthropicToOpenAIConverter.get_target_model(
            anthropic_request, request_id
        )

        bound_logger.debug(
            "将Anthropic请求转换为OpenAI格式",
            extra={
                "source_model": anthropic_request.model,
                "target_model": target_model,
                "message_count": len(anthropic_request.messages),
                "has_tools": anthropic_request.tools is not None,
                "has_system": anthropic_request.system is not None,
                "thinking": anthropic_request.thinking,
            },
        )

        # 转换消息列表
        messages = AnthropicToOpenAIConverter._convert_messages(anthropic_request)

        # 如果启用了thinking模式，确保所有带tool_calls的assistant消息都有reasoning_content
        # 某些上游API要求当thinking启用时，所有使用工具的assistant消息都必须包含reasoning_content
        think_enabled = (
            anthropic_request.thinking is not None
            and isinstance(anthropic_request.thinking, dict)
            and anthropic_request.thinking.get("type") == "enabled"
        )

        if think_enabled:
            messages = AnthropicToOpenAIConverter._ensure_reasoning_content_for_tool_calls(
                messages
            )

        # 提取system提示
        system_prompt = AnthropicToOpenAIConverter._extract_system_prompt(
            anthropic_request.system
        )

        # 获取配置中的参数覆盖设置（异步获取）
        from src.config.settings import get_config

        config = await get_config()

        # 可选：将 system prompt 注入到 messages 开头（作为 role=system 消息）。
        # llama.cpp / llama-server 等后端不识别请求顶层的 `system` 字段，只认 messages
        # 数组里的 system 消息。开启 inject_system_to_messages 后，system 会同时存在于
        # 顶层字段（兼容支持它的后端）和 messages 开头（兼容不支持的后端）。
        if system_prompt and getattr(config, "inject_system_to_messages", False):
            from src.models.openai import OpenAIMessage as _OpenAIMessage

            messages = [
                _OpenAIMessage(role="system", content=system_prompt),
                *messages,
            ]
            bound_logger.info(
                "已将 system prompt 注入 messages 开头 (inject_system_to_messages=True)"
            )

        # 可选：精简上下文 —— 移除 <system-reminder> 噪音、截断 tool result、限制消息轮数
        if getattr(config, "minimize_context", False):
            messages = AnthropicToOpenAIConverter._minimize_messages(messages, bound_logger)

        # 转换工具定义
        tools = AnthropicToOpenAIConverter._convert_tools(anthropic_request.tools)

        overrides = config.parameter_overrides

        # 应用参数覆盖逻辑（配置覆盖客户端请求参数）
        final_max_tokens = (
            overrides.max_tokens
            if overrides.max_tokens is not None
            else anthropic_request.max_tokens
        )
        final_temperature = (
            overrides.temperature
            if overrides.temperature is not None
            else anthropic_request.temperature
        )
        final_top_p = (
            overrides.top_p if overrides.top_p is not None else anthropic_request.top_p
        )
        final_top_k = (
            overrides.top_k if overrides.top_k is not None else anthropic_request.top_k
        )

        # 记录参数覆盖情况
        overridden_params = []
        if overrides.max_tokens is not None:
            overridden_params.append(
                f"max_tokens: {anthropic_request.max_tokens} -> {final_max_tokens}"
            )
        if overrides.temperature is not None:
            overridden_params.append(
                f"temperature: {anthropic_request.temperature} -> {final_temperature}"
            )
        if overrides.top_p is not None:
            overridden_params.append(
                f"top_p: {anthropic_request.top_p} -> {final_top_p}"
            )
        if overrides.top_k is not None:
            overridden_params.append(
                f"top_k: {anthropic_request.top_k} -> {final_top_k}"
            )

        if overridden_params:
            bound_logger.debug(f"应用参数覆盖: {', '.join(overridden_params)}")

        # 构建OpenAI请求
        openai_request = OpenAIRequest(
            model=target_model,
            system=system_prompt,
            messages=messages,
            max_tokens=final_max_tokens,
            temperature=final_temperature,
            top_p=final_top_p,
            top_k=final_top_k,
            stream=anthropic_request.stream,
            stop=anthropic_request.stop_sequences,
            tools=tools,
            tool_choice=AnthropicToOpenAIConverter._convert_tool_choice(
                anthropic_request.tool_choice
            ),
            think=think_enabled,
            # frequency_penalty=None,  # Anthropic没有直接对应的参数
            # presence_penalty=None,  # Anthropic没有直接对应的参数
            # logprobs=False,  # Anthropic默认不返回logprobs
            # n=1,  # Anthropic默认只生成一个响应
        )

        bound_logger.info(
            f"模型转换完成 - Anthropic: {anthropic_request.model} -> OpenAI: {openai_request.model}"
        )
        log_openai_request = openai_request.model_copy()
        log_openai_request.tools = []
        bound_logger.debug(
            f"OpenAI 请求体: {log_openai_request.model_dump_json(exclude_none=True)}"
        )
        return openai_request

    @staticmethod
    def _minimize_messages(messages: list, bound_logger) -> list:
        """精简上下文：删除冗余内容，帮助 12B 级别模型更稳定地处理长上下文。

        策略：
        1. 从所有消息的文本内容中移除 <system-reminder>...</system-reminder> 块
        2. 保留第一条 system 消息 + 最近 N 条消息
        3. 截断过长的 tool result 消息

        Args:
            messages: OpenAI 格式消息列表
            bound_logger: 日志器

        Returns:
            精简后的消息列表
        """
        MAX_MESSAGES = 8
        MAX_TOOL_RESULT_CHARS = 2000

        result = []

        # 第一步：从所有消息中移除 <system-reminder> 噪音
        reminder_pattern = re.compile(r'<system-reminder>.*?</system-reminder>', re.DOTALL)

        for msg in messages:
            if isinstance(msg.content, str):
                cleaned = reminder_pattern.sub('', msg.content).strip()
                if not cleaned and msg.role == "tool":
                    continue
                msg.content = cleaned
            result.append(msg)

        # 第二步：截断过长的 tool result
        for msg in result:
            if msg.role == "tool" and isinstance(msg.content, str) and len(msg.content) > MAX_TOOL_RESULT_CHARS:
                msg.content = msg.content[:MAX_TOOL_RESULT_CHARS] + "\n...(truncated by proxy)"
                bound_logger.debug(f"截断 tool result 到 {MAX_TOOL_RESULT_CHARS} 字符")

        # 第三步：限制消息数量，保留开头 system + 最近若干条
        if len(result) > MAX_MESSAGES:
            first = result[:1]
            last = result[-(MAX_MESSAGES - 1):]
            result = first + last
            bound_logger.info(
                f"精简上下文: {len(messages)} 条 → {len(result)} 条 "
                f"(移除 {len(messages) - len(result)} 条旧消息)"
            )

        return result

    @staticmethod
    def _convert_messages(
        anthropic_request: AnthropicRequest,
    ) -> list[OpenAIMessage]:
        """
        将Anthropic消息列表转换为OpenAI消息格式

        Args:
            anthropic_request: Anthropic请求

        Returns:
            OpenAI格式的消息列表（不包括system消息，因为system应该在request级别）
        """
        messages = []

        # 注意: 不在这里添加system消息
        # system消息应该通过request的system参数传递，而不是作为消息列表的一部分
        # 这符合某些API（如Anthropic）的期望，即system参数在request级别

        # 转换用户和助手消息
        for anthropic_msg in anthropic_request.messages:
            converted_messages = AnthropicToOpenAIConverter._convert_single_message(
                anthropic_msg
            )
            # _convert_single_message现在可能返回多个消息（当包含tool_result时）
            if isinstance(converted_messages, list):
                messages.extend(converted_messages)
            else:
                messages.append(converted_messages)

        # 过滤不完整的tool_calls序列
        filtered_messages = AnthropicToOpenAIConverter._filter_incomplete_tool_calls(
            messages
        )

        return filtered_messages

    @staticmethod
    def _extract_system_prompt(
        system: str | list[AnthropicSystemMessage] | None,
    ) -> str | None:
        """
        从Anthropic system字段提取系统提示（不作为消息，而是request参数）

        Args:
            system: Anthropic的system字段

        Returns:
            系统提示字符串或None
        """
        if not system:
            return None

        if isinstance(system, str):
            # 字符串格式的system提示
            return system
        elif isinstance(system, list):
            # 列表格式的system提示 - 拼接所有文本
            texts = []
            for system_msg in system:
                if isinstance(system_msg, dict):
                    texts.append(system_msg.get("text", ""))
                elif hasattr(system_msg, "text"):
                    texts.append(system_msg.text)
            return "\n".join(texts) if texts else None

        return None

    @staticmethod
    def _convert_system_message(
        system: str | list[AnthropicSystemMessage],
    ) -> list[OpenAIMessage]:
        """
        将Anthropic system字段转换为OpenAI system消息

        Args:
            system: Anthropic的system字段

        Returns:
            OpenAI格式的system消息列表
        """
        system_messages = []

        if isinstance(system, str):
            # 字符串格式的system提示
            system_messages.append(OpenAIMessage(role="system", content=system))
        elif isinstance(system, list):
            # 列表格式的system提示
            for system_msg in system:
                system_messages.append(
                    OpenAIMessage(role="system", content=system_msg.text)
                )

        return system_messages

    @staticmethod
    def _convert_single_message(
        anthropic_msg: AnthropicMessage,
    ) -> OpenAIMessage | list[OpenAIMessage]:
        """
        转换单个Anthropic消息为OpenAI格式

        Args:
            anthropic_msg: 单个Anthropic消息

        Returns:
            OpenAI格式的消息
        """
        if not anthropic_msg.content:
            raise ValueError("Anthropic消息内容不能为空")

        # 处理内容转换
        if isinstance(anthropic_msg.content, str):
            # 纯文本内容
            return OpenAIMessage(role=anthropic_msg.role, content=anthropic_msg.content)
        elif isinstance(anthropic_msg.content, list):
            # 复杂内容（包括工具调用等）
            content_parts = []
            tool_calls = []
            tool_results = []
            reasoning_content = ""  # 用于收集推理内容

            for content_block in anthropic_msg.content:
                if isinstance(content_block, dict):
                    # 处理字典格式的内容块
                    if content_block.get("type") == "tool_use":
                        # 将tool_use转换为OpenAI的tool_calls格式
                        original_id = content_block.get("id", "")
                        tool_call = {
                            "id": sanitize_tool_call_id(original_id),
                            "type": "function",
                            "function": {
                                "name": content_block.get("name", ""),
                                "arguments": json.dumps(
                                    content_block.get("input", {}), ensure_ascii=False
                                ),
                            },
                        }
                        tool_calls.append(tool_call)
                    elif content_block.get("type") == "tool_result":
                        # 收集tool_result，稍后转换为独立的tool消息
                        tool_result_content = content_block.get("content", "")
                        if isinstance(tool_result_content, list):
                            tool_result_content = json.dumps(
                                tool_result_content, ensure_ascii=False
                            )
                        original_id = content_block.get("tool_use_id", "")
                        tool_results.append(
                            {
                                "tool_call_id": sanitize_tool_call_id(original_id),
                                "content": tool_result_content,
                            }
                        )
                    elif content_block.get("type") == "thinking":
                        # 收集思考内容，作为reasoning_content用于有tool_calls的消息
                        thinking_text = content_block.get("thinking", "")
                        if thinking_text:
                            reasoning_content += thinking_text
                    elif content_block.get("type") in ["text", "image_url"]:
                        # 只保留OpenAI支持的内容类型
                        content_parts.append(content_block)
                elif hasattr(content_block, "type"):
                    # 处理Pydantic模型对象
                    if content_block.type == "tool_use":
                        # 将tool_use转换为OpenAI的tool_calls格式
                        original_id = getattr(content_block, "id", "")
                        tool_call = {
                            "id": sanitize_tool_call_id(original_id),
                            "type": "function",
                            "function": {
                                "name": getattr(content_block, "name", ""),
                                "arguments": json.dumps(
                                    getattr(content_block, "input", {}),
                                    ensure_ascii=False,
                                ),
                            },
                        }
                        tool_calls.append(tool_call)
                    elif content_block.type == "tool_result":
                        # 收集tool_result，稍后转换为独立的tool消息
                        tool_result_content = getattr(content_block, "content", "")
                        if isinstance(tool_result_content, list):
                            tool_result_content = json.dumps(
                                tool_result_content, ensure_ascii=False
                            )
                        original_id = getattr(content_block, "tool_use_id", "")
                        tool_results.append(
                            {
                                "tool_call_id": sanitize_tool_call_id(original_id),
                                "content": tool_result_content,
                            }
                        )
                    elif content_block.type == "thinking":
                        # 收集思考内容，作为reasoning_content用于有tool_calls的消息
                        thinking_text = getattr(content_block, "thinking", "")
                        if thinking_text:
                            reasoning_content += thinking_text
                    elif content_block.type in ["text", "image_url"]:
                        # 只保留OpenAI支持的内容类型
                        content_parts.append(content_block.model_dump())
                else:
                    # 简单文本内容
                    content_parts.append({"type": "text", "text": str(content_block)})

            # 如果有tool_result，需要返回多个消息
            if tool_results:
                messages = []

                # 首先创建主消息（如果有非tool_result内容）
                if content_parts or tool_calls:
                    content = None
                    if content_parts:
                        # 如果只有一个文本内容，简化为字符串
                        if (
                            len(content_parts) == 1
                            and content_parts[0]
                            and isinstance(content_parts[0], dict)
                            and content_parts[0].get("type") == "text"
                        ):
                            content = content_parts[0]["text"]
                        else:
                            # 多个内容部分，确保格式正确
                            validated_parts = []
                            for part in content_parts:
                                if isinstance(part, dict):
                                    # 确保至少有 type 字段
                                    if "type" not in part:
                                        part = {"type": "text", "text": str(part)}
                                    validated_parts.append(part)
                                else:
                                    # 转换为字典
                                    validated_parts.append({"type": "text", "text": str(part)})
                            content = validated_parts if validated_parts else None

                    # 创建主消息
                    # 如果有tool_calls但没有content，设置content为空字符串（上游API不接受content=null）
                    if content is None and tool_calls:
                        content = ""

                    main_msg = OpenAIMessage(role=anthropic_msg.role, content=content)
                    if tool_calls:
                        main_msg.tool_calls = tool_calls
                        # 如果有tool_calls且收集了思考内容，添加为reasoning_content
                        # 这对于某些需要推理内容的API很重要（如启用了thinking模式的模型）
                        if reasoning_content and reasoning_content.strip():
                            main_msg.reasoning_content = reasoning_content.strip()
                    messages.append(main_msg)

                # 然后为每个tool_result创建独立的tool消息
                for tool_result in tool_results:
                    tool_msg = OpenAIMessage(
                        role="tool",
                        content=tool_result["content"],
                        tool_call_id=tool_result["tool_call_id"],
                    )
                    messages.append(tool_msg)

                return messages
            else:
                # 没有tool_result，返回单个消息
                content = None
                if content_parts:
                    # 如果只有一个文本内容，简化为字符串
                    if (
                        len(content_parts) == 1
                        and content_parts[0]
                        and isinstance(content_parts[0], dict)
                        and content_parts[0].get("type") == "text"
                    ):
                        content = content_parts[0]["text"]
                    else:
                        # 多个内容部分，确保格式正确
                        # content_parts 应该已经是 list[dict]，符合 OpenAIMessageContent 的格式
                        # 但需要确保每个字典都有正确的字段
                        validated_parts = []
                        for part in content_parts:
                            if isinstance(part, dict):
                                # 确保至少有 type 字段
                                if "type" not in part:
                                    # 如果没有 type，默认为 text
                                    part = {"type": "text", "text": str(part)}
                                validated_parts.append(part)
                            else:
                                # 转换为字典
                                validated_parts.append({"type": "text", "text": str(part)})
                        content = validated_parts if validated_parts else None

                # 创建OpenAI消息
                # 如果有tool_calls但没有content，设置content为空字符串（上游API不接受content=null）
                if content is None and tool_calls:
                    content = ""

                openai_msg = OpenAIMessage(role=anthropic_msg.role, content=content)

                # 如果有工具调用，添加到消息中
                if tool_calls:
                    openai_msg.tool_calls = tool_calls
                    # 如果有tool_calls且收集了思考内容，添加为reasoning_content
                    # 这对于某些需要推理内容的API很重要（如启用了thinking模式的模型）
                    if reasoning_content and reasoning_content.strip():
                        openai_msg.reasoning_content = reasoning_content.strip()

                return openai_msg
        else:
            # 默认处理
            return OpenAIMessage(
                role=anthropic_msg.role, content=str(anthropic_msg.content)
            )

    @staticmethod
    def _convert_tools(
        anthropic_tools: list[AnthropicToolDefinition] | None,
    ) -> list[OpenAITool] | None:
        """
        将Anthropic工具定义转换为OpenAI工具格式

        Args:
            anthropic_tools: Anthropic工具定义列表

        Returns:
            OpenAI格式的工具列表或None
        """
        if not anthropic_tools:
            return None

        has_web_search = any(
            tool.type and "web_search" in tool.type for tool in anthropic_tools
        )
        openai_tools = []
        if has_web_search:
            openai_tool = OpenAITool(
                type="function",
                function=OpenAIToolFunction(
                    name="googleSearch",
                ),
            )
            openai_tools.append(openai_tool)
        else:
            for tool in anthropic_tools:
                openai_tool = OpenAITool(
                    type="function",
                    function=OpenAIToolFunction(
                        name=tool.name,
                        description=tool.description,
                        parameters=tool.input_schema,
                    ),
                )
                openai_tools.append(openai_tool)

        return openai_tools if openai_tools else None

    @staticmethod
    def _convert_tool_choice(
        anthropic_tool_choice: str | dict[str, Any] | None,
    ) -> str | dict[str, Any] | None:
        """
        转换Anthropic的tool_choice到OpenAI格式

        Args:
            anthropic_tool_choice: Anthropic的tool_choice配置

        Returns:
            OpenAI格式的tool_choice或None
        """
        if anthropic_tool_choice is None:
            return None

        if isinstance(anthropic_tool_choice, str):
            # 直接映射字符串值
            if anthropic_tool_choice == "any":
                return "required"  # Anthropic的"any"对应OpenAI的"required"
            elif anthropic_tool_choice == "auto":
                return "auto"
            else:
                return anthropic_tool_choice

        elif isinstance(anthropic_tool_choice, dict):
            # 处理复杂配置
            if anthropic_tool_choice.get("type") == "tool":
                tool_name = anthropic_tool_choice.get("name", "")
                if tool_name:
                    return {"type": "function", "function": {"name": tool_name}}

        # 默认返回原始值
        return anthropic_tool_choice

    @staticmethod
    def _filter_incomplete_tool_calls(
        messages: list[OpenAIMessage],
    ) -> list[OpenAIMessage]:
        """过滤不完整的tool_calls序列

        OpenAI要求每个带有tool_calls的assistant消息后面必须跟对应的tool消息。
        此方法会移除没有对应tool消息的assistant消息中的tool_calls序列。
        同时也会移除没有对应assistant消息的独立tool消息。

        改进策略：两遍扫描
        1. 第一遍：建立tool_call_id到assistant消息索引的映射
        2. 第二遍：只保留有完整对应关系的消息

        Args:
            messages: 原始消息列表

        Returns:
            过滤后的消息列表
        """
        if not messages:
            return messages

        # 第一遍：建立所有tool_call的映射关系
        tool_call_to_assistant_index: dict[str, int] = {}
        assistant_tool_calls: dict[int, set[str]] = {}  # assistant_index -> set of tool_call_ids

        for i, msg in enumerate(messages):
            if msg.role == "assistant" and msg.tool_calls:
                tool_call_ids = set()
                for call in msg.tool_calls:
                    call_id = call.get("id")
                    if call_id:
                        tool_call_to_assistant_index[call_id] = i
                        tool_call_ids.add(call_id)
                assistant_tool_calls[i] = tool_call_ids

        # 第二遍：验证每个tool消息是否有对应的assistant消息
        valid_tool_call_ids: set[str] = set()

        for i, msg in enumerate(messages):
            if msg.role == "tool" and msg.tool_call_id:
                # 检查这个tool_call_id是否有对应的assistant消息
                if msg.tool_call_id in tool_call_to_assistant_index:
                    valid_tool_call_ids.add(msg.tool_call_id)

        # 第三遍：构建过滤后的消息列表
        filtered_messages = []
        i = 0

        while i < len(messages):
            current_msg = messages[i]

            # 如果是assistant消息且有tool_calls
            if current_msg.role == "assistant" and current_msg.tool_calls:
                # 检查这个assistant消息的tool_calls是否都有对应的tool消息
                assistant_index = i
                if assistant_index in assistant_tool_calls:
                    tool_call_ids = assistant_tool_calls[assistant_index]

                    # 查找后续的tool消息
                    j = i + 1
                    found_tool_ids = set()
                    while j < len(messages) and messages[j].role == "tool":
                        tool_msg = messages[j]
                        if tool_msg.tool_call_id in tool_call_ids:
                            if tool_msg.tool_call_id in valid_tool_call_ids:
                                found_tool_ids.add(tool_msg.tool_call_id)
                        j += 1

                    # 只有所有tool_calls都有有效对应的tool消息时才保留
                    if found_tool_ids == tool_call_ids:
                        filtered_messages.append(current_msg)
                        # 添加对应的tool消息
                        for k in range(i + 1, j):
                            if messages[k].role == "tool":
                                filtered_messages.append(messages[k])
                        i = j
                    else:
                        # 不完整的序列，跳过
                        logger.debug(
                            f"过滤不完整的tool_calls序列: 期望{len(tool_call_ids)}个tool消息，实际找到{len(found_tool_ids)}个"
                        )
                        i = j
                else:
                    # 没有tool_calls的assistant消息，直接添加
                    filtered_messages.append(current_msg)
                    i += 1

            # 如果是tool消息，需要检查是否有对应的assistant消息
            elif current_msg.role == "tool":
                # 检查tool_call_id是否在有效列表中
                if current_msg.tool_call_id in valid_tool_call_ids:
                    # 这个tool消息有对应的assistant，应该被保留
                    # 但由于上面的逻辑已经添加了配对的tool消息，这里只处理漏掉的情况
                    # 检查是否已经被添加过（避免重复）
                    if not filtered_messages or filtered_messages[-1] != current_msg:
                        # 再检查一下前面是否有对应的assistant消息
                        has_corresponding = False
                        for k in range(len(filtered_messages) - 1, -1, -1):
                            prev_msg = filtered_messages[k]
                            if prev_msg.role == "assistant" and prev_msg.tool_calls:
                                for call in prev_msg.tool_calls:
                                    if call.get("id") == current_msg.tool_call_id:
                                        has_corresponding = True
                                        break
                                if has_corresponding:
                                    break
                            elif prev_msg.role != "tool":
                                break

                        if has_corresponding:
                            filtered_messages.append(current_msg)
                    i += 1
                else:
                    # 没有对应assistant的tool消息，过滤掉
                    logger.debug(
                        f"过滤没有对应assistant消息的独立tool消息: {current_msg.tool_call_id}"
                    )
                    i += 1

            # 普通消息，直接添加
            else:
                filtered_messages.append(current_msg)
                i += 1

        return filtered_messages

    @staticmethod
    def _ensure_reasoning_content_for_tool_calls(
        messages: list[OpenAIMessage],
    ) -> list[OpenAIMessage]:
        """确保所有带tool_calls的assistant消息都有reasoning_content

        当thinking模式启用时，某些上游API要求所有使用工具的assistant消息都必须包含reasoning_content。
        此方法会为缺少reasoning_content的assistant消息添加占位符内容。

        Args:
            messages: OpenAI格式的消息列表

        Returns:
            更新后的消息列表
        """
        updated_messages = []
        for msg in messages:
            # 创建消息的副本以避免修改原始消息
            updated_msg = msg.model_copy()

            # 如果是assistant消息且有tool_calls但没有reasoning_content
            if (
                updated_msg.role == "assistant"
                and updated_msg.tool_calls
                and len(updated_msg.tool_calls) > 0
                and not updated_msg.reasoning_content
            ):
                # 添加占位符reasoning_content
                # 使用空字符串表示"此消息是工具调用，没有单独的推理内容"
                updated_msg.reasoning_content = ""
                logger.debug(
                    f"为assistant消息添加reasoning_content占位符 (tool_calls数量: {len(updated_msg.tool_calls)})"
                )

            updated_messages.append(updated_msg)

        return updated_messages


async def validate_anthropic_request(
    request: AnthropicRequest, request_id: str = None
) -> None:
    """
    验证Anthropic请求的完整性

    Args:
        request: 要验证的Anthropic请求
        request_id: 请求ID用于日志追踪

    Raises:
        ValueError: 如果请求格式不正确
    """
    # 获取绑定了请求ID的logger
    from src.common.logging import get_logger_with_request_id

    bound_logger = get_logger_with_request_id(request_id)

    if not request.model:
        raise ValueError("模型字段不能为空")

    if not request.messages:
        raise ValueError("消息列表不能为空")

    if request.max_tokens <= 0:
        raise ValueError("max_tokens必须是正整数")

    if request.temperature is not None and not (0.0 <= request.temperature <= 1.0):
        raise ValueError("temperature必须在0.0到1.0之间")

    if request.top_p is not None and not (0.0 <= request.top_p <= 1.0):
        raise ValueError("top_p必须在0.0到1.0之间")

    for msg in request.messages:
        if not msg.role or msg.role not in ["user", "assistant"]:
            raise ValueError(f"消息角色必须是'user'或'assistant'，但得到: {msg.role}")

    bound_logger.debug("Anthropic请求验证通过")
