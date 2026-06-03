"""OpenAI Responses API (/v1/responses) endpoint handler.

Converts OpenAI Responses API requests to Chat Completions format,
forwards to upstream, and converts responses back.
"""

import asyncio
import json
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger

from src.core.clients.openai_client import OpenAIClientError, OpenAIServiceClient
from src.core.api_key_rotator import APIKeyRotator
from src.models.openai import OpenAIMessage

router = APIRouter(prefix="/v1", tags=["responses"])


def _convert_responses_input_to_messages(body: dict) -> list[dict]:
    """Convert Responses API 'input' field to Chat Completions 'messages'."""
    messages = []
    inp = body.get("input", [])

    # input can be a string (simple prompt)
    if isinstance(inp, str):
        return [{"role": "user", "content": inp}]

    # input can be a list of items
    for item in inp:
        if isinstance(item, str):
            messages.append({"role": "user", "content": item})
            continue

        role = item.get("role", "")

        if role == "user":
            content = item.get("content", "")
            if isinstance(content, str):
                messages.append({"role": "user", "content": content})
            elif isinstance(content, list):
                # content parts
                parts = []
                for part in content:
                    if isinstance(part, str):
                        parts.append({"type": "text", "text": part})
                    elif isinstance(part, dict):
                        if part.get("type") == "input_text":
                            parts.append({"type": "text", "text": part.get("text", "")})
                        else:
                            parts.append(part)
                messages.append({"role": "user", "content": parts})

        elif role == "assistant":
            content = item.get("content", "")
            if isinstance(content, str):
                messages.append({"role": "assistant", "content": content})
            elif isinstance(content, list):
                text_parts = []
                tool_calls = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "output_text":
                            text_parts.append(part.get("text", ""))
                        elif part.get("type") == "function_call":
                            tool_calls.append({
                                "id": part.get("call_id", part.get("id", f"call_{uuid.uuid4().hex[:24]}")),
                                "type": "function",
                                "function": {
                                    "name": part.get("name", ""),
                                    "arguments": part.get("arguments", "{}"),
                                },
                            })
                msg = {"role": "assistant"}
                if text_parts:
                    msg["content"] = "\n".join(text_parts)
                if tool_calls:
                    msg["tool_calls"] = tool_calls
                messages.append(msg)

        elif role == "system":
            content = item.get("content", "")
            if isinstance(content, list):
                text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "input_text"]
                content = "\n".join(text_parts)
            messages.append({"role": "system", "content": content})

        elif role == "tool":
            content = item.get("content", "")
            if isinstance(content, str):
                pass
            elif isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "tool_result":
                            text_parts.append(str(part.get("output", "")))
                        else:
                            text_parts.append(part.get("text", ""))
                content = "\n".join(text_parts)
            messages.append({
                "role": "tool",
                "tool_call_id": item.get("tool_call_id", item.get("call_id", "")),
                "content": content,
            })

        elif role == "developer":
            content = item.get("content", "")
            if isinstance(content, list):
                text_parts = [p.get("text", "") for p in content if isinstance(p, dict)]
                content = "\n".join(text_parts)
            messages.append({"role": "system", "content": content})

    return messages


def _convert_responses_tools_to_chat_tools(tools: list[dict] | None) -> list[dict] | None:
    """Convert Responses API tools to Chat Completions tools format."""
    if not tools:
        return None

    chat_tools = []
    for tool in tools:
        tool_type = tool.get("type", "")
        if tool_type == "function":
            fn = tool.get("function", {})
            chat_tools.append({
                "type": "function",
                "function": {
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {}),
                },
            })
        elif tool_type == "web_search_preview" or tool_type == "web_search":
            # Skip tools not supported by upstream
            pass
        else:
            # Pass through other tool types
            chat_tools.append(tool)

    return chat_tools if chat_tools else None


def _build_chat_request(body: dict) -> dict:
    """Build a Chat Completions request from Responses API body."""
    messages = _convert_responses_input_to_messages(body)

    # Prepend instructions as system message if present
    instructions = body.get("instructions")
    if instructions:
        messages.insert(0, {"role": "system", "content": instructions})

    req = {
        "model": body.get("model", "gpt-4o"),
        "messages": messages,
        "stream": body.get("stream", False),
    }

    # Optional fields passthrough
    for key in ("temperature", "top_p", "max_tokens", "max_completion_tokens",
                "presence_penalty", "frequency_penalty", "seed", "stop",
                "response_format", "user", "parallel_tool_calls", "tool_choice"):
        if key in body:
            req[key] = body[key]

    # Convert tools
    tools = _convert_responses_tools_to_chat_tools(body.get("tools"))
    if tools:
        req["tools"] = tools

    return req


def _build_responses_output_from_chat(chat_response: dict) -> list[dict]:
    """Convert Chat Completions response choices to Responses API output format."""
    outputs = []
    for choice in chat_response.get("choices", []):
        msg = choice.get("message", {})
        output_item = {
            "type": "message",
            "id": f"msg_{uuid.uuid4().hex[:24]}",
            "role": "assistant",
            "content": [],
        }

        # Text content
        content = msg.get("content")
        if content:
            output_item["content"].append({
                "type": "output_text",
                "text": content,
            })

        # Tool calls
        tool_calls = msg.get("tool_calls", [])
        for tc in tool_calls:
            fn = tc.get("function", {})
            output_item["content"].append({
                "type": "function_call",
                "id": tc.get("id", ""),
                "call_id": tc.get("id", ""),
                "name": fn.get("name", ""),
                "arguments": fn.get("arguments", "{}"),
            })

        if not output_item["content"]:
            output_item["content"].append({"type": "output_text", "text": ""})

        outputs.append(output_item)

    return outputs


def _build_responses_response(chat_response: dict, model: str) -> dict:
    """Build full Responses API response from Chat Completions response."""
    usage = chat_response.get("usage", {})
    return {
        "id": f"resp_{uuid.uuid4().hex[:24]}",
        "object": "response",
        "created_at": int(time.time()),
        "model": chat_response.get("model", model),
        "status": "completed",
        "output": _build_responses_output_from_chat(chat_response),
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
        "metadata": {},
    }


class ResponsesHandler:
    """Handles OpenAI Responses API /v1/responses requests."""

    def __init__(self, config):
        self.config = config
        api_keys_config = config.openai.get_effective_keys()
        if not api_keys_config:
            raise ValueError("No API keys configured")

        if len(api_keys_config) == 1:
            self.key_rotator = None
            self.client = OpenAIServiceClient(
                api_key=api_keys_config[0]["api_key"],
                base_url=api_keys_config[0]["base_url"],
            )
        else:
            self.key_rotator = APIKeyRotator(api_keys_config, strategy="round_robin")
            current_key = self.key_rotator.get_current_key()
            self.client = OpenAIServiceClient(
                api_key=current_key.api_key,
                base_url=current_key.base_url,
            )

    @classmethod
    async def create(cls, config=None):
        if config is None:
            from src.config.settings import get_config
            config = await get_config()
        return cls(config)

    async def _send_with_retry(self, chat_req: dict, request_id: str | None = None, max_retries: int = 3) -> dict:
        """Send chat completions request with key rotation retry."""
        from src.common.logging import get_logger_with_request_id
        bound_logger = get_logger_with_request_id(request_id)
        last_error = None

        for attempt in range(max_retries):
            try:
                response = await self.client.send_raw_request(chat_req, request_id=request_id)
                if self.key_rotator:
                    usage = response.get("usage", {})
                    total = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
                    self.key_rotator.mark_key_success(tokens_used=total)
                return response
            except OpenAIClientError as e:
                last_error = e
                bound_logger.warning(f"Request failed (attempt {attempt + 1}/{max_retries}): {e.status_code}")
                if self.key_rotator and attempt < max_retries - 1:
                    self.key_rotator.handle_error(e.status_code or 500, str(e))
                    current_key = self.key_rotator.api_keys[self.key_rotator.current_key_index]
                    self.client.update_credentials(current_key.api_key, current_key.base_url)
                else:
                    break

        raise HTTPException(
            status_code=last_error.status_code if last_error else 500,
            detail=last_error.error_response.model_dump(exclude_none=True) if last_error else {"error": "All retries failed"},
        )

    async def process_request(self, body: dict, request_id: str | None = None) -> dict:
        """Process non-streaming Responses API request."""
        from src.common.logging import get_logger_with_request_id
        bound_logger = get_logger_with_request_id(request_id)

        chat_req = _build_chat_request(body)
        chat_req["stream"] = False

        bound_logger.info(f"Responses API -> Chat Completions: model={chat_req.get('model')}, messages={len(chat_req.get('messages', []))}")

        chat_response = await self._send_with_retry(chat_req, request_id=request_id)
        resp = _build_responses_response(chat_response, body.get("model", "gpt-4o"))

        bound_logger.info(f"Responses API response: id={resp['id']}, output_items={len(resp['output'])}")
        return resp

    async def process_stream_request(self, body: dict, request_id: str | None = None) -> AsyncGenerator[str, None]:
        """Process streaming Responses API request."""
        from src.common.logging import get_logger_with_request_id
        bound_logger = get_logger_with_request_id(request_id)

        chat_req = _build_chat_request(body)
        chat_req["stream"] = True

        bound_logger.info(f"Responses API streaming -> Chat Completions: model={chat_req.get('model')}")

        resp_id = f"resp_{uuid.uuid4().hex[:24]}"
        model = body.get("model", "gpt-4o")

        # Send initial response.created event
        created_at = int(time.time())
        yield f"event: response.created\ndata: {json.dumps({'type': 'response.created', 'response': {'id': resp_id, 'object': 'response', 'created_at': created_at, 'model': model, 'status': 'in_progress', 'output': []}})}\n\n"

        # Send response.in_progress
        yield f"event: response.in_progress\ndata: {json.dumps({'type': 'response.in_progress', 'response': {'id': resp_id, 'object': 'response', 'created_at': created_at, 'model': model, 'status': 'in_progress', 'output': []}})}\n\n"

        # Output item and content item IDs
        msg_id = f"msg_{uuid.uuid4().hex[:24]}"
        text_id = f"item_{uuid.uuid4().hex[:24]}"
        output_index = 0
        content_index = 0

        # Send output_item.added
        yield f"event: response.output_item.added\ndata: {json.dumps({'type': 'response.output_item.added', 'output_index': output_index, 'item': {'type': 'message', 'id': msg_id, 'role': 'assistant', 'content': []}})}\n\n"

        # Send content_part.added
        yield f"event: response.content_part.added\ndata: {json.dumps({'type': 'response.content_part.added', 'output_index': output_index, 'content_index': content_index, 'part': {'type': 'output_text', 'text': ''}})}\n\n"

        # Stream text deltas
        # Filter out unsupported tool types before sending
        filtered_tools = []
        for t in chat_req.get("tools", []):
            if t.get("type") == "function":
                fn = t.get("function", {})
                if fn.get("name"):
                    filtered_tools.append(t)
        if filtered_tools:
            chat_req["tools"] = filtered_tools
        elif "tools" in chat_req:
            del chat_req["tools"]
        full_text = ""
        input_tokens = 0
        output_tokens = 0

        async for chunk in self.client.send_raw_streaming_request(chat_req, request_id=request_id):
            if chunk.startswith("data: "):
                data_str = chunk[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    # Extract usage from streaming chunks
                    if data.get("usage"):
                        u = data["usage"]
                        input_tokens = u.get("prompt_tokens", input_tokens)
                        output_tokens = u.get("completion_tokens", output_tokens)

                    for choice in data.get("choices", []):
                        delta = choice.get("delta", {})
                        text = delta.get("content", "")
                        if text:
                            full_text += text
                            yield f"event: response.output_text.delta\ndata: {json.dumps({'type': 'response.output_text.delta', 'output_index': output_index, 'content_index': content_index, 'delta': text})}\n\n"

                        # Handle tool calls in streaming
                        tool_calls = delta.get("tool_calls", [])
                        for tc in tool_calls:
                            fn_delta = tc.get("function", {})
                            if fn_delta.get("name"):
                                yield f"event: response.function_call_arguments.delta\ndata: {json.dumps({'type': 'response.function_call_arguments.delta', 'output_index': output_index, 'item_id': msg_id, 'call_id': tc.get('id', ''), 'delta': fn_delta.get('arguments', '')})}\n\n"
                except json.JSONDecodeError:
                    pass

        # Send content_part.done
        yield f"event: response.content_part.done\ndata: {json.dumps({'type': 'response.content_part.done', 'output_index': output_index, 'content_index': content_index, 'part': {'type': 'output_text', 'text': full_text}})}\n\n"

        # Send output_item.done
        yield f"event: response.output_item.done\ndata: {json.dumps({'type': 'response.output_item.done', 'output_index': output_index, 'item': {'type': 'message', 'id': msg_id, 'role': 'assistant', 'content': [{'type': 'output_text', 'text': full_text}]}})}\n\n"

        # Send response.completed
        yield f"event: response.completed\ndata: {json.dumps({'type': 'response.completed', 'response': {'id': resp_id, 'object': 'response', 'created_at': created_at, 'model': model, 'status': 'completed', 'output': [{'type': 'message', 'id': msg_id, 'role': 'assistant', 'content': [{'type': 'output_text', 'text': full_text}]}], 'usage': {'input_tokens': input_tokens, 'output_tokens': output_tokens, 'total_tokens': input_tokens + output_tokens}}})}\n\n"


@router.post("/responses")
async def responses_endpoint(request: Request, background_tasks: BackgroundTasks):
    """OpenAI Responses API /v1/responses endpoint."""
    handler: ResponsesHandler = request.app.state.responses_handler

    from src.common.logging import get_logger_with_request_id, get_request_id_from_request
    request_id = get_request_id_from_request(request)
    bound_logger = get_logger_with_request_id(request_id)

    client_ip = request.client.host if request.client else "unknown"
    bound_logger.info(f"Responses API request from {client_ip}")

    try:
        body = await request.json()
        bound_logger.debug(f"Responses API body: model={body.get('model')}, stream={body.get('stream', False)}, input_items={len(body.get('input', []))}")

        if body.get("stream"):
            async def stream_wrapper():
                try:
                    async for chunk in handler.process_stream_request(body, request_id=request_id):
                        yield chunk.encode("utf-8")
                        await asyncio.sleep(0)
                except Exception as e:
                    bound_logger.exception(f"Stream error: {e}")
                    error_event = f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
                    yield error_event.encode("utf-8")

            return StreamingResponse(
                stream_wrapper(),
                media_type="text/event-stream; charset=utf-8",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            response = await handler.process_request(body, request_id=request_id)
            return JSONResponse(content=response)

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        bound_logger.exception(f"Responses API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
