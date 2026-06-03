from src.core.clients.openai_client import OpenAIClientError
from src.models.errors import get_error_response
import json


async def send_raw_request(self, request_data, endpoint="/chat/completions", request_id=None):
    from src.common.logging import get_logger_with_request_id
    bound_logger = get_logger_with_request_id(request_id)
    url = f"{self._base_url}{endpoint}"
    bound_logger.info(f"Raw request - URL: {url}, Model: {request_data.get('model', 'unknown')}")
    try:
        response = await self.client.post(url, json=request_data)
        response.raise_for_status()
        return json.loads(response.text)
    except Exception as e:
        import httpx
        if isinstance(e, httpx.HTTPStatusError):
            body = e.response.text
            bound_logger.error(f"Raw request upstream error: status={e.response.status_code}, body={body[:1000]}")
            raise OpenAIClientError(get_error_response(status_code=e.response.status_code, message=body), e.response.status_code, body)
        if isinstance(e, httpx.TimeoutException):
            raise OpenAIClientError(get_error_response(status_code=504, message=str(e)), 504, str(e))
        if isinstance(e, httpx.ConnectError):
            raise OpenAIClientError(get_error_response(status_code=502, message=str(e)), 502, str(e))
        raise


async def send_raw_streaming_request(self, request_data, endpoint="/chat/completions", request_id=None):
    import httpx
    from src.common.logging import get_logger_with_request_id
    bound_logger = get_logger_with_request_id(request_id)
    url = f"{self._base_url}{endpoint}"
    request_data = {**request_data, "stream": True}
    # Log request details for debugging
    msg_count = len(request_data.get('messages', []))
    tools = request_data.get('tools')
    tool_count = len(tools) if tools else 0
    bound_logger.info(f"Raw streaming - URL: {url}, Model: {request_data.get('model', 'unknown')}, messages={msg_count}, tools={tool_count}")
    # Log first message roles for debugging
    for i, m in enumerate(request_data.get('messages', [])[:6]):
        role = m.get('role', '?')
        content_preview = str(m.get('content', ''))[:80]
        bound_logger.info(f"  msg[{i}] role={role}: {content_preview}")
    if tools:
        for i, t in enumerate(tools[:5]):
            ttype = t.get('type', '?')
            name = t.get('function', {}).get('name', '') if ttype == 'function' else ''
            bound_logger.info(f"  tool[{i}] type={ttype} name={name}")

    try:
        async with self.client.stream("POST", url, json=request_data) as response:
            if response.status_code >= 400:
                body = await response.aread()
                body_text = body.decode("utf-8", errors="ignore")
                bound_logger.error(f"Raw streaming upstream error: status={response.status_code}, body={body_text[:1000]}")
                raise OpenAIClientError(get_error_response(status_code=response.status_code, message=body_text), response.status_code, body_text)
            response.raise_for_status()
            buf = ""
            async for chunk_bytes in response.aiter_bytes(chunk_size=1024):
                buf += chunk_bytes.decode("utf-8", errors="ignore")
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if line:
                        yield line
                        if line == "data: [DONE]":
                            return
            if buf.strip():
                yield buf.strip()
    except OpenAIClientError:
        raise
    except Exception as e:
        if isinstance(e, httpx.HTTPStatusError):
            body = ""
            try:
                body = (await e.response.aread()).decode("utf-8", errors="ignore")
            except Exception:
                pass
            bound_logger.error(f"Raw streaming error: status={e.response.status_code}, body={body[:500]}")
            raise OpenAIClientError(get_error_response(status_code=e.response.status_code, message="HTTP error"), e.response.status_code, body)
        if isinstance(e, httpx.TimeoutException):
            raise OpenAIClientError(get_error_response(status_code=504, message="Timeout"), 504, str(e))
        if isinstance(e, httpx.ConnectError):
            raise OpenAIClientError(get_error_response(status_code=502, message="Connection error"), 502, str(e))
        raise
