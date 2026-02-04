"""API Key轮换器，支持多API key管理和智能切换"""

import hashlib
from datetime import date
from typing import Any

from loguru import logger


class APIKeyInfo:
    """API Key信息类，跟踪每个key的状态"""

    def __init__(
        self,
        api_key: str,
        name: str,
        index: int,
        daily_limit: float | None = None,
        base_url: str | None = None,
        weight: float = 1.0,
    ):
        """初始化API Key信息

        Args:
            api_key: API密钥
            name: API Key别名
            index: 在配置列表中的索引
            daily_limit: 每日限额（元）
            base_url: API基础URL（可选）
            weight: 权重，用于初始选择时的加权随机（默认1.0）
        """
        self.api_key = api_key
        self.name = name
        self.index = index
        self.daily_limit = daily_limit
        self.base_url = base_url
        self.weight = weight
        self.quota_exhausted = False
        self.failure_count = 0
        self.usage_count = 0  # 使用次数统计
        self.daily_tokens_used = 0  # 每日使用的 tokens 数量统计
        self.last_error: str | None = None
        self.last_used_date: str | None = None

    def mark_failure(self, error: str | None = None):
        """标记key失败"""
        self.failure_count += 1
        self.last_error = error
        logger.warning(
            f"API Key [{self.name}] 失败 - Failure Count: {self.failure_count}, Error: {error}"
        )

    def mark_success(self, tokens_used: int = 0):
        """标记key成功使用

        Args:
            tokens_used: 本次请求使用的 tokens 数量（包括输入和输出）
        """
        self.failure_count = 0
        self.last_error = None
        self.usage_count += 1
        self.daily_tokens_used += tokens_used
        if tokens_used > 0:
            logger.info(
                f"API Key [{self.name}] 使用成功 - 本次 tokens: {tokens_used}, 累计 tokens: {self.daily_tokens_used}, 使用次数: {self.usage_count}"
            )

    def mark_quota_exhausted(self, error: str | None = None):
        """标记key配额用尽"""
        self.quota_exhausted = True
        self.last_error = error
        logger.error(
            f"API Key [{self.name}] 配额用尽 - Total Usage: {self.usage_count} 次, Tokens: {self.daily_tokens_used}, Error: {error}"
        )

    def is_available(self) -> bool:
        """检查key是否可用"""
        return not self.quota_exhausted

    def reset_daily_status(self):
        """重置每日状态（每天调用一次）"""
        self.quota_exhausted = False
        self.failure_count = 0
        self.last_error = None
        tokens_before_reset = self.daily_tokens_used
        self.daily_tokens_used = 0
        logger.debug(
            f"API Key [{self.name}] 状态已重置 - 昨日使用: {tokens_before_reset} tokens"
        )

    def __repr__(self) -> str:
        return f"APIKeyInfo(name={self.name}, index={self.index}, exhausted={self.quota_exhausted}, failures={self.failure_count}, usage={self.usage_count}, tokens={self.daily_tokens_used})"


class APIKeyRotator:
    """API Key轮换器

    功能：
    1. 均衡随机选择API key（每次请求时选择使用次数最少的key）
    2. 检测配额用尽错误并自动切换到下一个可用key
    3. 跟踪每个key的使用状态
    """

    # 配额用尽的错误特征
    QUOTA_ERROR_CODES = [402, 429]
    QUOTA_ERROR_PATTERNS = [
        "quota",
        "limit",
        "exceeded",
        "insufficient",
        "balance",
        "credit",
    ]

    def __init__(
        self,
        api_keys_config: list[dict[str, Any]],
        strategy: str = "session_affinity",
    ):
        """初始化API Key轮换器

        Args:
            api_keys_config: API key配置列表，每个元素包含:
                - api_key: API密钥
                - daily_limit: 每日限额（可选）
                - base_url: API基础URL（可选）
                - weight: 权重，用于初始选择（可选，默认1.0）
            strategy: 选择策略
                - "session_affinity": 会话粘性（默认），每个session使用同一个key
                - "balanced": 均衡选择（优先选择使用次数最少的key）
                - "daily": 每日固定选择（使用日期作为随机种子）
        """
        if not api_keys_config:
            raise ValueError("API keys配置不能为空")

        self.strategy = strategy
        self.api_keys: list[APIKeyInfo] = []
        for idx, config in enumerate(api_keys_config):
            self.api_keys.append(
                APIKeyInfo(
                    api_key=config["api_key"],
                    name=config.get("name", f"key-{idx}"),
                    index=idx,
                    daily_limit=config.get("daily_limit"),
                    base_url=config.get("base_url"),
                    weight=config.get("weight", 1.0),
                )
            )

        self.current_key_index: int | None = None
        self.current_date: str | None = None

        # 会话粘性：session_id -> key_index 的映射
        self.session_key_mapping: dict[str, int] = {}

        # 初始化时选择今天的key
        self._initialize_key_selection()

    def _initialize_key_selection(self):
        """初始化key选择策略"""
        if self.strategy == "daily":
            self._select_daily_key()
        elif self.strategy == "session_affinity":
            # 会话粘性策略：不需要预选，根据session动态选择
            self.current_key_index = 0
        else:  # balanced
            # 均衡策略：每次动态选择，不需要预选
            self.current_key_index = 0

    def _select_daily_key(self):
        """选择今天的API key（使用日期作为随机种子）"""
        today = date.today().isoformat()

        # 如果日期没变，不重新选择
        if self.current_date == today:
            return

        logger.info(
            f"日期变更，重新选择API key - Previous Date: {self.current_date}, New Date: {today}"
        )

        # 重置所有key的每日状态
        for key_info in self.api_keys:
            key_info.reset_daily_status()

        # 使用日期作为随机种子，保证同一天总是选择相同的key
        seed = int(hashlib.sha256(today.encode()).hexdigest(), 16)
        self.current_key_index = seed % len(self.api_keys)
        self.current_date = today

        selected_key = self.api_keys[self.current_key_index]
        logger.info(f"今日选择的API Key - Name: [{selected_key.name}], Date: {today}")

    def _select_key_by_weight(self) -> APIKeyInfo:
        """根据权重随机选择一个可用的 API key

        Returns:
            选择的API key信息

        Raises:
            RuntimeError: 如果所有key都不可用
        """
        # 获取所有可用的keys
        available_keys = [k for k in self.api_keys if k.is_available()]

        if not available_keys:
            logger.error("所有API Key都不可用！")
            raise RuntimeError("所有API Key都不可用，请检查配置或等待配额重置")

        # 按权重随机选择
        import random

        # 计算总权重
        total_weight = sum(k.weight for k in available_keys)

        # 加权随机选择
        rand = random.uniform(0, total_weight)
        cumulative_weight = 0
        selected = available_keys[0]

        for key in available_keys:
            cumulative_weight += key.weight
            if rand <= cumulative_weight:
                selected = key
                break

        self.current_key_index = selected.index
        logger.debug(
            f"加权随机选择API Key - Name: [{selected.name}], Weight: {selected.weight}, Total Weight: {total_weight}"
        )

        return selected

    def _select_key_for_session(self, session_id: str) -> APIKeyInfo:
        """为指定会话选择或获取已分配的 API key

        Args:
            session_id: 会话标识符

        Returns:
            选择的API key信息
        """
        # 检查日期是否变更，如果变更则重置所有key的每日状态和会话映射
        today = date.today().isoformat()
        if self.current_date != today:
            logger.info(
                f"日期变更 - Previous Date: {self.current_date}, New Date: {today}, 重置所有API Key状态和会话映射"
            )
            for key_info in self.api_keys:
                key_info.reset_daily_status()
            self.session_key_mapping.clear()  # 清空会话映射
            self.current_date = today

        # 如果该session已经有分配的key，检查该key是否仍然可用
        if session_id in self.session_key_mapping:
            key_index = self.session_key_mapping[session_id]
            key_info = self.api_keys[key_index]

            if key_info.is_available():
                # key仍然可用，继续使用
                logger.debug(
                    f"会话 {session_id[:8]}... 继续使用 Key-[{key_info.name}] (已使用 {key_info.usage_count} 次)"
                )
                self.current_key_index = key_index
                return key_info
            else:
                # key已不可用（配额用尽），需要重新分配
                logger.warning(
                    f"会话 {session_id[:8]}... 原分配的 Key-{key_info.index} 已不可用，重新分配"
                )
                del self.session_key_mapping[session_id]

        # 该session没有分配的key，或原key已不可用，按权重选择新的key
        selected_key = self._select_key_by_weight()

        # 记录会话到key的映射
        self.session_key_mapping[session_id] = selected_key.index

        logger.info(
            f"会话 {session_id[:8]}... 分配到 Key-[{selected_key.name}] (权重: {selected_key.weight}, 总使用: {selected_key.usage_count} 次)"
        )

        return selected_key

    def get_current_key(self, session_id: str | None = None) -> APIKeyInfo:
        """获取当前使用的API key

        Args:
            session_id: 会话标识符（用于会话粘性策略）

        Returns:
            当前使用的API key信息
        """
        if self.strategy == "session_affinity":
            if session_id is None:
                # 如果没有提供session_id，使用默认的"global"会话
                session_id = "global"
            return self._select_key_for_session(session_id)
        elif self.strategy == "balanced":
            return self._select_balanced_key()
        else:  # daily
            self._select_daily_key()
            return self.api_keys[self.current_key_index]

    def _select_balanced_key(self) -> APIKeyInfo:
        """选择使用次数最少的可用key（均衡策略）

        Returns:
            选择的API key信息
        """
        # 检查日期是否变更，如果变更则重置所有key的每日状态
        today = date.today().isoformat()
        if self.current_date != today:
            logger.info(
                f"日期变更 - Previous Date: {self.current_date}, New Date: {today}, 重置所有API Key状态"
            )
            for key_info in self.api_keys:
                key_info.reset_daily_status()
            self.current_date = today

        # 获取所有可用的keys
        available_keys = [k for k in self.api_keys if k.is_available()]

        if not available_keys:
            logger.error("所有API Key都不可用！")
            raise RuntimeError("所有API Key都不可用，请检查配置或等待配额重置")

        # 优先选择使用次数（失败计数）最少的key
        # 如果失败次数相同，随机选择一个
        min_failures = min(k.failure_count for k in available_keys)
        least_used_keys = [k for k in available_keys if k.failure_count == min_failures]

        import random

        selected = random.choice(least_used_keys)
        self.current_key_index = selected.index

        logger.debug(
            f"均衡选择API Key - Name: [{selected.name}], Failures: {selected.failure_count}, Usage: {selected.usage_count}, Available Keys: {len(available_keys)}/{len(self.api_keys)}"
        )

        return selected

    def get_current_api_key(self, session_id: str | None = None) -> str:
        """获取当前使用的API key字符串

        Args:
            session_id: 会话标识符（用于会话粘性策略）
        """
        return self.get_current_key(session_id).api_key

    def get_current_base_url(self, session_id: str | None = None) -> str | None:
        """获取当前使用的base URL

        Args:
            session_id: 会话标识符（用于会话粘性策略）
        """
        return self.get_current_key(session_id).base_url

    def mark_key_failure(self, error_message: str | None = None):
        """标记当前key失败"""
        current_key = self.get_current_key()
        current_key.mark_failure(error_message)

    def mark_key_success(self, tokens_used: int = 0):
        """标记当前key成功并增加使用计数

        Args:
            tokens_used: 本次请求使用的 tokens 数量
        """
        current_key = self.get_current_key()
        current_key.mark_success(tokens_used)

    def mark_key_quota_exhausted(self, error_message: str | None = None):
        """标记当前key配额用尽并切换到下一个可用key"""
        current_key = self.get_current_key()
        current_key.mark_quota_exhausted(error_message)

        # 切换到下一个可用的key
        self._switch_to_next_available_key()

    def _switch_to_next_available_key(self):
        """切换到下一个可用的API key"""
        # 从当前key的下一个开始搜索
        start_index = (self.current_key_index + 1) % len(self.api_keys)
        index = start_index
        original_key_index = self.current_key_index

        while index != original_key_index:
            key_info = self.api_keys[index]
            if key_info.is_available():
                self.current_key_index = index
                logger.info(
                    f"切换到新的API Key - From: [{self.api_keys[original_key_index].name}] To: [{key_info.name}]"
                )
                return

            index = (index + 1) % len(self.api_keys)

        # 如果所有key都不可用，记录错误
        logger.error("所有API Key都不可用！")
        raise RuntimeError("所有API Key都不可用，请检查配置或等待配额重置")

    def is_quota_error(self, status_code: int, error_message: str | None = None) -> bool:
        """判断是否为配额用尽错误

        Args:
            status_code: HTTP状态码
            error_message: 错误消息

        Returns:
            是否为配额用尽错误
        """
        # 检查状态码
        if status_code in self.QUOTA_ERROR_CODES:
            return True

        # 检查错误消息
        if error_message:
            error_lower = error_message.lower()
            for pattern in self.QUOTA_ERROR_PATTERNS:
                if pattern in error_lower:
                    return True

        return False

    def handle_error(self, status_code: int, error_message: str | None = None):
        """处理API错误，自动判断是否需要切换key

        Args:
            status_code: HTTP状态码
            error_message: 错误消息
        """
        if self.is_quota_error(status_code, error_message):
            logger.warning(
                f"检测到配额用尽错误 - Status: {status_code}, Error: {error_message}"
            )
            self.mark_key_quota_exhausted(error_message)
        elif status_code == 401:
            # API key无效
            logger.error(f"检测到无效的API Key - Status: {status_code}")
            self.mark_key_quota_exhausted(f"Invalid API key: {error_message}")
        else:
            # 其他错误，仅记录不切换
            self.mark_key_failure(error_message)

    def get_status(self) -> dict[str, Any]:
        """获取所有key的状态信息和使用统计"""
        return {
            "current_date": self.current_date,
            "current_key_index": self.current_key_index,
            "total_keys": len(self.api_keys),
            "total_sessions": len(self.session_key_mapping),
            "keys": [
                {
                    "name": key.name,
                    "index": key.index,
                    "api_key": f"{key.api_key[:8]}...{key.api_key[-4:]}",
                    "is_current": key.index == self.current_key_index,
                    "is_available": key.is_available(),
                    "quota_exhausted": key.quota_exhausted,
                    "failure_count": key.failure_count,
                    "daily_usage_count": key.usage_count,
                    "daily_tokens_used": key.daily_tokens_used,
                    "weight": key.weight,
                    "last_error": key.last_error,
                }
                for key in self.api_keys
            ],
        }

    def reset_all_keys(self):
        """重置所有key的状态（用于手动重置）"""
        logger.info("手动重置所有API Key状态")
        for key_info in self.api_keys:
            key_info.reset_daily_status()
        # 重新选择今天的key
        self._select_daily_key()
