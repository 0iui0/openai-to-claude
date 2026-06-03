"""API Key轮换器，支持多API key管理和智能切换"""

import hashlib
import time
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
        price_input: float | None = None,
        price_output: float | None = None,
        model: str | None = None,
    ):
        """初始化API Key信息

        Args:
            api_key: API密钥
            name: API Key别名
            index: 在配置列表中的索引
            daily_limit: 每日限额（元）
            base_url: API基础URL（可选）
            weight: 权重，用于初始选择时的加权随机（默认1.0）
            price_input: 输入单价（元/1K tokens）
            price_output: 输出单价（元/1K tokens）
            model: 模型覆盖，使用此key时固定使用该模型名（如本地vLLM固定模型）
        """
        self.api_key = api_key
        self.name = name
        self.index = index
        self.daily_limit = daily_limit
        self.base_url = base_url
        self.weight = weight
        self.price_input = price_input
        self.price_output = price_output
        self.model = model
        self.quota_exhausted = False
        self.failure_count = 0
        self.usage_count = 0
        self.daily_tokens_used = 0
        self.estimated_cost: float = 0.0  # 预估花费（元）
        self.first_used_at: float | None = None  # 首次使用时间
        self.last_error: str | None = None
        self.last_used_date: str | None = None
        self.rate_limited_until: float = 0

    def mark_failure(self, error: str | None = None):
        """标记key失败"""
        self.failure_count += 1
        self.last_error = error
        logger.warning(
            f"API Key [{self.name}] 失败 - Failure Count: {self.failure_count}, Error: {error}"
        )

    def mark_success(self, tokens_used: int = 0, input_tokens: int = 0, output_tokens: int = 0):
        """标记key成功使用

        Args:
            tokens_used: 本次请求使用的 tokens 数量（包括输入和输出）
            input_tokens: 输入 tokens 数量
            output_tokens: 输出 tokens 数量
        """
        self.failure_count = 0
        self.last_error = None
        self.usage_count += 1
        self.daily_tokens_used += tokens_used
        if self.first_used_at is None:
            self.first_used_at = time.time()

        # 预估花费
        if input_tokens > 0 or output_tokens > 0:
            in_cost = (input_tokens / 1000) * (self.price_input or 0)
            out_cost = (output_tokens / 1000) * (self.price_output or 0)
            self.estimated_cost += in_cost + out_cost
        elif tokens_used > 0 and (self.price_input or self.price_output):
            # 没有 input/output 分离时，用平均价估算
            avg_price = ((self.price_input or 0) + (self.price_output or 0)) / 2
            self.estimated_cost += (tokens_used / 1000) * avg_price

        if tokens_used > 0:
            cost_info = f", 预估花费: ¥{self.estimated_cost:.2f}" if self.estimated_cost > 0 else ""
            limit_info = ""
            if self.daily_limit and self.estimated_cost > 0:
                pct = self.estimated_cost / self.daily_limit * 100
                limit_info = f" ({pct:.0f}%/{self.daily_limit}元)"
            logger.info(
                f"API Key [{self.name}] 使用成功 - tokens: {tokens_used}, 累计: {self.daily_tokens_used}, 次数: {self.usage_count}{cost_info}{limit_info}"
            )

    def mark_quota_exhausted(self, error: str | None = None):
        """标记key配额用尽"""
        self.quota_exhausted = True
        self.last_error = error
        logger.error(
            f"API Key [{self.name}] 配额用尽 - Total Usage: {self.usage_count} 次, Tokens: {self.daily_tokens_used}, Error: {error}"
        )

    def is_available(self) -> bool:
        """检查key是否可用（配额未耗尽且不在临时限流期内）"""
        if self.quota_exhausted:
            return False
        if self.rate_limited_until and time.time() < self.rate_limited_until:
            return False
        return True

    def cost_usage_ratio(self) -> float:
        """返回当前预估花费占日限额的比例（0.0~1.0+）"""
        if not self.daily_limit or self.daily_limit <= 0:
            return 0.0
        return self.estimated_cost / self.daily_limit

    def mark_rate_limited(self, cooldown_seconds: float = 300):
        """标记key被临时限流

        Args:
            cooldown_seconds: 冷却时间（秒），默认5分钟
        """
        self.rate_limited_until = time.time() + cooldown_seconds
        logger.warning(
            f"API Key [{self.name}] 被临时限流，冷却 {cooldown_seconds}s"
        )

    def reset_daily_status(self):
        """重置每日状态（每天调用一次）"""
        self.quota_exhausted = False
        self.failure_count = 0
        self.last_error = None
        self.rate_limited_until = 0
        self.estimated_cost = 0.0
        self.first_used_at = None
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
        "exceeded",
        "insufficient",
        "balance",
        "credit",
        "额度",
        "用完",
        "限额",
    ]
    # 临时限流特征（不应永久标记为耗尽）
    TRANSIENT_RATE_LIMIT_PATTERNS = [
        "limit_burst_rate",
        "负载已饱和",
        "rate limit",
        "too many requests",
        "请稍后再试",
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
                - "round_robin": 轮询策略，依次循环使用每个可用key（最简单可靠）
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
                    price_input=config.get("price_input"),
                    price_output=config.get("price_output"),
                    model=config.get("model"),
                )
            )

        # 额度预警阈值（默认 85%），超过后主动切换到下一个 key
        self.cost_warning_threshold: float = 0.85

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
        elif self.strategy == "round_robin":
            # 轮询策略：从第一个开始
            self.current_key_index = 0
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
        elif self.strategy == "round_robin":
            return self._select_round_robin_key()
        else:  # daily
            self._select_daily_key()
            return self.api_keys[self.current_key_index]

    def _select_balanced_key(self) -> APIKeyInfo:
        """选择使用次数最少的可用key（均衡策略）

        优化策略：
        1. 只在可用 key 中选择，避免无效请求
        2. 优先选择使用次数最少的 key
        3. 使用次数相同时，选择失败次数最少的
        4. 都相同时随机选择（避免热点）

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

        # 一次遍历同时过滤可用 key 并找出最优 key（O(n) 复杂度）
        available_keys = []
        min_usage = float('inf')
        min_failures = float('inf')
        best_key = None

        for key_info in self.api_keys:
            if not key_info.is_available():
                continue

            available_keys.append(key_info)

            # 找出使用次数最少的
            if key_info.usage_count < min_usage:
                min_usage = key_info.usage_count
                min_failures = key_info.failure_count
                best_key = key_info
            # 使用次数相同时，比较失败次数
            elif key_info.usage_count == min_usage:
                if key_info.failure_count < min_failures:
                    min_failures = key_info.failure_count
                    best_key = key_info

        if not available_keys:
            logger.error("所有API Key都不可用！")
            raise RuntimeError("所有API Key都不可用，请检查配置或等待配额重置")

        # 如果有多个最优 key，随机选择一个
        if best_key is None:
            import random
            best_key = random.choice(available_keys)

        self.current_key_index = best_key.index

        logger.debug(
            f"均衡选择API Key - Name: [{best_key.name}], Usage: {best_key.usage_count}, "
            f"Failures: {best_key.failure_count}, Tokens: {best_key.daily_tokens_used}, "
            f"Available: {len(available_keys)}/{len(self.api_keys)}"
        )

        return best_key

    def _select_round_robin_key(self) -> APIKeyInfo:
        """轮询选择下一个可用的key（最简单可靠的负载均衡）

        优化策略：
        1. 每次选择都跳过不可用的 key，避免无效请求
        2. 从当前 key 的下一个开始查找，确保真正的轮询
        3. 如果当前 key 不可用，从当前位置开始查找
        4. 最多遍历所有 key 一次（O(n) 复杂度）

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

        total_keys = len(self.api_keys)

        # 确定起始搜索位置：
        # - 如果当前 key 可用，从下一个开始
        # - 如果当前 key 不可用，从当前位置开始
        if self.current_key_index is not None and self.api_keys[self.current_key_index].is_available():
            start_index = (self.current_key_index + 1) % total_keys
        else:
            start_index = self.current_key_index % total_keys if self.current_key_index is not None else 0

        index = start_index
        checked_count = 0
        first_available_index = None

        # 遍历所有 key，最多循环一次
        while checked_count < total_keys:
            key_info = self.api_keys[index]

            if key_info.is_available():
                # 找到第一个可用的 key
                if first_available_index is None:
                    first_available_index = index

                # 优先选择下一个可用的 key（实现真正的轮询）
                self.current_key_index = index

                # 只在真正切换 key 时记录日志（避免日志噪音）
                if index != start_index or not self.api_keys[start_index].is_available():
                    logger.debug(
                        f"轮询选择API Key - Name: [{key_info.name}], Index: {index}, Usage: {key_info.usage_count}, Tokens: {key_info.daily_tokens_used}"
                    )

                return key_info

            # 跳过不可用的 key
            index = (index + 1) % total_keys
            checked_count += 1

        # 如果遍历完所有 key 都没有可用的
        logger.error(
            f"所有API Key都不可用！已检查 {checked_count} 个key，"
            f"可用: 0/{total_keys}"
        )
        raise RuntimeError("所有API Key都不可用，请检查配置或等待配额重置")

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

    def mark_key_failure(self, error_message: str | None = None, switch_key: bool = False):
        """标记当前key失败

        Args:
            error_message: 错误消息
            switch_key: 是否切换到下一个可用key
        """
        if self.current_key_index is None:
            logger.error("无法标记密钥失败：current_key_index 为 None")
            return

        current_key = self.api_keys[self.current_key_index]
        current_key.mark_failure(error_message)

        if switch_key:
            self._switch_to_next_available_key()

    def mark_key_success(self, tokens_used: int = 0):
        """标记当前key成功并增加使用计数

        Args:
            tokens_used: 本次请求使用的 tokens 数量
        """
        if self.current_key_index is None:
            logger.error("无法标记密钥成功：current_key_index 为 None")
            return

        current_key = self.api_keys[self.current_key_index]
        current_key.mark_success(tokens_used)

        # 额度预警：接近阈值时主动切换
        self._check_proactive_rotation(current_key)

    def _check_proactive_rotation(self, current_key: APIKeyInfo):
        """检查是否需要主动轮转（额度接近阈值时提前切换）

        Args:
            current_key: 当前使用的 key
        """
        if not current_key.daily_limit or current_key.daily_limit <= 0:
            return

        ratio = current_key.cost_usage_ratio()
        if ratio < self.cost_warning_threshold:
            return

        # 尝试找到一个花费比例更低的 key
        candidates = [
            k for k in self.api_keys
            if k.is_available() and k.index != current_key.index
            and k.cost_usage_ratio() < ratio
        ]
        if not candidates:
            return

        # 选花费比例最低的
        best = min(candidates, key=lambda k: k.cost_usage_ratio())
        old_pct = f"{ratio:.0%}"
        new_pct = f"{best.cost_usage_ratio():.0%}"
        logger.warning(
            f"额度预警 - [{current_key.name}] 已用 {old_pct} (¥{current_key.estimated_cost:.2f}/{current_key.daily_limit}元)"
            f"，主动切换到 [{best.name}] ({new_pct})"
        )
        self.current_key_index = best.index

    def mark_key_quota_exhausted(self, error_message: str | None = None, session_id: str | None = None):
        """标记当前key配额用尽并切换到下一个可用key

        Args:
            error_message: 错误消息
            session_id: 触发配额用尽的会话ID（仅在session_affinity策略下使用）
        """
        # 直接使用 current_key_index，避免触发密钥选择逻辑（防止竞态条件）
        if self.current_key_index is None:
            logger.error("无法标记密钥配额用尽：current_key_index 为 None")
            raise RuntimeError("无法标记密钥配额用尽：current_key_index 为 None")

        current_key = self.api_keys[self.current_key_index]
        current_key.mark_quota_exhausted(error_message)

        # 从会话映射中移除该密钥（如果存在）
        sessions_to_remove = [
            sid
            for sid, key_idx in self.session_key_mapping.items()
            if key_idx == self.current_key_index
        ]
        for sid in sessions_to_remove:
            del self.session_key_mapping[sid]
            logger.debug(f"从会话映射中移除密钥 - Session: {sid[:8]}..., Key: [{current_key.name}]")

        # 切换到下一个可用的key
        self._switch_to_next_available_key()

        # 如果是session_affinity策略且提供了session_id，自动将新key分配给该会话
        if self.strategy == "session_affinity" and session_id is not None:
            new_key = self.api_keys[self.current_key_index]
            self.session_key_mapping[session_id] = self.current_key_index
            logger.debug(
                f"自动分配新key到会话 - Session: {session_id[:8]}..., Key: [{new_key.name}]"
            )

    def _switch_to_next_available_key(self):
        """切换到下一个可用的API key（错误处理路径）

        优化策略：
        1. 从当前 key 的下一个开始查找（避免重复使用刚失败的 key）
        2. 最多遍历所有 key 一次（O(n) 复杂度）
        3. 使用计数器代替索引比较，避免边界条件问题

        Raises:
            RuntimeError: 如果所有 key 都不可用
        """
        total_keys = len(self.api_keys)
        original_key_index = self.current_key_index
        original_key_name = self.api_keys[original_key_index].name

        # 从当前 key 的下一个开始搜索
        start_index = (original_key_index + 1) % total_keys
        index = start_index
        checked_count = 0

        # 遍历所有 key，最多循环一次
        while checked_count < total_keys:
            key_info = self.api_keys[index]

            if key_info.is_available():
                self.current_key_index = index
                logger.info(
                    f"切换到新的API Key - From: [{original_key_name}] To: [{key_info.name}] "
                    f"(Index: {original_key_index} → {index}, Checked: {checked_count + 1}/{total_keys})"
                )
                return

            # 继续查找下一个
            index = (index + 1) % total_keys
            checked_count += 1

        # 如果所有 key 都不可用
        logger.error(
            f"切换 API Key 失败 - 从 [{original_key_name}] 无法找到可用 key，"
            f"已检查所有 {total_keys} 个 key"
        )
        raise RuntimeError("所有API Key都不可用，请检查配置或等待配额重置")

    def is_quota_error(self, status_code: int, error_message: str | None = None) -> bool:
        """判断是否为配额用尽错误（排除临时限流）

        Args:
            status_code: HTTP状态码
            error_message: 错误消息

        Returns:
            是否为配额用尽错误
        """
        # 临时限流优先检测：即使状态码是429/402，如果消息明确是临时限流，不算配额耗尽
        if self.is_transient_rate_limit(error_message):
            return False

        # 检查状态码
        if status_code in self.QUOTA_ERROR_CODES:
            return True

        # 检查错误消息是否包含配额相关关键词（适用于所有状态码）
        if error_message:
            error_lower = error_message.lower()
            for pattern in self.QUOTA_ERROR_PATTERNS:
                if pattern in error_lower:
                    return True

        return False

    def is_transient_rate_limit(self, error_message: str | None) -> bool:
        """判断是否为临时限流错误（非配额耗尽）

        Args:
            error_message: 错误消息

        Returns:
            是否为临时限流错误
        """
        if not error_message:
            return False
        error_lower = error_message.lower()
        return any(p in error_lower for p in self.TRANSIENT_RATE_LIMIT_PATTERNS)

    def handle_error(self, status_code: int, error_message: str | None = None, session_id: str | None = None):
        """处理API错误，自动判断是否需要切换key

        Args:
            status_code: HTTP状态码
            error_message: 错误消息
            session_id: 触发错误的会话ID（用于自动分配新key到该会话）
        """
        if self.is_quota_error(status_code, error_message):
            logger.warning(
                f"检测到配额用尽错误 - Status: {status_code}, Error: {error_message}"
            )
            self.mark_key_quota_exhausted(error_message, session_id=session_id)
        elif status_code == 401:
            # API key无效
            logger.error(f"检测到无效的API Key - Status: {status_code}")
            self.mark_key_quota_exhausted(f"Invalid API key: {error_message}", session_id=session_id)
        elif self.is_transient_rate_limit(error_message):
            # 临时限流（任何状态码）：冷却5分钟后自动恢复
            logger.warning(
                f"临时限流（冷却5分钟） - Status: {status_code}, Error: {error_message}"
            )
            current_key = self.api_keys[self.current_key_index]
            current_key.mark_rate_limited(cooldown_seconds=300)
            self._switch_to_next_available_key()
        elif status_code == 500:
            # 500服务端错误：检查错误消息判断是配额耗尽还是临时限流
            if self.is_transient_rate_limit(error_message):
                # 临时限流：冷却5分钟后自动恢复
                logger.warning(
                    f"500临时限流（切换key，冷却5分钟） - Status: {status_code}, Error: {error_message}"
                )
                current_key = self.api_keys[self.current_key_index]
                current_key.mark_rate_limited(cooldown_seconds=300)
                self._switch_to_next_available_key()
            elif error_message and any(p in error_message.lower() for p in self.QUOTA_ERROR_PATTERNS):
                # 500但消息包含配额关键词（如"额度已用完"）：标记为配额耗尽
                logger.warning(
                    f"500配额耗尽错误 - Status: {status_code}, Error: {error_message}"
                )
                self.mark_key_quota_exhausted(error_message, session_id=session_id)
            else:
                # 其他500错误：仅标记失败，切换key重试
                logger.warning(
                    f"500服务端错误（切换key重试） - Status: {status_code}, Error: {error_message}"
                )
                self.mark_key_failure(error_message, switch_key=True)
        elif status_code in (502, 503, 504):
            # 上游网关错误：所有key共用同一个base_url，换key无意义，仅记录
            logger.warning(
                f"上游网关错误（不换key，直接重试） - Status: {status_code}, Error: {error_message}"
            )
            self.mark_key_failure(error_message)
        else:
            # 其他错误，仅记录不切换
            self.mark_key_failure(error_message)

    def get_status(self) -> dict[str, Any]:
        """获取所有key的状态信息和使用统计"""
        now = time.time()
        total_cost = sum(k.estimated_cost for k in self.api_keys)
        total_limit = sum(k.daily_limit for k in self.api_keys if k.daily_limit)

        keys_status = []
        for key in self.api_keys:
            entry = {
                "name": key.name,
                "index": key.index,
                "api_key": f"{key.api_key[:8]}...{key.api_key[-4:]}",
                "is_current": key.index == self.current_key_index,
                "is_available": key.is_available(),
                "quota_exhausted": key.quota_exhausted,
                "rate_limited": now < key.rate_limited_until if key.rate_limited_until else False,
                "failure_count": key.failure_count,
                "daily_usage_count": key.usage_count,
                "daily_tokens_used": key.daily_tokens_used,
                "estimated_cost": round(key.estimated_cost, 4),
                "daily_limit": key.daily_limit,
                "cost_ratio": f"{key.cost_usage_ratio():.1%}" if key.daily_limit else None,
                "weight": key.weight,
                "last_error": key.last_error,
            }

            # 预估剩余时间和耗尽时间
            if key.daily_limit and key.estimated_cost > 0 and key.first_used_at:
                elapsed = now - key.first_used_at
                if elapsed > 60:
                    cost_per_minute = key.estimated_cost / (elapsed / 60)
                    remaining = key.daily_limit - key.estimated_cost
                    if remaining > 0 and cost_per_minute > 0:
                        eta_minutes = remaining / cost_per_minute
                        entry["eta_exhaust_minutes"] = round(eta_minutes, 1)
                    entry["cost_per_minute"] = round(cost_per_minute, 4)

            keys_status.append(entry)

        return {
            "current_date": self.current_date,
            "current_key_index": self.current_key_index,
            "total_keys": len(self.api_keys),
            "total_sessions": len(self.session_key_mapping),
            "total_estimated_cost": round(total_cost, 2),
            "total_daily_limit": round(total_limit, 2) if total_limit else None,
            "total_cost_ratio": f"{total_cost / total_limit:.1%}" if total_limit else None,
            "cost_warning_threshold": f"{self.cost_warning_threshold:.0%}",
            "keys": keys_status,
        }

    def reset_all_keys(self):
        """重置所有key的状态（用于手动重置）"""
        logger.info("手动重置所有API Key状态")
        for key_info in self.api_keys:
            key_info.reset_daily_status()
        # 重新选择今天的key
        self._select_daily_key()
