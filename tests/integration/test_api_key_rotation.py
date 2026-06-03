"""测试 API Key 轮换功能的集成测试"""

import pytest
from src.core.api_key_rotator import APIKeyRotator, APIKeyInfo


class TestAPIKeyRotator:
    """测试 APIKeyRotator 类"""

    def test_init_with_multiple_keys(self):
        """测试使用多个 API keys 初始化"""
        api_keys_config = [
            {"api_key": "key1", "daily_limit": 150.0},
            {"api_key": "key2", "daily_limit": 150.0},
            {"api_key": "key3", "daily_limit": 150.0},
        ]

        rotator = APIKeyRotator(api_keys_config)

        assert len(rotator.api_keys) == 3
        assert rotator.current_key_index is not None
        assert rotator.current_date is not None

    def test_get_current_key(self):
        """测试获取当前 API key"""
        api_keys_config = [
            {"api_key": "key1", "daily_limit": 150.0},
            {"api_key": "key2", "daily_limit": 150.0},
        ]

        rotator = APIKeyRotator(api_keys_config)
        current_key = rotator.get_current_key()

        assert isinstance(current_key, APIKeyInfo)
        assert current_key.api_key in ["key1", "key2"]

    def test_get_current_api_key(self):
        """测试获取当前 API key 字符串"""
        api_keys_config = [
            {"api_key": "key1", "daily_limit": 150.0},
            {"api_key": "key2", "daily_limit": 150.0},
        ]

        rotator = APIKeyRotator(api_keys_config)
        api_key = rotator.get_current_api_key()

        assert api_key in ["key1", "key2"]

    def test_mark_quota_exhausted_and_switch(self):
        """测试标记配额用尽并切换到下一个 key"""
        api_keys_config = [
            {"api_key": "key1", "daily_limit": 150.0},
            {"api_key": "key2", "daily_limit": 150.0},
        ]

        rotator = APIKeyRotator(api_keys_config)
        initial_index = rotator.current_key_index

        # 标记当前 key 配额用尽
        rotator.mark_key_quota_exhausted("Quota exceeded")

        # 验证已切换到不同的 key
        assert rotator.current_key_index != initial_index
        assert rotator.api_keys[initial_index].quota_exhausted is True

    def test_is_quota_error(self):
        """测试配额错误识别"""
        api_keys_config = [{"api_key": "key1", "daily_limit": 150.0}]
        rotator = APIKeyRotator(api_keys_config)

        # 测试各种配额错误
        assert rotator.is_quota_error(429) is True  # Rate limit
        assert rotator.is_quota_error(402) is True  # Payment required
        assert rotator.is_quota_error(429, "quota exceeded") is True
        assert rotator.is_quota_error(402, "insufficient credits") is True

        # 500不应视为配额错误（服务端错误不应导致key永久禁用）
        assert rotator.is_quota_error(500) is False
        assert rotator.is_quota_error(500, "quota exceeded") is False
        assert rotator.is_quota_error(500, "internal server error") is False

    def test_handle_error_with_quota_error(self):
        """测试处理配额错误"""
        api_keys_config = [
            {"api_key": "key1", "daily_limit": 150.0},
            {"api_key": "key2", "daily_limit": 150.0},
        ]

        rotator = APIKeyRotator(api_keys_config)
        initial_index = rotator.current_key_index

        # 处理配额错误
        rotator.handle_error(429, "Rate limit exceeded")

        # 验证已切换 key
        assert rotator.current_key_index != initial_index
        assert rotator.api_keys[initial_index].quota_exhausted is True

    def test_handle_error_with_invalid_key(self):
        """测试处理无效 key 错误"""
        api_keys_config = [
            {"api_key": "key1", "daily_limit": 150.0},
            {"api_key": "key2", "daily_limit": 150.0},
        ]

        rotator = APIKeyRotator(api_keys_config)
        initial_index = rotator.current_key_index

        # 处理 401 错误（无效 key）
        rotator.handle_error(401, "Invalid API key")

        # 验证已切换 key
        assert rotator.current_key_index != initial_index
        assert rotator.api_keys[initial_index].quota_exhausted is True

    def test_all_keys_exhausted(self):
        """测试所有 keys 都不可用时的情况"""
        api_keys_config = [
            {"api_key": "key1", "daily_limit": 150.0},
            {"api_key": "key2", "daily_limit": 150.0},
        ]

        rotator = APIKeyRotator(api_keys_config)

        # 标记所有 key 为配额用尽
        for key_info in rotator.api_keys:
            key_info.mark_quota_exhausted()

        # 尝试再次标记配额用尽，应该抛出异常
        with pytest.raises(RuntimeError, match="所有API Key都不可用"):
            rotator.mark_key_quota_exhausted()

    def test_mark_success(self):
        """测试标记 key 使用成功"""
        api_keys_config = [{"api_key": "key1", "daily_limit": 150.0}]
        rotator = APIKeyRotator(api_keys_config)

        # 标记一些失败
        rotator.mark_key_failure("Test error")
        assert rotator.api_keys[0].failure_count == 1

        # 标记成功，应该重置失败计数
        rotator.mark_key_success()
        assert rotator.api_keys[0].failure_count == 0
        assert rotator.api_keys[0].last_error is None

    def test_get_status(self):
        """测试获取状态信息"""
        api_keys_config = [
            {"api_key": "sk-abc123def456", "daily_limit": 150.0},
            {"api_key": "sk-xyz789uvw012", "daily_limit": 150.0},
        ]

        rotator = APIKeyRotator(api_keys_config)
        status = rotator.get_status()

        assert "current_date" in status
        assert "current_key_index" in status
        assert "total_keys" in status
        assert status["total_keys"] == 2
        assert "keys" in status
        assert len(status["keys"]) == 2

        # 检查 key 信息是否正确脱敏
        key_info = status["keys"][0]
        assert key_info["api_key"].startswith("sk-abc12")
        assert key_info["api_key"].endswith("e456")
        assert "index" in key_info
        assert "is_current" in key_info
        assert "is_available" in key_info

    def test_reset_all_keys(self):
        """测试重置所有 keys"""
        api_keys_config = [
            {"api_key": "key1", "daily_limit": 150.0},
            {"api_key": "key2", "daily_limit": 150.0},
        ]

        rotator = APIKeyRotator(api_keys_config)

        # 标记第一个 key 为配额用尽
        rotator.mark_key_quota_exhausted("Test")
        initial_index = rotator.current_key_index

        # 重置所有 keys
        rotator.reset_all_keys()

        # 验证所有 keys 都已重置
        for key_info in rotator.api_keys:
            assert key_info.quota_exhausted is False
            assert key_info.failure_count == 0
            assert key_info.last_error is None

    def test_daily_key_selection_consistency(self):
        """测试每天选择的 key 一致性"""
        api_keys_config = [
            {"api_key": "key1", "daily_limit": 150.0},
            {"api_key": "key2", "daily_limit": 150.0},
            {"api_key": "key3", "daily_limit": 150.0},
        ]

        rotator1 = APIKeyRotator(api_keys_config, strategy="daily")
        rotator2 = APIKeyRotator(api_keys_config, strategy="daily")

        # 同一天创建的两个 rotator 应该选择相同的 key
        assert rotator1.current_key_index == rotator2.current_key_index

    def test_balanced_strategy_daily_reset(self):
        """测试均衡策略下每日额度重置"""
        from unittest.mock import patch
        from datetime import date

        api_keys_config = [
            {"api_key": "key1", "daily_limit": 150.0},
            {"api_key": "key2", "daily_limit": 150.0},
        ]

        rotator = APIKeyRotator(api_keys_config, strategy="balanced")

        # 第一天：标记第一个 key 为配额用尽
        first_key = rotator.get_current_key()
        assert first_key.quota_exhausted is False
        rotator.mark_key_quota_exhausted("Daily quota exceeded")

        # 验证第一个 key 已被标记为配额用尽
        assert rotator.api_keys[first_key.index].quota_exhausted is True

        # 模拟第二天：修改返回的日期
        tomorrow = "2025-01-16"
        with patch("src.core.api_key_rotator.date") as mock_date:
            mock_date.today.return_value = date.fromisoformat(tomorrow)

            # 获取 key 时应该检测到日期变更并重置所有 key
            next_key = rotator.get_current_key()

            # 验证所有 key 的状态已重置
            for key_info in rotator.api_keys:
                assert (
                    key_info.quota_exhausted is False
                ), f"Key {key_info.index} should be reset"
                assert (
                    key_info.failure_count == 0
                ), f"Key {key_info.index} failure count should be reset"

            # 验证日期已更新
            assert rotator.current_date == tomorrow

    def test_quotas_exhausted_all_keys_scenario(self):
        """测试所有 keys 在单日内配额用尽的场景"""
        api_keys_config = [
            {"api_key": "key1", "daily_limit": 150.0},
            {"api_key": "key2", "daily_limit": 150.0},
        ]

        rotator = APIKeyRotator(api_keys_config, strategy="balanced")

        # 标记所有 key 为配额用尽
        rotator.mark_key_quota_exhausted("Key 1 quota exceeded")
        rotator.mark_key_quota_exhausted("Key 2 quota exceeded")

        # 验证所有 key 都不可用
        for key_info in rotator.api_keys:
            assert key_info.quota_exhausted is True
            assert key_info.is_available() is False

        # 尝试获取 key 应该抛出异常
        with pytest.raises(RuntimeError, match="所有API Key都不可用"):
            rotator.get_current_key()

    def test_balanced_selection_with_different_failure_counts(self):
        """测试均衡策略根据失败次数选择 key"""
        import random

        api_keys_config = [
            {"api_key": "key1", "daily_limit": 150.0},
            {"api_key": "key2", "daily_limit": 150.0},
            {"api_key": "key3", "daily_limit": 150.0},
        ]

        rotator = APIKeyRotator(api_keys_config, strategy="balanced")

        # 给不同的 key 设置不同的失败次数
        rotator.api_keys[0].failure_count = 5
        rotator.api_keys[1].failure_count = 1
        rotator.api_keys[2].failure_count = 3

        # 多次选择，应该倾向于选择失败次数最少的 key (key1)
        selected_counts = {0: 0, 1: 0, 2: 0}
        for _ in range(20):
            selected = rotator.get_current_key()
            selected_counts[selected.index] += 1

        # key1（失败次数最少）应该被选择最多
        assert selected_counts[1] > selected_counts[0]
        assert selected_counts[1] > selected_counts[2]


class TestOpenAIConfig:
    """测试 OpenAI 配置"""

    def test_get_effective_keys_with_single_key(self):
        """测试单个 key 配置"""
        from src.config.settings import OpenAIConfig

        config = OpenAIConfig(
            api_key="sk-test123",
            base_url="https://api.openai.com/v1",
            api_keys=None,
        )

        keys = config.get_effective_keys()
        assert len(keys) == 1
        assert keys[0]["api_key"] == "sk-test123"
        assert keys[0]["base_url"] == "https://api.openai.com/v1"

    def test_get_effective_keys_with_multiple_keys(self):
        """测试多个 keys 配置"""
        from src.config.settings import OpenAIConfig, OpenAIKeyConfig

        config = OpenAIConfig(
            api_key=None,
            base_url="https://api.openai.com/v1",
            api_keys=[
                OpenAIKeyConfig(api_key="sk-key1", daily_limit=150.0),
                OpenAIKeyConfig(api_key="sk-key2", daily_limit=150.0),
            ],
        )

        keys = config.get_effective_keys()
        assert len(keys) == 2
        assert keys[0]["api_key"] == "sk-key1"
        assert keys[1]["api_key"] == "sk-key2"

    def test_get_effective_keys_with_key_specific_base_url(self):
        """测试 key 特定的 base_url"""
        from src.config.settings import OpenAIConfig, OpenAIKeyConfig

        config = OpenAIConfig(
            api_key=None,
            base_url="https://api.openai.com/v1",
            api_keys=[
                OpenAIKeyConfig(
                    api_key="sk-key1",
                    daily_limit=150.0,
                    base_url="https://custom.api.com/v1",
                ),
            ],
        )

        keys = config.get_effective_keys()
        assert len(keys) == 1
        assert keys[0]["base_url"] == "https://custom.api.com/v1"

    def test_get_effective_keys_empty(self):
        """测试空配置"""
        from src.config.settings import OpenAIConfig

        config = OpenAIConfig(
            api_key=None,
            base_url="https://api.openai.com/v1",
            api_keys=None,
        )

        keys = config.get_effective_keys()
        assert len(keys) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
