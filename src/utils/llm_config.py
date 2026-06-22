"""
LLM provider配置工具。
默认使用通义千问（DashScope）兼容模式，通过 OpenAI/LiteLLM 兼容接口工作。
"""
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def _set_if_missing(key: str, value: Optional[str]):
    if value and not os.getenv(key):
        os.environ[key] = value


def configure_llm_provider():
    """
    根据环境变量配置默认的LLM提供商。
    - 默认 provider: qwen（通义千问，DashScope OpenAI兼容模式）
    - 兼容 LiteLLM 的代理模式（如设置 LITELLM_PROXY_URL）
    """
    provider = os.getenv("LLM_PROVIDER", "qwen").lower()
    proxy_url = os.getenv("LITELLM_PROXY_URL")

    if provider == "qwen":
        api_key = os.getenv("QWEN_API_KEY")
        base_url = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        model = os.getenv("QWEN_MODEL_NAME", "qwen-plus")

        if api_key:
            _set_if_missing("OPENAI_API_KEY", api_key)
        _set_if_missing("OPENAI_BASE_URL", proxy_url or base_url)
        _set_if_missing("OPENAI_MODEL_NAME", model)
        # 让 LiteLLM/兼容客户端知道使用的是千问
        _set_if_missing("LITELLM_PROVIDER", "qwen")
        logger.info("LLM provider已设置为通义千问 (Qwen)")

    elif provider == "openai":
        # 保持现有 OPENAI_* 配置；如需代理可通过 LITELLM_PROXY_URL 指向 LiteLLM 代理
        if proxy_url:
            _set_if_missing("OPENAI_BASE_URL", proxy_url)
        logger.info("LLM provider已设置为 OpenAI")

    else:
        logger.warning(f"未知的 LLM_PROVIDER={provider}，将沿用现有环境配置")

