"""
LLM 接入配置：所有模型调用共用这一份 (model, base_url, api_key, timeout)。

"""
import os

FALLBACK_MODEL = "deepseek-flash"
FALLBACK_BASE_URL = "https://api.deepseek.com"
FALLBACK_TIMEOUT = 180


def model() -> str:
    return os.getenv("LLM_MODEL") or FALLBACK_MODEL


def options() -> dict:
    """塞给 StreamFn 的 option：api_key / base_url / timeout。"""
    base_url = (os.getenv("LLM_BASE_URL") or FALLBACK_BASE_URL).rstrip("/")
    timeout = int(os.getenv("LLM_TIMEOUT") or FALLBACK_TIMEOUT)
    return {
        "api_key": os.getenv("LLM_API_KEY") or "",
        "base_url": base_url,
        "timeout": timeout,
    }
