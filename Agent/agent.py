# memory/agent.py
"""
LangChain Agent 管理器（新版）
支持注入物品习惯摘要
"""
import os
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
import config
from Agent.tools import ALL_TOOLS

_agent = None

def get_agent(system_prompt: str = None):
    if system_prompt is None:
        system_prompt = config.AGENT_SYSTEM_PROMPT
    model = init_chat_model(
        model="deepseek-chat",
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.DEEPSEEK_BASE_URL
    )
    return create_agent(
        model=model,
        tools=ALL_TOOLS,
        system_prompt=system_prompt
    )