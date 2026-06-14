# memory/agent.py
"""
LangChain Agent 管理器（新版超简洁版）
"""
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
import config
from memory.tools import ALL_TOOLS

_agent = None

def get_agent(system_prompt: str = None):
    if system_prompt is None:
        system_prompt = config.AGENT_SYSTEM_PROMPT
    model = init_chat_model(model="deepseek-chat")
    return create_agent(
        model=model,
        tools=ALL_TOOLS,
        system_prompt=system_prompt
    )