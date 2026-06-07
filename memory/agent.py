# memory/agent.py
"""
LangChain Agent 管理器（新版超简洁版）
"""
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
import config
from memory.tools import ALL_TOOLS

_agent = None

def get_agent():
    """获取全局 Agent 单例"""
    global _agent
    if _agent is None:
        # 初始化模型，自动读取环境变量 DEEPSEEK_API_KEY
        model = init_chat_model(model="deepseek-chat")
        # 创建智能体，传入系统提示词和工具列表
        _agent = create_agent(
            model=model,
            tools=ALL_TOOLS,
            system_prompt=config.AGENT_SYSTEM_PROMPT,
        )
    return _agent