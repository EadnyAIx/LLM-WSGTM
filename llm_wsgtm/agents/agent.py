import json
from typing import Optional

from .mcp_client import MCPClient
from .skills import SkillRegistry


class TopicModelAgent:
    def __init__(self, llm_client, skills: Optional[SkillRegistry]=None, rag_index=None, mcp_client: Optional[MCPClient]=None):
        self.llm_client = llm_client
        self.skills = skills or SkillRegistry()
        self.rag_index = rag_index
        self.mcp_client = mcp_client

    def execute(self, instruction, context=None):
        retrieved = self.rag_index.build_context(instruction, top_k=5) if self.rag_index is not None else ""
        available_skills = [{"name": skill.name, "description": skill.description, "tags": list(skill.tags)} for skill in self.skills.list()]
        mcp_tools = []
        if self.mcp_client is not None:
            try:
                mcp_tools = [{"name": tool.name, "description": tool.description, "input_schema": tool.input_schema} for tool in self.mcp_client.list_tools()]
            except Exception:
                mcp_tools = []
        routing_prompt = self._routing_prompt(instruction, context or "", retrieved, available_skills, mcp_tools)
        raw = self.llm_client.generate(routing_prompt, json_output=True, temperature=0.0)
        decision = json.loads(raw)
        action = decision.get("action", "respond")
        if action == "skill":
            return self.skills.invoke(decision["name"], **decision.get("arguments", {}))
        if action == "mcp" and self.mcp_client is not None:
            return self.mcp_client.call_tool(decision["name"], decision.get("arguments", {}))
        return decision.get("response", "")

    def _routing_prompt(self, instruction, context, retrieved, skills, mcp_tools):
        return f"""你是 LLM-WSGTM 项目的任务编排器。根据用户任务选择本地 Skill、MCP 工具或直接回答。
严格返回 JSON。
可选格式：
{{"action":"skill","name":"...","arguments":{{}}}}
{{"action":"mcp","name":"...","arguments":{{}}}}
{{"action":"respond","response":"..."}}

用户任务：
{instruction}

上下文：
{context}

RAG：
{retrieved}

Skills：
{json.dumps(skills, ensure_ascii=False)}

MCP Tools：
{json.dumps(mcp_tools, ensure_ascii=False)}
""".strip()
