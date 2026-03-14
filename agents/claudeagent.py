from dotenv import load_dotenv
load_dotenv()

from collections.abc import AsyncIterator

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    query,
)


async def claude_agent_stream(prompt: str) -> AsyncIterator[dict]:
    async for message in query(
        prompt=prompt,
        options=ClaudeAgentOptions(
            allowed_tools=["Read", "Edit", "Glob"],
            ),
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    yield {"type": "assistant_text", "text": block.text}
                elif isinstance(block, ToolUseBlock):
                    yield {"type": "tool_use", "name": block.name, "input": block.input, "id": block.id}
        elif isinstance(message, ResultMessage):
            yield {"type": "result", "subtype": message.subtype}
