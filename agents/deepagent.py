from dotenv import load_dotenv
load_dotenv()

import os
from pathlib import Path
from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI

from deepagents.backends import FilesystemBackend

base_dir = Path(__file__).parent

model = ChatOpenAI(
    model=os.getenv("LM_STUDIO_MODEL"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
)

deepagents_agent = create_deep_agent(
    model=model,
    memory=["AGENTS.md"],
    skills=["skills/"],
    backend=FilesystemBackend(root_dir=base_dir, virtual_mode=True)
)