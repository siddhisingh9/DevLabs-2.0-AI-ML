from typing import TypedDict,Annotated

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

from langgraph.prebuilt import ToolNode,tools_condition
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from langgraph_supervisor import create_supervisor

from tavily import TavilyClient

import os

from dotenv import load_dotenv
load_dotenv()

llm = init_chat_model("google_genai:gemini-3.1-flash-lite")

@tool
def web_search(query:str) -> str:
    """
    Search the internet for current information on the Topic
    """
    client = TavilyClient(os.getenv("TAVILY_API_KEY"))
    response = client.search(
        query=query,
        include_answer="basic",
        search_depth="advanced"
    )
    return response

@tool
def run_python(code: str) -> str:
    """Execute Python code and return the output."""
    # use a sandboxed executor in production
    import io, contextlib
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(code)
    return output.getvalue()


# Create Specialised Agents

researcher = create_react_agent(
    llm,
    tools=[web_search],
    name="researcher",
    prompt=(
        "You are a research specialist. Your job is to find accurate, "
        "up-to-date information.Use the Available Tools and get Accurate Data"
    )
)

coder = create_react_agent(
    llm,
    tools=[run_python],
    name="coder",
    prompt=(
        "You are a Python expert. Write clean, well-commented code. "
        "Always test your code before returning results. Handle edge cases."
    )
)

writer = create_react_agent(
    llm,
    tools=[],
    name="writer",
    prompt=(
        "You are a professional technical writer. Synthesise research "
        "and code into clear, structured reports. Use markdown formatting to write about that research text"
    )
)


# Create the Supervisor

supervisor = create_supervisor(
    agents=[researcher, coder, writer],
    model=llm,
    prompt=(
        """You are a supervisor.
            Rules:
                - Every research request MUST be handled by the researcher.
                - Every coding request MUST be handled by the coder.
                - After the researcher or coder finishes, you MUST delegate to the writer.
                - Never answer the user yourself.
                - The writer's response must always be the final response returned to the user.
                - Return the writer's response exactly as written without modifying, shortening, or rephrasing it.
        """
    ),
    output_mode="full_history"
).compile(checkpointer=MemorySaver())

print("\n ########################## \n")

message = input("Enter your Message : ")

print("\n ########################## \n")

result = supervisor.invoke(
    {"messages": [("user", message)]},
    {"configurable": {"thread_id": "project-1"}}
)
print(result["messages"][-1].content[0]["text"])

print("\n ########################## \n")
