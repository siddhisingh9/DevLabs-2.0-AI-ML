# LangGraph & Multi-Agent Systems — Complete Study Notes

> **How to use this file:** Part 1 is a fast, dense revision of LangGraph fundamentals. Part 2 is the deep-dive into multi-agent systems. Code blocks are copy-paste ready. Every section ends with a "Key Takeaways" box.

---

# PART 1 — LangGraph Fundamentals (Revision)

---

## 1.1 Why LangGraph Exists

Traditional LangChain "chains" are linear: A → B → C. That's fine for pipelines, but **real agents need to**:

- **Loop** — retry a failed tool call, ask a follow-up question
- **Branch** — take different paths based on what the LLM decided
- **Remember** — carry context across many turns without re-prompting
- **Pause** — wait for human approval mid-execution

None of these are natural in a chain. LangGraph solves this by modelling agent logic as a **directed graph** — a set of nodes connected by edges, where loops, branches, and conditional routing are first-class citizens.

> **Mental model:** A LangGraph agent is a state machine. Nodes are states. Edges are transitions. The LLM decides which transition to take.

---

## 1.2 The Three Core Primitives

Everything in LangGraph is built from exactly three things:

| Primitive | What it is | Analogy |
|-----------|------------|---------|
| **State** | A TypedDict that holds all data flowing through the graph | Shared whiteboard |
| **Node** | A Python function that reads state and returns updates | Worker at the whiteboard |
| **Edge** | A connection between nodes, optionally conditional | Road between workers |

That's it. Every feature — memory, multi-agent, human-in-the-loop — is built on top of these three primitives.

---

## 1.3 State — The Shared Whiteboard

State is a `TypedDict` class. Every node in the graph receives the **full state** as input and returns a **dict of updates** (only the fields that changed).

```python
from typing import TypedDict, Annotated
from langgraph.graph import add_messages

class AgentState(TypedDict):
    # add_messages is a REDUCER — it appends new messages instead of overwriting
    messages: Annotated[list, add_messages]
    
    # Regular fields — last writer wins
    user_query: str
    tool_results: list
    final_answer: str | None
    iteration_count: int
```

### What is a Reducer?

Without a reducer, every node that updates `messages` would **overwrite** the entire list. That's almost never what you want.

The `add_messages` reducer tells LangGraph: "When a node returns new messages, **append** them to the existing list rather than replace it."

```python
# Node returns a partial update — only changed fields
def my_node(state: AgentState) -> dict:
    return {
        "messages": [AIMessage(content="Hello")],  # appended, not replaced
        "iteration_count": state["iteration_count"] + 1  # overwritten (no reducer)
    }
```

You can write your own reducers for any merge logic you need:

```python
import operator

class ResearchState(TypedDict):
    # Using operator.add as reducer — concatenates lists
    search_results: Annotated[list, operator.add]
    
    # Custom reducer function
    errors: Annotated[list, lambda old, new: old + new if new else old]
```

> **Key Takeaways — State**
> - State is the only way nodes "communicate" — they don't call each other
> - Nodes return dicts, not the full state object
> - Reducers control how updates merge into existing state
> - Design your state schema first — it's your agent's data contract

---

## 1.4 Nodes — Units of Work

A node is **just a Python function**. It takes state as input, does something (LLM call, tool execution, logic, DB query, anything), and returns a dict of state updates.

```python
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-sonnet-4-5")

# ── LLM Node ─────────────────────────────────────────────────────────────────
def llm_node(state: AgentState) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}  # add_messages reducer appends this

# ── Router Node — decides what happens next ───────────────────────────────────
def router(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "use_tools"
    return "end"

# ── Async nodes are supported too ────────────────────────────────────────────
async def async_llm_node(state: AgentState) -> dict:
    response = await llm.ainvoke(state["messages"])
    return {"messages": [response]}
```

### Built-in Nodes

LangGraph ships with ready-made nodes for the most common patterns:

```python
from langgraph.prebuilt import ToolNode, tools_condition

# ToolNode: automatically executes whatever tool_calls the LLM requested
tool_node = ToolNode(tools=[search, calculator, read_file])

# tools_condition: returns "tools" if there are tool_calls, "__end__" otherwise
# Use this as your routing function instead of writing your own
```

> **Key Takeaways — Nodes**
> - Nodes are plain Python functions — test them independently
> - Return only the fields that changed, not the whole state
> - `ToolNode` handles all tool execution boilerplate for you
> - Async nodes work out of the box — just use `async def`

---

## 1.5 Edges — Control Flow

Edges define which node runs after the current one.

### Normal Edge — Always go here

```python
graph.add_edge("llm_node", "tool_node")  # always goes to tool_node after llm_node
graph.add_edge(START, "first_node")       # START is where the graph begins
graph.add_edge("last_node", END)          # END terminates the graph
```

### Conditional Edge — Go somewhere based on state

```python
# The routing function inspects state and returns a string key
def route_after_llm(state: AgentState) -> str:
    if state["messages"][-1].tool_calls:
        return "use_tools"
    return "done"

# Map that string key to a node name
graph.add_conditional_edges(
    "llm_node",           # from this node
    route_after_llm,      # call this function to decide
    {
        "use_tools": "tool_node",   # "use_tools" → go to tool_node
        "done":      END            # "done" → terminate
    }
)

# Shorthand: if your function returns a node name directly (not a key),
# you can omit the mapping dict
graph.add_conditional_edges("llm_node", route_after_llm)
```

### Fan-out — Multiple edges from one node (parallel execution)

```python
# All three run simultaneously when dispatcher completes
graph.add_edge("dispatcher", "research_agent")
graph.add_edge("dispatcher", "analysis_agent")
graph.add_edge("dispatcher", "writer_agent")
```

> **Key Takeaways — Edges**
> - `add_edge` = always go there, `add_conditional_edges` = decide at runtime
> - The routing function is just a Python function returning a string
> - Multiple edges from one node = parallel execution (fan-out)
> - Use `START` and `END` constants, not strings

---

## 1.6 Building and Compiling a Graph

```python
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool

# 1. Define tools
@tool
def web_search(query: str) -> str:
    """Search the web for current information."""
    return f"Results for: {query}"  # replace with real implementation

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely."""
    return str(eval(expression))  # use a safe eval in production

tools = [web_search, calculator]
llm_with_tools = llm.bind_tools(tools)

# 2. Define nodes
def agent_node(state: AgentState) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# 3. Build the graph
graph = StateGraph(AgentState)

graph.add_node("agent",  agent_node)
graph.add_node("tools",  ToolNode(tools))

# 4. Add edges (the ReAct loop)
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition)  # built-in router
graph.add_edge("tools", "agent")                       # loop back

# 5. Compile — this validates the graph and returns a runnable
app = graph.compile()

# 6. Run
result = app.invoke({
    "messages": [("user", "What is the square root of 144?")],
    "user_query": "square root of 144",
    "tool_results": [],
    "final_answer": None,
    "iteration_count": 0
})
print(result["messages"][-1].content)
```

---

## 1.7 Memory & Checkpointing

By default, every `invoke()` call is stateless — the graph forgets everything after it finishes. Checkpointers fix this.

### How Checkpointing Works

LangGraph saves the full state after **every node execution** to a persistent store. Each conversation is identified by a `thread_id`. Same `thread_id` = same memory.

```python
from langgraph.checkpoint.memory import MemorySaver      # in-memory (dev)
from langgraph.checkpoint.sqlite import SqliteSaver      # SQLite (prod-lite)
# pip install langgraph-checkpoint-postgres
from langgraph.checkpoint.postgres import PostgresSaver  # PostgreSQL (prod)

# Attach checkpointer at compile time
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

# Thread config — identifies the session
config = {"configurable": {"thread_id": "user-alice-session-1"}}

# Turn 1
app.invoke({"messages": [("user", "My name is Alice")]}, config)

# Turn 2 — remembers turn 1 automatically
result = app.invoke({"messages": [("user", "What's my name?")]}, config)
# → "Your name is Alice."

# Different thread = fresh start
config2 = {"configurable": {"thread_id": "user-bob-session-1"}}
app.invoke({"messages": [("user", "What's my name?")]}, config2)
# → "I don't know your name yet."
```

### Short-Term vs Long-Term Memory

| | Short-Term | Long-Term |
|---|---|---|
| **Scope** | Within one `thread_id` | Across all `thread_id`s |
| **Where** | State → `messages` list | LangGraph Store API |
| **Managed by** | Checkpointer | External DB / vector store |
| **Use case** | Conversation history | User profiles, knowledge base |
| **Limit** | Context window | Storage capacity |

```python
# Long-term memory with the Store API
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

def memory_node(state: AgentState, store) -> dict:
    # Read user's long-term preferences
    memories = store.search(("user", "alice", "preferences"))
    
    # Write a new memory
    store.put(
        ("user", "alice", "preferences"),
        key="communication_style",
        value={"style": "formal", "language": "English"}
    )
    return {}

app = graph.compile(checkpointer=checkpointer, store=store)
```

### Managing Context Window Growth

Long conversations fill the context window. Trim or summarise proactively:

```python
from langchain_core.messages import trim_messages, RemoveMessage

def trim_node(state: AgentState) -> dict:
    # Keep only the last 20 messages
    trimmed = trim_messages(
        state["messages"],
        max_tokens=4000,
        strategy="last",
        token_counter=llm,
        include_system=True
    )
    # Return RemoveMessage objects to delete specific messages
    to_remove = [RemoveMessage(id=m.id) for m in state["messages"][:-20]]
    return {"messages": to_remove}
```

> **Key Takeaways — Memory**
> - `thread_id` is your session ID — one per user conversation
> - `MemorySaver` for dev, `SqliteSaver` for single-machine prod, `PostgresSaver` for distributed
> - Checkpointing also enables human-in-the-loop and time-travel debugging
> - Long-term memory (Store API) is separate from short-term (checkpointer)

---

## 1.8 Human-in-the-Loop

LangGraph can pause mid-execution, wait for a human, and resume from exactly where it stopped — all because of checkpointing.

```python
# Pause BEFORE tool_node executes — human approves the tool call first
app = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["tools"]
)

config = {"configurable": {"thread_id": "session-1"}}

# Run until the interrupt
app.invoke({"messages": [("user", "Search for latest AI news")]}, config)
# → Graph pauses. State is saved. Tool call is pending.

# Inspect what the LLM wants to do
current_state = app.get_state(config)
print(current_state.values["messages"][-1].tool_calls)
# → [{"name": "web_search", "args": {"query": "latest AI news 2025"}}]

# Human approves — resume with None (no new input, just continue)
result = app.invoke(None, config)

# Human rejects — edit the state before resuming
app.update_state(config, {"messages": [("user", "Actually, search for ML news instead")]})
result = app.invoke(None, config)
```

### The `Command` Pattern for Dynamic Routing

```python
from langgraph.types import Command

def approval_node(state: AgentState) -> Command:
    # This node can both update state AND decide where to go next
    if state.get("human_approved"):
        return Command(goto="execute_node", update={"status": "approved"})
    else:
        return Command(goto="rejection_node", update={"status": "rejected"})
```

> **Key Takeaways — Human-in-the-Loop**
> - `interrupt_before` / `interrupt_after` set at compile time
> - Requires a checkpointer — state must be saved somewhere to resume
> - Resume by calling `invoke(None, config)` with the same `thread_id`
> - `update_state()` lets you edit state before resuming

---

## 1.9 Streaming Output

Streaming is critical for good UX — users see output token by token instead of waiting for the full response.

```python
# Stream state updates after each node
for chunk in app.stream(
    {"messages": [("user", "Explain quantum computing")]},
    config,
    stream_mode="updates"    # or "values", "messages", "debug"
):
    node_name, state_update = next(iter(chunk.items()))
    print(f"[{node_name}] → {list(state_update.keys())}")

# Stream individual tokens as they generate
async for event in app.astream_events(inputs, config, version="v2"):
    if event["event"] == "on_chat_model_stream":
        token = event["data"]["chunk"].content
        print(token, end="", flush=True)
```

| `stream_mode` | What you get |
|---|---|
| `"values"` | Full state after each node |
| `"updates"` | Only what changed in each node |
| `"messages"` | Individual LLM tokens as they stream |
| `"debug"` | Full execution trace + metadata |

---

# PART 2 — Multi-Agent Systems in LangGraph

---

## 2.1 What Is a Multi-Agent System?

A multi-agent system is a collection of individual AI agents that **collaborate** to complete a task that would be too complex, too long, or too broad for any single agent.

Each agent is:
- A complete LangGraph graph (with its own state, nodes, edges)
- Specialised for a specific domain or task
- Connected to other agents through shared state, handoffs, or a coordinator

### When to Use Multi-Agent (and When Not To)

**Use multi-agent when you have:**

| Signal | Why Multi-Agent Helps |
|--------|----------------------|
| More than ~5-7 tools | Agents with too many tools have diluted attention — split into specialists |
| Tasks that can run in parallel | Different agents can work simultaneously, cutting wall-clock time |
| Subtasks requiring different models | Use GPT-4 for reasoning, a cheap model for classification, vision model for images |
| Long tasks exceeding one context window | Split across agents, each working with full context on its slice |
| Quality pipelines (write → review → refine) | Critic agents catch errors the writer agent misses |
| Independent team ownership | Different teams own different agents |

**Don't use multi-agent when:**
- A single agent with a clear prompt and 3-5 tools can do the job
- You're adding complexity for its own sake
- You haven't built and debugged a single-agent version first

> **Rule of thumb:** Build the single-agent version. If it struggles, identify *why* — and then split along those exact pain points.

---

## 2.2 The Four Multi-Agent Architectures

LangGraph supports four primary patterns. Each is a different answer to the question: *"Who decides what agent runs next?"*

| Architecture | Who decides next step | Best for |
|---|---|---|
| **Supervisor** | A central LLM orchestrator | Sequential pipelines, strict quality control |
| **Swarm / Handoff** | Each agent decides who to hand off to | Open-ended tasks, peer collaboration |
| **Subgraph** | Parent graph structure | Reusable agents, team-owned components |
| **Parallel / Map-Reduce** | Graph structure (fan-out then fan-in) | Independent subtasks, batch processing |

---

## 2.3 Architecture 1: Supervisor

### Concept

One "boss" LLM sees the overall task and delegates to specialist workers. The supervisor sees all results and decides what to do next. Workers do not communicate with each other.

```
User Query
    ↓
 SUPERVISOR (LLM)
  ↙    ↓    ↘
R     C     W        ← Researcher, Coder, Writer
  ↘    ↑    ↗
 SUPERVISOR (LLM)    ← sees all results, decides next step
    ↓
  Answer
```

### Implementation with `langgraph-supervisor`

```bash
pip install langgraph-supervisor
```

```python
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool

llm = ChatAnthropic(model="claude-sonnet-4-5")

# ── Define tools for each specialist ─────────────────────────────────────────

@tool
def web_search(query: str) -> str:
    """Search the internet for current information on a topic."""
    # integrate with Tavily, SerpAPI, etc.
    return f"Search results for '{query}': [results here]"

@tool
def run_python(code: str) -> str:
    """Execute Python code and return the output."""
    # use a sandboxed executor in production
    import io, contextlib
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(code)
    return output.getvalue()

@tool
def read_file(path: str) -> str:
    """Read the contents of a file."""
    with open(path) as f:
        return f.read()

# ── Create specialist agents ──────────────────────────────────────────────────

researcher = create_react_agent(
    llm,
    tools=[web_search, read_file],
    name="researcher",
    prompt=(
        "You are a research specialist. Your job is to find accurate, "
        "up-to-date information. Always cite your sources. Be thorough."
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
    tools=[],  # writer needs no tools — just crafts prose from context
    name="writer",
    prompt=(
        "You are a professional technical writer. Synthesise research "
        "and code into clear, structured reports. Use markdown formatting."
    )
)

# ── Create the supervisor ─────────────────────────────────────────────────────

supervisor = create_supervisor(
    llm,
    agents=[researcher, coder, writer],
    prompt=(
        "You are a project manager coordinating a research and analysis team. "
        "Break down the user's request, delegate to the right specialist, "
        "review their output, and continue until the task is fully complete. "
        "Always delegate writing the final report to the writer agent."
    ),
    # output_mode controls how agent outputs flow back to supervisor:
    # "last_message" — only the agent's final response (default, saves tokens)
    # "full_history" — everything the agent did (more context, more tokens)
    output_mode="last_message"
).compile(checkpointer=MemorySaver())

# ── Run ───────────────────────────────────────────────────────────────────────

result = supervisor.invoke(
    {"messages": [("user", "Analyse the top 3 Python web frameworks and write a comparison report")]},
    {"configurable": {"thread_id": "project-1"}}
)
print(result["messages"][-1].content)
```

### Building a Supervisor from Scratch (Without the Library)

Understanding the manual approach reveals what the library abstracts:

```python
from langgraph.graph import StateGraph, START, END
from typing import Literal
from pydantic import BaseModel

# Supervisor decides next action via structured output
class SupervisorDecision(BaseModel):
    next: Literal["researcher", "coder", "writer", "FINISH"]
    reasoning: str

def supervisor_node(state: AgentState) -> dict:
    system_prompt = """You are a supervisor managing: researcher, coder, writer.
    Given the conversation, decide who should act next, or FINISH if done.
    Return your decision as JSON."""
    
    structured_llm = llm.with_structured_output(SupervisorDecision)
    decision = structured_llm.invoke([
        SystemMessage(content=system_prompt),
        *state["messages"]
    ])
    return {"next_agent": decision.next}

def route_to_agent(state: AgentState) -> str:
    return state["next_agent"]

graph = StateGraph(AgentState)
graph.add_node("supervisor", supervisor_node)
graph.add_node("researcher", researcher_node)
graph.add_node("coder", coder_node)
graph.add_node("writer", writer_node)

graph.add_edge(START, "supervisor")
graph.add_conditional_edges(
    "supervisor",
    route_to_agent,
    {"researcher": "researcher", "coder": "coder",
     "writer": "writer", "FINISH": END}
)
# All workers report back to supervisor
graph.add_edge("researcher", "supervisor")
graph.add_edge("coder", "supervisor")
graph.add_edge("writer", "supervisor")

app = graph.compile(checkpointer=MemorySaver())
```

### Supervisor Pros and Cons

| ✅ Pros | ❌ Cons |
|---------|---------|
| Easy to reason about — one entity in control | Supervisor is a bottleneck — every step goes through it |
| Easy to debug — clear chain of delegation | Extra LLM call per step (supervisor overhead) |
| Workers are isolated — easy to swap out | Supervisor can get confused with many agents |
| Good for sequential, dependent steps | Not great for tasks where workers need to collaborate directly |

---

## 2.4 Architecture 2: Swarm (Decentralised / Handoff)

### Concept

No central controller. Each agent decides, based on the current context, whether to continue working or **hand off** to a peer. Agents are peers, not a hierarchy.

```
User Query
    ↓
Agent A (active)
    ↓  [decides to hand off]
Agent B (active)
    ↓  [decides to hand off]
Agent C (active)
    ↓  [completes task]
  Answer
```

### Implementation with `langgraph-swarm`

```bash
pip install langgraph-swarm
```

```python
from langgraph_swarm import create_swarm, create_handoff_tool
from langgraph.prebuilt import create_react_agent

# ── Create handoff tools ──────────────────────────────────────────────────────
# Each agent gets tools to transfer control to specific peers

transfer_to_analyst = create_handoff_tool(
    agent_name="analyst",
    description="Transfer to the analyst when you have raw data that needs interpretation or statistical analysis."
)

transfer_to_writer = create_handoff_tool(
    agent_name="writer",
    description="Transfer to the writer when analysis is complete and a final report needs to be written."
)

transfer_to_researcher = create_handoff_tool(
    agent_name="researcher",
    description="Transfer to the researcher when additional information or data needs to be gathered."
)

# ── Create agents with handoff tools ────────────────────────────────────────

researcher = create_react_agent(
    llm,
    tools=[web_search, transfer_to_analyst],
    name="researcher",
    prompt="You gather information. When you have enough data, transfer to the analyst."
)

analyst = create_react_agent(
    llm,
    tools=[run_python, transfer_to_writer, transfer_to_researcher],
    name="analyst",
    prompt=(
        "You interpret data and run analysis. If you need more data, "
        "transfer back to researcher. When analysis is done, transfer to writer."
    )
)

writer = create_react_agent(
    llm,
    tools=[transfer_to_researcher, transfer_to_analyst],
    name="writer",
    prompt=(
        "You write the final report. If anything is unclear, "
        "transfer back to researcher or analyst for clarification."
    )
)

# ── Create the swarm ──────────────────────────────────────────────────────────

swarm = create_swarm(
    agents=[researcher, analyst, writer],
    default_active_agent="researcher"  # first agent to activate
).compile(checkpointer=MemorySaver())

result = swarm.invoke(
    {"messages": [("user", "Research AI chip performance trends and write a summary")]},
    {"configurable": {"thread_id": "swarm-session-1"}}
)
```

### Building Handoff from Scratch

```python
from langgraph.types import Command

# Handoff is just a node that returns a Command routing to another agent
def transfer_to_analyst(state: AgentState) -> Command:
    return Command(
        goto="analyst",
        update={
            "messages": [
                ToolMessage(
                    content="Transferring to analyst with research results",
                    tool_call_id="handoff"
                )
            ],
            "active_agent": "analyst"
        }
    )

# Agents can navigate to any other node in the parent graph
def researcher_agent(state: AgentState) -> dict | Command:
    response = llm_with_tools.invoke(state["messages"])
    
    # If the LLM called a handoff tool, execute the handoff
    if response.tool_calls and response.tool_calls[0]["name"] == "transfer_to_analyst":
        return Command(goto="analyst", update={"messages": [response]})
    
    return {"messages": [response]}
```

### Swarm Pros and Cons

| ✅ Pros | ❌ Cons |
|---------|---------|
| No bottleneck — agents work peer-to-peer | Harder to trace execution path |
| Very flexible — any agent can hand off to any other | Can loop unexpectedly without termination logic |
| Natural for open-ended, exploratory tasks | Harder to guarantee all required steps happen |
| Each agent retains full control of its domain | Debugging requires full conversation history |

---

## 2.5 Architecture 3: Subgraphs

### Concept

A compiled LangGraph is just a Python object. You can use it **as a node** inside another graph. This is called a subgraph — and it's how you build large, modular, maintainable multi-agent systems.

```
Parent Graph
├── Node: "intake"        (plain node)
├── Node: "research_agent" ← compiled subgraph
├── Node: "analysis_agent" ← compiled subgraph  
├── Node: "writer_agent"   ← compiled subgraph
└── Node: "output"        (plain node)
```

### State Schema Compatibility

Subgraphs can have different state schemas from their parent. LangGraph handles the boundary translation automatically — but only for fields with the same name.

```python
# ── Subgraph state (own schema) ──────────────────────────────────────────────
class ResearchState(TypedDict):
    query: str                                      # input from parent
    search_results: Annotated[list, operator.add]   # internal
    summary: str                                    # output to parent

# ── Subgraph definition ───────────────────────────────────────────────────────
research_graph = StateGraph(ResearchState)
research_graph.add_node("search",    do_web_search)
research_graph.add_node("summarize", do_summarize)
research_graph.add_edge(START, "search")
research_graph.add_edge("search", "summarize")
research_graph.add_edge("summarize", END)

# Compile — this is now a runnable you can use anywhere
research_agent = research_graph.compile()

# ── Parent graph state ────────────────────────────────────────────────────────
class ParentState(TypedDict):
    messages: Annotated[list, add_messages]
    query: str       # ← shared field name: automatically passed to subgraph
    summary: str     # ← shared field name: automatically received from subgraph
    final_report: str

# ── Parent graph ──────────────────────────────────────────────────────────────
parent = StateGraph(ParentState)
parent.add_node("intake",    intake_node)
parent.add_node("research",  research_agent)   # ← subgraph as a node!
parent.add_node("write",     write_node)

parent.add_edge(START, "intake")
parent.add_edge("intake", "research")
parent.add_edge("research", "write")
parent.add_edge("write", END)

app = parent.compile(checkpointer=MemorySaver())
```

### State Transformation at Subgraph Boundaries

If your field names don't match, use a wrapper node to translate:

```python
def prep_for_research(state: ParentState) -> dict:
    """Transform parent state into research agent input."""
    return {
        "query": state["messages"][-1].content,
        "search_results": [],
        "summary": ""
    }

def extract_from_research(state: ParentState) -> dict:
    """Extract research agent output back to parent state."""
    # The subgraph's output is available in the parent's state
    return {"research_summary": state.get("summary", "")}

parent.add_node("prep_research", prep_for_research)
parent.add_node("run_research",  research_agent)
parent.add_node("post_research", extract_from_research)
```

### Subgraph Pros and Cons

| ✅ Pros | ❌ Cons |
|---------|---------|
| Maximum modularity and reusability | State schema coordination can be tricky |
| Each subgraph is independently testable | More upfront design work |
| Teams can own and deploy individual subgraphs | Debugging spans multiple graph levels |
| Clean separation of concerns | Overhead from state translation at boundaries |

---

## 2.6 Architecture 4: Parallel (Fan-Out / Fan-In)

### Concept

When subtasks are independent, run them simultaneously. LangGraph handles this with multiple edges from one node — each target runs in parallel. A "fan-in" node waits for all parallel branches to complete before proceeding.

```
                  ┌→ Agent A ─┐
User → Dispatcher─┼→ Agent B ─┼→ Aggregator → Answer
                  └→ Agent C ─┘
```

### Basic Parallel Execution

```python
from langgraph.graph import StateGraph, START, END
from typing import Annotated
import operator

class ParallelResearchState(TypedDict):
    topic: str
    # Annotated with operator.add so results from all branches are merged
    news_results:     Annotated[list, operator.add]
    academic_results: Annotated[list, operator.add]
    social_results:   Annotated[list, operator.add]
    final_report: str

def news_agent(state: ParallelResearchState) -> dict:
    results = search_news(state["topic"])
    return {"news_results": results}

def academic_agent(state: ParallelResearchState) -> dict:
    results = search_arxiv(state["topic"])
    return {"academic_results": results}

def social_agent(state: ParallelResearchState) -> dict:
    results = search_twitter(state["topic"])
    return {"social_results": results}

def aggregator(state: ParallelResearchState) -> dict:
    # All three agents have finished by the time this runs
    all_results = {
        "news":     state["news_results"],
        "academic": state["academic_results"],
        "social":   state["social_results"]
    }
    report = llm.invoke(f"Synthesise these research results: {all_results}")
    return {"final_report": report.content}

graph = StateGraph(ParallelResearchState)
graph.add_node("news_agent",     news_agent)
graph.add_node("academic_agent", academic_agent)
graph.add_node("social_agent",   social_agent)
graph.add_node("aggregator",     aggregator)

graph.add_edge(START, "news_agent")      # fan-out:
graph.add_edge(START, "academic_agent")  # all three start simultaneously
graph.add_edge(START, "social_agent")    # when graph begins

graph.add_edge("news_agent",     "aggregator")  # fan-in:
graph.add_edge("academic_agent", "aggregator")  # aggregator waits for
graph.add_edge("social_agent",   "aggregator")  # all three to finish

graph.add_edge("aggregator", END)
app = graph.compile()
```

### Map-Reduce Pattern

For processing a list of items in parallel:

```python
from langgraph.constants import Send

class MapReduceState(TypedDict):
    documents: list[str]                         # input
    summaries: Annotated[list, operator.add]     # collected from map step
    final_summary: str                           # produced by reduce step

def map_node(state: MapReduceState) -> list[Send]:
    # Return a list of Send objects — one per document
    # Each spawns a parallel "summarise_doc" execution
    return [
        Send("summarise_doc", {"document": doc, "summaries": []})
        for doc in state["documents"]
    ]

def summarise_doc(state: dict) -> dict:
    # Receives a single document; runs in parallel with all other docs
    summary = llm.invoke(f"Summarise in 2 sentences: {state['document']}")
    return {"summaries": [summary.content]}

def reduce_node(state: MapReduceState) -> dict:
    # Runs once all summaries are collected
    final = llm.invoke(f"Combine these summaries: {state['summaries']}")
    return {"final_summary": final.content}

graph = StateGraph(MapReduceState)
graph.add_node("map",          map_node)
graph.add_node("summarise_doc", summarise_doc)
graph.add_node("reduce",       reduce_node)

graph.add_conditional_edges("map", lambda s: s, ["summarise_doc"])  # fan-out via Send
graph.add_edge("summarise_doc", "reduce")
graph.add_edge(START, "map")
graph.add_edge("reduce", END)
```

---

## 2.7 Agent Communication Patterns

How agents share information is as important as the architecture. Four patterns, ordered from simplest to most powerful:

### Pattern 1: Shared State

All agents read and write to the same TypedDict. No explicit messaging. The simplest option.

```python
class SharedState(TypedDict):
    messages: Annotated[list, add_messages]
    research_notes: str        # researcher writes, writer reads
    code_output: str           # coder writes, writer reads
    draft_report: str          # writer writes, reviewer reads
    review_feedback: str       # reviewer writes, writer reads (revision loop)

# Any agent can read/write any field
def researcher(state: SharedState) -> dict:
    notes = do_research(state["messages"][-1].content)
    return {"research_notes": notes}

def writer(state: SharedState) -> dict:
    # Reads from researcher's output
    draft = write_report(state["research_notes"], state["code_output"])
    return {"draft_report": draft}
```

**Best for:** Tightly coupled agents in a clear pipeline. **Avoid when** many agents write to the same fields (state becomes messy).

### Pattern 2: Message Passing

Agents communicate through the `messages` list — the natural conversation history.

```python
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

def researcher(state: AgentState) -> dict:
    research = do_research(state["messages"])
    # Communicate by appending a message
    return {"messages": [AIMessage(
        content=f"Research complete. Key findings: {research}",
        name="researcher"
    )]}

def analyst(state: AgentState) -> dict:
    # Read researcher's message from history
    research_msg = next(m for m in state["messages"] if getattr(m, "name", "") == "researcher")
    analysis = analyse(research_msg.content)
    return {"messages": [AIMessage(content=analysis, name="analyst")]}
```

**Best for:** Natural conversation flows, when full history matters. **Avoid when** the context window fills up quickly.

### Pattern 3: Handoff Tools (Swarm-Style)

An agent signals readiness to transfer control by calling a handoff tool. The most explicit pattern.

```python
from langgraph_swarm import create_handoff_tool

# Creating the handoff tool automatically creates a node for routing
transfer_to_reviewer = create_handoff_tool(
    agent_name="reviewer",
    description=(
        "Transfer to the reviewer after completing a draft. "
        "Include a summary of what you wrote and what you need reviewed."
    )
)

# The agent has this tool alongside its regular tools
writer = create_react_agent(
    llm,
    tools=[format_text, transfer_to_reviewer],
    name="writer"
)
```

**Best for:** Open-ended workflows, peer-to-peer collaboration, when agents decide routing themselves.

### Pattern 4: Command Objects

The most powerful pattern. A node returns a `Command` that both updates state AND directs the graph where to go next.

```python
from langgraph.types import Command
from typing import Literal

def reviewer_node(state: AgentState) -> Command[Literal["writer", "publisher"]]:
    review = llm.invoke(state["messages"])
    
    if "APPROVED" in review.content:
        return Command(
            goto="publisher",
            update={
                "messages": [review],
                "status": "approved",
                "approved_at": datetime.now().isoformat()
            }
        )
    else:
        return Command(
            goto="writer",
            update={
                "messages": [review],
                "status": "needs_revision",
                "revision_notes": review.content
            }
        )
```

**Best for:** Complex routing with simultaneous state updates, when the routing decision and state update are tightly coupled.

---

## 2.8 Full Example: Research Assistant with Supervisor + Memory

A complete, production-style multi-agent research assistant combining everything covered:

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from typing import TypedDict, Annotated
from langgraph.graph import add_messages
import operator

# ── LLM ──────────────────────────────────────────────────────────────────────
llm = ChatAnthropic(model="claude-sonnet-4-5", temperature=0)

# ── Tools ─────────────────────────────────────────────────────────────────────
@tool
def web_search(query: str) -> str:
    """Search the web for recent information. Use for current events and latest data."""
    # Integrate with Tavily: from tavily import TavilyClient
    return f"[Web results for '{query}']"

@tool
def arxiv_search(query: str) -> str:
    """Search academic papers on arXiv. Use for research papers and technical content."""
    return f"[arXiv papers for '{query}']"

@tool
def run_code(code: str) -> str:
    """Execute Python code for data analysis, calculations, or data processing."""
    import io, contextlib
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, {})
        return buf.getvalue() or "Code executed successfully (no output)"
    except Exception as e:
        return f"Error: {e}"

@tool
def generate_chart(data: str, chart_type: str, title: str) -> str:
    """Generate a data visualisation. Returns a description of the chart created."""
    return f"[Chart '{title}' ({chart_type}) generated from data]"

# ── Specialist Agents ──────────────────────────────────────────────────────────
researcher = create_react_agent(
    llm,
    tools=[web_search, arxiv_search],
    name="researcher",
    prompt="""You are a rigorous research specialist.
    
    Your responsibilities:
    - Search for accurate, up-to-date information from multiple sources
    - Cross-reference facts across sources
    - Note the date and credibility of each source
    - Flag any conflicting information
    - Return structured, well-organised findings
    
    Always search at least 2-3 times with different query angles before concluding."""
)

analyst = create_react_agent(
    llm,
    tools=[run_code, generate_chart],
    name="analyst",
    prompt="""You are a data analysis expert.
    
    Your responsibilities:
    - Interpret research findings quantitatively where possible
    - Run calculations and data analysis in Python
    - Generate visualisations to support conclusions
    - Identify trends, patterns, and statistical insights
    - Provide confidence levels for your conclusions
    
    Always validate your code runs correctly before presenting results."""
)

writer = create_react_agent(
    llm,
    tools=[],
    name="writer",
    prompt="""You are a senior technical writer specialising in research reports.
    
    Your responsibilities:
    - Synthesise research and analysis into clear, structured reports
    - Use markdown formatting with proper headers, sections, and lists
    - Include an executive summary at the top
    - Cite sources inline (use [Source: ...] format)
    - End with key conclusions and recommended next steps
    - Calibrate technical depth to the audience
    
    Write for a technically-literate but non-specialist audience unless instructed otherwise."""
)

# ── Supervisor ────────────────────────────────────────────────────────────────
SUPERVISOR_PROMPT = """You are a research project coordinator managing a specialist team.

Your team:
- researcher: Finds information from web and academic sources
- analyst: Analyses data, runs code, creates visualisations  
- writer: Writes final reports and summaries

Your workflow for research tasks:
1. Send to researcher first to gather information
2. Send to analyst to interpret and analyse the findings
3. Send to writer to produce the final deliverable
4. Review the writer's output — if it needs more data, loop back to researcher

Rules:
- Only mark FINISH when the user's request is completely addressed
- If the user asks a follow-up, continue from where you left off
- Keep track of what each agent has done to avoid redundant work"""

app = create_supervisor(
    llm,
    agents=[researcher, analyst, writer],
    prompt=SUPERVISOR_PROMPT,
    output_mode="last_message"
).compile(
    checkpointer=SqliteSaver.from_conn_string("research_assistant.db")
)

# ── Usage ──────────────────────────────────────────────────────────────────────
def chat(user_message: str, session_id: str) -> str:
    """Multi-turn research assistant with persistent memory."""
    config = {"configurable": {"thread_id": session_id}}
    
    result = app.invoke(
        {"messages": [("user", user_message)]},
        config
    )
    return result["messages"][-1].content

# Turn 1
response1 = chat("Research the current state of AI reasoning models", "research-session-1")
print(response1)

# Turn 2 — remembers the research from turn 1
response2 = chat("Now compare this to the state of AI reasoning 2 years ago", "research-session-1")
print(response2)

# Turn 3 — asks for a deliverable based on accumulated research
response3 = chat("Write a 500-word executive summary of all our findings", "research-session-1")
print(response3)
```

---

## 2.9 Adding Human-in-the-Loop to Multi-Agent Systems

Multi-agent systems particularly benefit from human oversight — the longer the chain of agents, the more important it is to have checkpoints.

```python
# Pause before researcher runs web searches (human approves the search strategy)
app = create_supervisor(
    llm,
    agents=[researcher, analyst, writer],
    prompt=SUPERVISOR_PROMPT,
).compile(
    checkpointer=MemorySaver(),
    interrupt_before=["researcher"]  # pause before researcher node
)

config = {"configurable": {"thread_id": "hitl-session-1"}}

# Start the run — pauses before researcher
snapshot = app.invoke(
    {"messages": [("user", "Research quantum computing market size")]},
    config
)

# Inspect what the supervisor planned
current_state = app.get_state(config)
print("Supervisor's plan:", current_state.values["messages"][-1].content)

# Option A: Approve — just continue
result = app.invoke(None, config)

# Option B: Redirect — update the state before continuing
app.update_state(
    config,
    {"messages": [("user", "Focus specifically on enterprise quantum computing adoption")]}
)
result = app.invoke(None, config)

# Option C: Add a critic agent that reviews each agent's output
critic = create_react_agent(
    llm,
    tools=[],
    name="critic",
    prompt=(
        "You are a quality reviewer. Check the previous agent's output for: "
        "accuracy, completeness, and relevance to the original request. "
        "Return APPROVED or NEEDS_REVISION with specific feedback."
    )
)

# Build a pipeline with a critic after the writer
app_with_critic = create_supervisor(
    llm,
    agents=[researcher, analyst, writer, critic],
    prompt=SUPERVISOR_PROMPT + "\n\nAlways send writer output to critic for review before finishing.",
).compile(checkpointer=MemorySaver())
```

---

## 2.10 Debugging Multi-Agent Systems

Multi-agent systems are harder to debug than single agents. Use these tools systematically.

### LangSmith Tracing (Recommended First Step)

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "ls-your-key-here"
os.environ["LANGCHAIN_PROJECT"] = "multi-agent-research"

# Every subsequent run is automatically traced in LangSmith
# Go to smith.langchain.com to see full execution trees
```

### Streaming to See Execution in Real Time

```python
for chunk in app.stream(inputs, config, stream_mode="updates"):
    for node_name, state_update in chunk.items():
        print(f"\n{'='*50}")
        print(f"NODE: {node_name}")
        print(f"UPDATED FIELDS: {list(state_update.keys())}")
        if "messages" in state_update:
            last_msg = state_update["messages"][-1]
            print(f"LAST MESSAGE: {getattr(last_msg, 'content', str(last_msg))[:200]}")
```

### State Inspection and Time-Travel

```python
# Get current state
state = app.get_state(config)
print("Current state values:", state.values)
print("Next node to run:", state.next)
print("Graph configuration:", state.config)

# List all checkpoint history
history = list(app.get_state_history(config))
print(f"Total checkpoints: {len(history)}")

# Each entry in history is a StateSnapshot
for i, snapshot in enumerate(history[:5]):
    print(f"Step {i}: next={snapshot.next}, "
          f"messages={len(snapshot.values.get('messages', []))}")

# Time-travel: replay from step 3
step_3_config = history[-3].config
result = app.invoke(None, step_3_config)

# Edit state at a past checkpoint and replay from there
app.update_state(
    history[-3].config,
    {"messages": [("user", "Actually, focus on European markets only")]}
)
result = app.invoke(None, history[-3].config)
```

### Recursion Limit — Preventing Infinite Loops

```python
# Default recursion_limit is 25 — set it explicitly
result = app.invoke(
    inputs,
    config,
    recursion_limit=50   # max total node executions before raising GraphRecursionError
)

# Per-call override
result = app.invoke(inputs, {**config, "recursion_limit": 10})
```

---

## 2.11 Production Checklist

Before deploying a multi-agent system, verify each of these:

### Architecture

- [ ] Single-agent version was built and tested first
- [ ] Split along genuine pain points (too many tools / parallelism / specialisation), not arbitrary divisions
- [ ] Each agent has a clear, non-overlapping responsibility
- [ ] Each agent has been tested independently before composition
- [ ] State schema is documented and stable

### Reliability

- [ ] `recursion_limit` set explicitly (not relying on default 25)
- [ ] Error handling in every tool node (`try/except` with useful error messages)
- [ ] `raise_for_status()` and timeout on all HTTP calls inside tools
- [ ] Fallback behaviour when an agent fails (graceful degradation)
- [ ] Tested with malformed inputs and edge cases

### Memory and State

- [ ] Production checkpointer configured (not `MemorySaver`)
- [ ] `thread_id` strategy documented (one per user? per session? per task?)
- [ ] Context window growth managed (`trim_messages` or summarisation)
- [ ] Long-term memory (Store API) if cross-session knowledge is needed

### Observability

- [ ] LangSmith tracing enabled from day one
- [ ] Streaming enabled for user-facing responses
- [ ] Logging on all agent transitions and handoffs
- [ ] Alerts on `GraphRecursionError` and tool failures

### Security

- [ ] Tool permissions reviewed — agents only have the tools they need
- [ ] Sandboxed code execution (never `exec()` in production without sandboxing)
- [ ] Human-in-the-loop for irreversible actions (send email, delete data, make purchases)
- [ ] Input validation on all tool arguments (Pydantic schemas)

---

## 2.12 Quick Reference: Choosing the Right Architecture

```
Your task is clearly a sequence (A then B then C)?
  └─ Yes → Supervisor (sequential)
  └─ No
      └─ Subtasks are independent and can run at the same time?
            └─ Yes → Parallel / Fan-out
            └─ No
                └─ Agents need to collaborate fluidly without a fixed order?
                      └─ Yes → Swarm / Handoff
                      └─ No
                            └─ You want maximum reusability and team ownership?
                                  └─ Yes → Subgraphs
                                  └─ No → Supervisor with flexible delegation
```

### Architecture Comparison at a Glance

| | Supervisor | Swarm | Subgraphs | Parallel |
|---|---|---|---|---|
| **Control** | Centralised | Decentralised | Structural | Structural |
| **Routing** | Supervisor LLM | Each agent | Parent graph | Graph edges |
| **Traceability** | ⭐⭐⭐ Easy | ⭐ Hard | ⭐⭐ Medium | ⭐⭐⭐ Easy |
| **Flexibility** | ⭐⭐ Medium | ⭐⭐⭐ High | ⭐⭐ Medium | ⭐ Low |
| **Bottleneck** | Supervisor | None | None | None |
| **Best for** | Pipelines | Open tasks | Reusability | Parallel work |
| **Library** | `langgraph-supervisor` | `langgraph-swarm` | Core LangGraph | Core LangGraph |

---

## 2.13 Common Mistakes and How to Fix Them

**Mistake 1: State grows unboundedly**
```python
# ❌ No trimming — context window fills up after ~20 turns
def agent(state):
    return {"messages": [llm.invoke(state["messages"])]}

# ✅ Trim before the LLM call
def agent(state):
    trimmed = trim_messages(state["messages"], max_tokens=4000, strategy="last")
    return {"messages": [llm.invoke(trimmed)]}
```

**Mistake 2: Agents with too many tools**
```python
# ❌ One agent, 12 tools — LLM gets confused about which to pick
agent = create_react_agent(llm, tools=[t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12])

# ✅ Three specialist agents, 4 tools each
research_agent = create_react_agent(llm, tools=[web_search, arxiv, wiki, news])
code_agent     = create_react_agent(llm, tools=[run_python, read_file, write_file, exec_shell])
data_agent     = create_react_agent(llm, tools=[query_db, fetch_api, parse_csv, generate_chart])
```

**Mistake 3: No error handling in tools**
```python
# ❌ Exceptions propagate up and crash the graph
@tool
def fetch_data(url: str) -> str:
    response = requests.get(url)
    return response.json()

# ✅ Catch errors, return useful messages so the LLM can recover
@tool
def fetch_data(url: str) -> str:
    """Fetch data from a URL. Returns the response body or an error message."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.Timeout:
        return "Error: Request timed out after 10 seconds. Try a different URL."
    except requests.HTTPError as e:
        return f"Error: HTTP {e.response.status_code} from {url}."
    except Exception as e:
        return f"Error: {type(e).__name__}: {str(e)}"
```

**Mistake 4: Forgetting recursion_limit for multi-agent loops**
```python
# ❌ Supervisor + workers can easily exceed 25 steps on complex tasks
result = app.invoke(inputs, config)  # raises GraphRecursionError on step 26

# ✅ Set an appropriate limit based on expected task complexity
result = app.invoke(inputs, config, recursion_limit=100)
# Or in config:
config = {
    "configurable": {"thread_id": "session-1"},
    "recursion_limit": 100
}
```

**Mistake 5: Hardcoding node names in routing**
```python
# ❌ If you rename a node, routing breaks silently
graph.add_conditional_edges("supervisor", router, {"researcher": "researcher"})

# ✅ Use constants
RESEARCHER = "researcher"
ANALYST    = "analyst"
WRITER     = "writer"

graph.add_node(RESEARCHER, researcher_node)
graph.add_conditional_edges("supervisor", router, {
    RESEARCHER: RESEARCHER,
    ANALYST:    ANALYST,
    WRITER:     WRITER,
    "FINISH":   END
})
```

---

## Summary: The Complete Mental Model

```
LangGraph Single Agent
═══════════════════════
State (TypedDict)  ←──── shared memory for the whole run
    ↕
Nodes (functions)  ←──── do the work (LLM calls, tools, logic)
    ↕
Edges (connections) ←─── decide what runs next (normal or conditional)
    ↕
Checkpointer       ←──── save state after every node (enables memory + HITL)


LangGraph Multi-Agent
══════════════════════
Each agent is itself a compiled LangGraph ↗
                                          
Agents collaborate via one of:
  ├── Supervisor   → central LLM orchestrator routes between agents
  ├── Swarm        → agents hand off to each other peer-to-peer
  ├── Subgraphs    → agents used as nodes inside parent graphs
  └── Parallel     → multiple agents run simultaneously (fan-out/fan-in)

Communication happens through:
  ├── Shared State  → all agents read/write the same TypedDict
  ├── Messages      → agents append to the messages list
  ├── Handoff Tools → agents call tools that transfer control
  └── Command       → nodes return explicit routing + state update together
```

---

*End of notes. For hands-on practice: build a single ReAct agent → add checkpointing → add HITL → refactor into a 2-agent supervisor system → add parallelism. Each step is a complete, runnable system.*
