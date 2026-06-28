# 📚 Week 5 Resources — Multi-Agent Systems with LangGraph 🤖🤝🤖

Hey Builders! 👋

This week we go beyond single agents — we're building **crews** of AI agents that collaborate, hand off work, and reason together. Let's dive in! 🚀

---

## 🎯 What We're Covering

- Multi-agent architectures (Sequential, Supervisor, Hierarchical)
- Agent-to-agent handoffs in LangGraph
- Shared state across agents
- Orchestration patterns — who's in charge?

---

## 🧠 Core Reading

📖 **LangGraph Multi-Agent Docs**
https://langchain-ai.github.io/langgraph/concepts/multi_agent/
→ The official architecture guide. Read this first!

📖 **LangGraph How-To: Multi-Agent Networks**
https://langchain-ai.github.io/langgraph/how-tos/multi-agent-network/
→ Practical implementation with code

📖 **Agent Supervisor Pattern**
https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/
→ Exactly the pattern we're using this week ✅

---

## 🎥 Video Tutorials

▶️ **LangGraph Multi-Agent Tutorial — LangChain (Official)**
https://www.youtube.com/watch?v=hvAPnpSfSGo
→ Supervisor + worker agents explained visually

▶️ **Build a Multi-Agent System from Scratch**
https://www.youtube.com/watch?v=_0_UOgPLSuI
→ Step-by-step walkthrough

---

## 🛠️ Reference Code

💻 **LangGraph Examples — Multi-Agent Repo**
https://github.com/langchain-ai/langgraph/tree/main/examples/multi_agent
→ Production-ready patterns to study

💻 **LangGraph Agent Handoffs**
https://langchain-ai.github.io/langgraph/how-tos/agent-handoffs/
→ Key skill: passing control between agents cleanly

---

## 🏗️ This Week's Task

### Build a 3-Agent Crew!

| Agent | Role |
|-------|------|
| 🔬 **Researcher** | Finds and summarizes info on a topic |
| ✍️ **Writer** | Turns research into a structured draft |
| 🔍 **Reviewer** | Critiques and improves the draft |

Pick **any real topic** you care about. Watch your agents collaborate like a real team!

> 💡 **Tip:** Use `StateGraph` + `Command` for handoffs. Each agent is just a node. Start simple, then add a Supervisor node if you want a bonus challenge!

---

## ⭐ Bonus Challenge

Add a **Supervisor** agent that decides which agent to call next dynamically. This is the real power of LangGraph!

---

*See you at the session! 🙌*
