#Declare Variables
from typing import TypedDict,Annotated
#Setup Environment
from dotenv import load_dotenv
load_dotenv()
#LangGraph Imports
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode,tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt,Command
#LangChain Imports
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
#Request Stocks Price
import requests
#Import venv using os
import os
#Import Date and Time for Stocks
from datetime import date,timedelta


#Memory Initialisation
memory = MemorySaver()

#Date for ALphavantage Get Stock

yesterday = date.today() - timedelta(days=1)
yesterday_str = yesterday.strftime("%Y-%m-%d")

#Building Tools
@tool
def get_stock_price(symbol:str) -> float:
    '''Return the Stock Price Data'''
    response = requests.get(f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&interval=5min&apikey={os.getenv('ALPHAVANTAGE_API_KEY')}")
    data = response.json()
    return data["Time Series (Daily)"][yesterday_str]["4. close"]

@tool
def buy_stock_price(symbol:str,quantity:int,total_price:float) -> str:
    '''Buy Stocks given the stock symbol and quantity'''
    decision = interrupt(f"Approve Buying {quantity} {symbol} stocks for {total_price:.2f}")
    if (decision == "yes"):
        return "Buy Successful"
    else:
        return "Buy Declined"
    
tools = [get_stock_price,buy_stock_price]

#Initialze LLM via Langchain

llm = init_chat_model("google_genai:gemini-3.1-flash-lite")

llm_with_tools = llm.bind_tools(tools)

class State(TypedDict):
    messages:Annotated[list,add_messages]


def chatbot(state:State) -> State:
    return {"messages":[llm_with_tools.invoke(state["messages"])]}

build = StateGraph(State)

# Making Nodes and Edges

build.add_node(chatbot)
build.add_node("tools",ToolNode(tools))

build.add_edge(START,"chatbot")
build.add_conditional_edges("chatbot",tools_condition)
build.add_edge("tools","chatbot")
build.add_edge("chatbot",END)

graph = build.compile(checkpointer=memory)

# Command 1

config = {'configurable':{'thread_id':'1'}}

msg = "Can you get me the stock price of 10 AAPL stocks and Take it as Total and Let me know what the Total "

state = graph.invoke({"messages":[{"role":"user","content":msg}]},config=config)
print(state["messages"][-1].text)

# Command 2

config = {'configurable':{'thread_id':'1'}}

msg = "I want to buy 10  MSFT Stocks too and Add it to the Previous Total"
state = graph.invoke({"messages":[{"role":"user","content":msg}]},config=config)

print(state.get("__interrupt__"))

decision = input("Approve(yes/no): ")

state = graph.invoke(Command(resume=decision),config=config)

print(state["messages"][-1].text)
