#Declare Variables
from typing import TypedDict,Annotated
#Setup Environment
from dotenv import load_dotenv
load_dotenv()
#LangGraph Imports
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode,tools_condition
# from langgraph.checkpoint.memory import MemorySaver
# from langgraph.types import interrupt,Command
#LangChain Imports
from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage,BaseMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings

llm = init_chat_model("google_genai:gemini-3.1-flash-lite")

loader = PyPDFLoader(
    "/home/jaswanthkumarkamireddi/Desktop/Python_Works/Devlabs/DevLabs-2.0-AI-ML/Week-6/Jaswanth_Kumar/RAG_Using_Langgraph/cos.pdf"
)
docs = loader.load()
len(docs)

splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=400)
chunks = splitter.split_documents(docs)

print(len(chunks))


# embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vector_store = FAISS.from_documents(chunks,embeddings)

retriever = vector_store.as_retriever(search_type='similarity',search_kwargs={'k':4})

@tool
def rag_tool(query:str):
    """
    Retrieve Relevant Information from the pdf document.
    Use this Tool when the user asks factual / conceptual questions
    that might be answered from the stored documents
    """

    result = retriever.invoke(query)

    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        'query':query,
        'context':context,
        'metadata':metadata
    }

tools = [rag_tool]

llm_with_tools = llm.bind_tools(tools)

class ChatState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]

def chat_node(state:ChatState):
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    return {'messages':[response]}

tool_node = ToolNode(tools)

graph = StateGraph(ChatState)

graph.add_node('chat_node',chat_node)
graph.add_node('tools',tool_node)

graph.add_edge(START,'chat_node')
graph.add_conditional_edges('chat_node',tools_condition)
graph.add_edge('tools','chat_node')

chatbot = graph.compile()

result = chatbot.invoke(
    {
        "messages":[
            HumanMessage(
                content=(
                    "Tell me about the DSAI Course Subjects"
                )
            )
        ]
    }
)

print(result['messages'][-1].text)



