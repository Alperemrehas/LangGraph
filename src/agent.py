from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage
from langgraph.graph import Graph, StateGraph, END
import os 
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import AzureChatOpenAI
from rag_pipeline import EmbeddingsHandler
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

class IntentionSelector(BaseModel):
    Intention: str = Field(description="User Intention", example="H1: Reactive Service - Replacement Defective Parts")

class LLMHandler:
    def __init__(self, temperature=0.7, azure_deployment="gpt-40"):
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

        if not self.api_key or not self.azure_endpoint:
            raise ValueError("API Key and Azure Endpoint must be set in environment variables.")

        self.llm = AzureChatOpenAI(
            azure_endpoint=self.azure_endpoint,
            azure_deployment=azure_deployment,
            api_version="2024-02-01",
            api_key=self.api_key,
            temperature=temperature,
        )

    def analyze_user_intention(self, state: AgentState):
        print('-> Calling Intent Classifier')
        messages = state['messages']
        question = messages[-1]

        template = """Your task is to understand user intention based on the user query.
        Only output the intention among ['H1: Reactive Service - Replacement Defective Parts',
        'H2: Proactive Service - Maintenance', 'H3: Inquiry - General Information']."""
        # Add more logic here as needed

class Agent:
    def __init__(self, graph):
        self.app = graph.compile()

    def invoke(self, inputs):
        return self.app.invoke(inputs)

    def run_workflow(self, query):
        graph = Graph()

        graph.add_node("agent", LLMHandler.analyze_user_intention)
        graph.add_node("RAG", LLMHandler.format_user_intention)
        graph.add_node("LLM", EmbeddingsHandler.function_3)
        
        graph.set_entry_point("agent")
        graph.add_conditional_edges("agent", LLMHandler.router, {
            "RAG Call": "RAG",
            "LLM Call": "LLM",
        })

        graph.add_edge("RAG", END)
        graph.add_edge("LLM", END)

        app = graph.compile()

        inputs = {"messages": [query]}
        out = app.invoke(inputs)
        return out