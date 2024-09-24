import os 
import operator

from typing import TypedDict, Sequence, Annotated
import operator
from langgraph.graph import Graph, StateGraph
from langchain_core.messages import BaseMessage


from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from rag_pipeline import EmbeddingsHandler
from pydantic import BaseModel, Field


load_dotenv()

#AgentSate = {}
#AgentState['messages'] = []

class LLMHandler:
    def __init__(self):
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_deployment = "gpt-40"    
        temperature = 0.7

        self.llm = AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            api_version="2024-02-01",
            api_key=api_key,
            temperature=temperature,
        )

    def function_1(self, input_1):
        print('-> Calling Intent Classifier')
        messages = state['messages']
        question = messages[-1]

        complete_query = """Your task is to understand user intention based on the user query.
        Only output the intention among ['H1: Reactive Service - Replacement Defective Parts',
        'H2: Preventative Service - HVAC Maintenance | Fuel Filter or Printer Paper Replacement',
        'H3: Proactive Service - Elevator | Dispenser Troubleshooting, time to replace a nozzle, hose, breakaway or filter']. The user query is: {question}
        {format_instructions}""" 
        prompt = PromptTemplate(template = template,
                                input_variable = [question],
                                partial_query = {"format_instructions": parser.get_format_instructions()} 
                                )
        response = self.llm.invoke(complete_query)
        
        chain = prompt | llm  | parser

        respoonse = chain.invoke({"question": question, format_instructions : parser.get_format_instructions()})
        print(response)

        return {"messages": [response.Intention] }
        #return response.content

    def function_2(self, input_2):
        print('-> Calling Router')
        messages = state['messages']
        question = messages[0]
        tools = ["Reactive Service", "Preventative Service", "Proactive Service"]
        template = """Call the neccesary {tools} based on the user's following intention: {USER_INTENTION}"""

        retrieval_chain = (
            {"tools": tools, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        result = retrieval_chain.invoke(question)
        return {"messages": [result]}
        '''prompt = ChatPromptTemplate.from_template(template)

        #USER_INTENTION = input_2.upper()

        response = f"User Intention is {USER_INTENTION}"
        return response'''

    def router(self):
        print("-> Roueter ->")

        messages = state['messages']
        last_message = messages[-1]

        print(last_message)

        if 'Reactive' in last_message:
            return "Calling RAG"
        elif 'Preventative' in last_message:
            return "Calling Part Request Form"
        elif 'Proactive' in last_message:
            return "Calling SQL Agent"
        else:
            return "Default Answer"

       

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

class IntentionSelector(BaseModel):
    Intention: str = Field(description="User Intention", example="H1: Reactive Service - Replacement Defective Parts")

parser = PydanticOutputParser(pydantic_object = IntentionSelector)

def main():
    handler = LLMHandler()
    embeddings_handler = EmbeddingsHandler()

    query = 'Hi, my fuel filter is not working. Can you help me with that?'
    print(f"Input query: {query}")
    
    workflow = Graph()
    workflow = StateGraph(AgentState)
    workflow.add_node("node_1", handler.function_1)
    workflow.add_node("node_2", handler.function_2)
    workflow.add_node("node_3", embeddings_handler.function_3)

    workflow.add_edge('node_1', 'node_2')
    workflow.add_edge('node_2', 'node_3')

    workflow.set_entry_point("node_1")
    #workflow.set_finish_point("node_2")
    workflow.set_finish_point("node_3")

    app = workflow.compile()


    app.invoke(query)

    for output in app.stream(input):
        for key, value in output.items():
            print(f"Output from node '{key}':")
            print("----")
            print(value)
        print("\n----\n")

if __name__ == "__main__":
    main()