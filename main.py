from langgraph.graph import Graph
import os 
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import AzureChatOpenAI
from rag_pipeline import EmbeddingsHandler

load_dotenv()

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
        complete_query = """Your task is to understand user intention based on the user query.
        Only output the intention among ['H1: Reactive Service - Replacement Defective Parts',
        'H2: Preventative Service - HVAC Maintenance | Fuel Filter or Printer Paper Replacement',
        'H3: Proactive Service - Elevator | Dispenser Troubleshooting, time to replace a nozzle, hose, breakaway or filter']"""
        response = self.llm.invoke(complete_query)
        return response.content

    def function_2(self, input_2):
        USER_INTENTION = input_2.upper()
        response = f"User Intention is {USER_INTENTION}"
        return response


def main():
    handler = LLMHandler()
    embeddings_handler = EmbeddingsHandler()

    query = 'Hi, my fuel filter is not working. Can you help me with that?'
    print(f"Input query: {query}")
    
    workflow = Graph()

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