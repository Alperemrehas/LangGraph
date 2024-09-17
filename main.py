from langgraph.graph import Graph
import os 
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import AzureChatOpenAI

load_dotenv()


def function_1(input_1):
  return input_1 + " First Function"

def function_2(input_2):
  return input_2 + " to Second Function"

def main():
    workflow = Graph()

    workflow.add_node("node_1", function_1)
    workflow.add_node("node_2", function_2)

    workflow.add_edge('node_1', 'node_2')

    workflow.set_entry_point("node_1")
    workflow.set_finish_point("node_2")

    app=workflow.compile()

    input = ' I am moving from'

    for output in app.stream(input):
        for key, value in output.items():
            print(f"Output from node '{key}':")
            print("----")
            print(value)
        print("\n----\n")

    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_deployment = "gpt-40"
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    temperature = 0.7
    
    llm = AzureChatOpenAI(
                azure_endpoint=azure_endpoint,
                azure_deployment=azure_deployment,
                api_version="2024-02-01",
                api_key=api_key,
                temperature=temperature,               
                
            )

    print(llm.invoke("Hello, how are you?").content)

if __name__ == "__main__":
    main()