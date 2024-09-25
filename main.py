import re
import os 
import json

from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage
from langgraph.graph import Graph, StateGraph, END
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import AzureChatOpenAI
from rag_pipeline import EmbeddingsHandler
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate



load_dotenv()
# assign AgentState as an empty dict
#AgentState = {}

# messages key will be assigned as an empty array. We will append new messages as we pass along nodes. 
#AgentState["messages"] = []
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

from pydantic import BaseModel, Field, ValidationError

class IntentionSelector(BaseModel):
    user_intention: str = Field(
        description="User Intention",
        example="H1 Reactive Service",
        title="Intention",
        type="string"
    )

    @classmethod
    def validate_intention(cls, user_intention: str):
        valid_intentions = [
            "H1 Reactive Service - Replacement Defective Parts",
            "H2 Preventative Service - HVAC Maintenance - Fuel Filter",
            "H3 Proactive Service - Elevator - Dispenser Troubleshooting, time to replace a nozzle, hose, breakaway or filter"
        ]
        print("-> Validating Intention")
        print(f"Intention: {user_intention}")
        print(f"Valid Intentions: {valid_intentions}")
        if user_intention not in valid_intentions:
            raise ValueError(f"Invalid intention: {user_intention}. Must be one of {valid_intentions}")
        return user_intention

# Example usage
'''try:
    intention = IntentionSelector(Intention="H1 Reactive Service")
    print(intention)
except ValidationError as e:
    print(e)'''

class LLMHandler:
    def __init__(self, temperature=0, azure_deployment="gpt-40"):
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

        # Get and print format instructions
        print("**********************************************")
        format_instructions = parser.get_format_instructions()
        #print(f"Format Instructions: {format_instructions}")
        prompt_template = f"""Your task is to understand user intention based on the user query.
        Provide a well-structured JSON output containing the user intention among:
        - H1 Reactive Service - Replacement Defective Parts,
        - H2 Preventative Service - HVAC Maintenance - Fuel Filter or Printer Paper Replacement,
        - H3 Proactive Service - Elevator - Dispenser Troubleshooting, time to replace a nozzle, hose, breakaway or filter.
        The user query is: {question}""" 
        prompt = ChatPromptTemplate.from_template(
            template = prompt_template,
            input_variable = [question],
        )
        complete_query = prompt.format(question=question)
        print(f"Complete Query: {complete_query}")
        try:
        # Create the chain
            chain = prompt | self.llm | parser

            # Invoke the chain
            response = chain.invoke({"question": question, "format_instructions": format_instructions})

            # Extracting content from the response
            # If the response is an object like AIMessage, extract the 'content' attribute.
            if isinstance(response, BaseMessage):
                response_text = response.content
            else:
                response_text = str(response)  # Convert response to string if it’s not the expected format.

            # Print the response for debugging
            print(f"Response Text: {response_text}")
            
            #clean_response = response_text.strip().replace('```json', '').replace('```', '').strip()

            #clean_response = clean_response.replace("'", '"')

            #print(f"Clean Response Text: {clean_response}")

            parsed_response = json.loads(response_text)

            #IntentionSelector.validate_intention(parsed_response["user_intention"])
            #return {"Intention": parsed_response["user_intention"]}  # Return the intention directly as specified
            return {"messages": [parsed_response["user_intention"]]}


            # Parse the response text using the parser
            #parsed_response = parser.parse(response_text)

            # Validate the parsed response
            #IntentionSelector.validate_intention(parsed_response.Intention)

            #return {"messages": [parsed_response.Intention]}
        except Exception as e:
            print(f"An error occurred: {e}")
            raise
        '''
        template = f"""Your task is to understand user intention based on the user query.
        Provide a well-structured JSON output containing the user intention among:
        - H1 Reactive Service - Replacement Defective Parts,
        - H2 Preventative Service - HVAC Maintenance - Fuel Filter or Printer Paper Replacement,
        - H3 Proactive Service - Elevator - Dispenser Troubleshooting, time to replace a nozzle, hose, breakaway or filter.
        The user query is: {question} {format_instructions}""" 
        prompt = ChatPromptTemplate(
            template = template,
            messages=messages,
            input_variable = [question],
            partial_query = {"format_instructions": parser.get_format_instructions()},
            pattern=re.compile(r"\`\`\`\n\`\`\`") 
            )
        #response = self.llm.invoke(complete_query)
        
        # Generate the complete query
        #complete_query = prompt.make_query()

        # Print the complete query for debugging
        print(f"Complete Query: {prompt}")

        try:
            # Create the chain
            chain = prompt | self.llm | parser

            # Invoke the chain
            response = chain.invoke({"question": question, "format_instructions": format_instructions})
            
            # Print the response for debugging
            print(f"Response: {response}")

            # Validate the parsed response
            IntentionSelector.validate_intention(response.Intention)
            
            return {"messages": [response.Intention]}
        except Exception as e:
            print(f"An error occurred: {e}")
            raise
    '''

    def format_user_intention(self, intention):
        print('-> Formatting User Intention')
        USER_INTENTION = intention.upper()
        return f"User Intention is {USER_INTENTION}"
    
    def database_checker(self, intention):
        print('-> SQL Query')
        return f"SELECT * FROM TABLE WHERE INTENTION = {intention}"
    
    def router(state):
        print('-> Router ->')
        
        messages = state["messages"]
        last_message = messages[-1]
        print("last message:",last_message)

        if 'Reactive' in last_message:
            return 'RAG Call'
        elif 'Preventative' in last_message:
            return 'LLM Call'
        #elif 'Proactive' in last_message:
        else:
            return 'end'

parser = PydanticOutputParser(pydantic_object = IntentionSelector)

def main():
    handler = LLMHandler()
    embeddings_handler = EmbeddingsHandler()



    graph = StateGraph(AgentState)

    graph.add_node("agent", handler.analyze_user_intention)
    graph.add_node("RAG", handler.format_user_intention)
    graph.add_node("LLM", handler.database_checker)
    #graph.add_node("LLM", embeddings_handler.function_3)
    
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent",handler.router,{
        "RAG Call": "RAG",
        "LLM Call": "LLM",
    })

    graph.add_edge("RAG", END)
    graph.add_edge("LLM", END)

    app = graph.compile()

    #query = 'Hi, my fuel filter is not working. Can you help me with that?'
    #print(f"Input query: {query}")
    inputs = {"messages": ["Hi, my fuel filter is not working. Can you help me with that?"]}
    out = app.invoke(inputs)
    
    '''workflow = Graph()

    workflow.add_node("analyze_intention", handler.analyze_user_intention)
    workflow.add_node("format_intention", handler.format_user_intention)
    workflow.add_node("embeddings", embeddings_handler.function_3)

    workflow.add_edge('analyze_intention', 'format_intention')
    workflow.add_edge('format_intention', 'embeddings')

    workflow.set_entry_point("analyze_intention")
    workflow.set_finish_point("embeddings")

    app = workflow.compile()

    try:
        app.invoke(query)

        for output in app.stream(query):
            for key, value in output.items():
                print(f"Output from node '{key}':")
                print("----")
                print(value)
            print("\n----\n")

    except Exception as e:
        print(f"An error occurred: {e}")'''
    

if __name__ == "__main__":
    main()
