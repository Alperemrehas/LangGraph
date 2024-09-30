import os
import json
import ast
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field, ValidationError
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

class IntentionSelector(BaseModel):
    user_intention: str = Field(
        description="user_intention",
        example="H1",
        title="Intention",
        type="string"
    )

    @classmethod
    def validate_intention(cls, user_intention: str):
        valid_intentions = ["H1", "H2", "H3"]
        if user_intention not in valid_intentions:
            raise ValueError(f"Invalid intention: {user_intention}. Must be one of {valid_intentions}")
        return user_intention

class LLMHandler:
    def __init__(self, temperature=0, azure_deployment="gpt-4o-mini"):
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
        messages = state['messages']
        question = messages[-1]

        prompt_template = """Your task is to understand user intention based on the user query.
        Provide a well-structured JSON output containing the user intention among:
        [H1, H2, H3]. Don't include reasoning. H1 refers Reactive Service - Replacement Defective Parts,
        H2 refers Preventative Service, HVAC Maintenance,Fuel Filter or Printer Paper Replacement,
        H3 refers Proactive Service, Elevator, Dispenser Troubleshooting,time to replace a nozzle, hose, breakaway or filter.
        Following is the user query. \n{format_instructions}\n{question}"""

        prompt = PromptTemplate.from_template(
            template=prompt_template,
            input_variable=[question],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        complete_query = prompt.format(question=question)

        try:
            chain = prompt | self.llm | parser
            response = chain.invoke({"question": question, "format_instructions": parser.get_format_instructions()})

            if isinstance(response, BaseMessage):
                response_text = response.content
            else:
                response_text = str(response)

            return {"messages": [response.user_intention]}
        except Exception as e:
            print(f"An error occurred: {e}")
            raise

    def format_user_intention(self, intention):
        print(f"User intention: {intention} Calling --> RAG Call")
        complete_query = "Answer this question like you are checking a part order. \n{question}"
        response = self.llm.invoke(complete_query)
        return {"messages": [response.content]}

    def database_checker(self, intention):
        print(f"User intention: {intention} Calling --> LLM Call")
        complete_query = "Answer this question like you are checking a database for a customer need. \n{question}"
        response = self.llm.invoke(complete_query)
        return {"messages": [response.content]}
    
    def document_checker(self, intention):
        print(f"User intention: {intention} Calling --> Document Call")
        complete_query = "Answer this question like you are checking a document for a customer need. \n{question}"
        response = self.llm.invoke(complete_query)
        return {"messages": [response.content]}

    def router(self, state):
        messages = state["messages"]
        last_message = messages[-1]

        if 'H1' in last_message:
            print('H1')
            return 'RAG Call'
        elif 'H2' in last_message:
            print('H2')
            return 'LLM Call'
        elif 'H3' in last_message:
            print('H3')
            return 'DOC Call'
        else:
            return 'end'

parser = PydanticOutputParser(pydantic_object = IntentionSelector)