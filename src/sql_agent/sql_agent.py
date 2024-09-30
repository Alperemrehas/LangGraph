import sys
import os


sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from langchain_openai import AzureChatOpenAI
from langchain_community.utilities import SQLDatabase


from src.settings.base_settings import ClassSettings



class SQLAgent:


    def __init__(self, temperature=0, azure_deployment="gpt-40-mini"):
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

        if not self.api_key or not self.azure_endpoint:
            raise ValueError(
                "API Key and Azure Endpoint must be set in environment variables."
            )

        self.llm = AzureChatOpenAI(
            azure_endpoint=self.azure_endpoint,
            azure_deployment=azure_deployment,
            api_version="2024-02-01",
            api_key=self.api_key,
            temperature=temperature,
        )

basesettings = ClassSettings()
database_url = basesettings.set_database_url()
db = SQLDatabase.from_uri(database_url)
print(db.dialect)
print(db.get_usable_table_names())
test_query = db.run("SELECT * FROM public.sales LIMIT 5")
print(test_query)