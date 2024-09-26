from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain.schema import Document
import os
import pandas as pd
from io import StringIO

class EmbeddingsHandler:
    def __init__(self):
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_deployment = "text-embedding-ada-002"    
        temperature = 0.7
        azure_model = "text-embedding-ada-002"

        self.embeddings = AzureOpenAIEmbeddings(
            model= azure_model,
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            api_version="2024-02-01",
            api_key=api_key,
        )
    def load_csv_files(self,directory, glob_pattern):
        loader = DirectoryLoader(directory, glob=glob_pattern)
        csv_files = loader.load()
        documents = []
        for file in csv_files:
            # Assuming the content of the Document object is the CSV data
            csv_content = file.page_content
            df = pd.read_csv(StringIO(csv_content))  # Read CSV data from the content
            documents.extend([Document(page_content=str(row)) for row in df.itertuples(index=False)])
        return documents

    def function_3(self,input_3):
        print('-> Calling Embeddings Handler')
        if input_3 == None:
            pass
        else:
            # Load CSV files and convert them to documents
            documents = self.load_csv_files('products', '*.csv')
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=100,
                chunk_overlap=10,
                length_function=len
            )
        new_documents = text_splitter.split_documents(documents=documents)
        docs_strings = [doc.page_content for doc in new_documents]

        db = Chroma.from_documents(new_documents, self.embeddings)
        retriever = db.as_retriever(search_kwargs={"k": 5})

        docs = retriever.get_relevant_documents(input_3)
            
        #response = f"Your task is to understand user intention based on the user query. Only output the intention among ['H1: Reactive Service - Replacement Defective Parts', 'H2: Preventative Service - HVAC Maintenance | Fuel Filter or Printer Paper Replacement', 'H3: Proactive Service - Elevator | Dispenser Troubleshooting, time to replace a nozzle, hose, breakaway or filter']"
        return docs
       


    # Load CSV files and convert them to documents
    '''documents = load_csv_files('products', '*.csv')

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=10,
        length_function=len
    )

    #embeddings = EmbeddingsHandler()

    new_documents = text_splitter.split_documents(documents=documents)
    docs_strings = [doc.page_content for doc in new_documents]

    db = Chroma.from_documents(new_documents, embeddings.embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 5})

    query = "I am looking for a product that can help me with my fuel filter"
    docs = retriever.get_relevant_documents(query)
    print(docs)'''