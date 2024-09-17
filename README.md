# LLM-Based Service Intention Recognition and Product Retrieval

This project demonstrates the integration of Azure OpenAI models for service intention recognition and document embedding for product retrieval. It consists of two main modules:

1. **main.py**: Implements a service workflow that uses an Azure GPT model to understand user service queries and respond based on predefined intentions.
2. **rag_pipeline.py**: Uses document embeddings and vector search to retrieve relevant product information from CSV files.

## Features

- **Intention Recognition**: Identifies user service needs from input queries using the GPT-4 model deployed on Azure.
- **Document Embedding**: Embeds documents from CSV files and allows for semantic search based on user queries.
- **Workflow Automation**: The service process is organized into a directed graph to handle multiple nodes of operations and output final results.

## Prerequisites

- Python 3.8+
- Azure OpenAI API credentials (API Key and Endpoint)
- Installed dependencies: 
  - `langgraph`
  - `langchain_community`
  - `langchain_openai`
  - `dotenv`
  - `pandas`
  - `chroma`

## Setup

1. Clone the repository:

    ```bash
    git clone https://github.com/Alperemrehas/LangGraph.git
    cd your-repo-name
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Set up environment variables for Azure OpenAI API:

    - Create a `.env` file in the root directory with the following content:

    ```bash
    AZURE_OPENAI_API_KEY=your-api-key
    AZURE_OPENAI_ENDPOINT=your-endpoint
    ```

## Usage

### Service Intention Recognition

Run the `main.py` file to identify user service intentions:

```bash
python main.py
