from langgraph.graph import StateGraph, END
from src.rag_pipeline import EmbeddingsHandler
from src.agent import AgentState, LLMHandler

def main():
    handler = LLMHandler()
    embeddings_handler = EmbeddingsHandler()

    graph = StateGraph(AgentState)

    graph.add_node("agent", handler.analyze_user_intention)
    graph.add_node("LLM", handler.database_checker)
    graph.add_node("DOC", handler.document_checker)
    graph.add_node("RAG", handler.format_user_intention)
    
    

    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        handler.router,
        {
            "RAG Call": "RAG",
            "LLM Call": "LLM",
            "DOC Call": "DOC"
        },
    )

    graph.add_edge("RAG", END)
    graph.add_edge("LLM", END)
    graph.add_edge("DOC", END)


    app = graph.compile()

    # query = 'Hi, my fuel filter is not working. Can you help me with that?'
    # print(f"Input query: {query}")
    #Greeting Message to Customer
    print("Welcome to our Website. I am your Customer Support Agent John. How can I help you today?")

    user_query = input("Please enter your query: ")

    inputs = {"messages": [user_query]}
    # H1 Question
    '''inputs = {
        "messages": [
            "Hi, one of my part is sseems to be defective. Can you help me with that?"
        ]
    }
    # H2 Question
    # inputs = {"messages": ["Hi, my fuel filter is not working. Can you help me with that?"]}
    # H3 Question
    # inputs = {"messages": ["Hi,I wonder that what is the time for replacment for my nozzle? Can you help me with that?"]}'''
    # Debugging: Print the graph structure
    print("Graph Nodes:", graph.nodes)
    print("Graph Edges:", graph.edges)

    # Debugging: Check the output of the router function
    #router_output = handler.router(inputs)
    #print(f"Router output: {router_output}")
    out = app.invoke(inputs)
    print(out)

if __name__ == "__main__":
    main()
