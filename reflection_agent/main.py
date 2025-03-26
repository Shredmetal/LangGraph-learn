from typing import List, Sequence
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, MessageGraph
from dotenv import load_dotenv
from chains import generate_chain, reflect_chain
load_dotenv()


REFLECT = "reflect"
GENERATE = "generate"


def generation_node(state: Sequence[BaseMessage]):
    return generate_chain.invoke({"messages": state})


def reflect_node(state: Sequence[BaseMessage]):
    res = reflect_chain.invoke({"messages": state})
    return [HumanMessage(content=res.content)]


builder = MessageGraph()
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflect_node)
builder.set_entry_point(GENERATE)


def should_continue(state: List[BaseMessage]):
    if len(state) > 6:
        return END
    else:
        return REFLECT


builder.add_conditional_edges(GENERATE, should_continue)
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()
print(graph.get_graph().draw_mermaid())


if __name__ == "__main__":
    print("Hello Langgraph")

    inputs = HumanMessage(content=
                          """
                          Make this tweet better:
                          "@LangChainAI - new Tool Calling feature is seriously underrated.
                          
                          After a long wait, it's here - making the implementation of agents across different models 
                          with function calling - super easy.
                          
                          Made a video covering their newest blog post"
                          """
                          )

    response = graph.invoke(inputs)
