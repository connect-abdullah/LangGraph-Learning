# Prompt Chaining LLM Chain Using LangGraph

from langgraph.graph import StateGraph,START,END
from langchain.prompts import ChatPromptTemplate
from typing import TypedDict
from IPython.display import Image
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# Making State
class BlogState(TypedDict):
    question : str
    outline : str
    blog : str

# Initialize the LLM with OpenRouter and OpenAI GPT-3.5 Turbo
llm = ChatOpenAI(
    model_name="openai/gpt-3.5-turbo",
    base_url="https://openrouter.ai/api/v1",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0.7,
    max_tokens=500,
)

# Node function: fetches answer from LLM given a question
def gen_outline(state: BlogState) -> BlogState:
    # fetching question from state
    query = state['question']
    # Create prompt for the LLM
    prompt = ChatPromptTemplate.from_template(f"Generate a outline for making blog of this question: {query}. It should be concise and readable for other llm to generate a good blog.")
    # Chain prompt to LLM
    chain = prompt | llm
    # Invoke the chain and get response
    response = chain.invoke({"query": query})
    state['outline'] = response.content
    return state

def gen_blog(state: BlogState) -> BlogState:
    # fetching question from state
    outline = state['outline']
    question = state["question"]
    # Create prompt for the LLM
    prompt = ChatPromptTemplate.from_template(f"Write a blog on the title :{question},\n Using the following Outline:{outline}.")
    # Chain prompt to LLM
    chain = prompt | llm
    # Invoke the chain and get response
    response = chain.invoke({"question": question, "outline" : outline})
    state['blog'] = response.content
    return state


# Create a state graph for the QA workflow
graph = StateGraph(BlogState)

# Add fetch_answer node to the graph
graph.add_node("gen_outline", gen_outline)
graph.add_node("gen_blog", gen_blog)

# Define graph edges: start -> fetch_answer -> end
graph.add_edge(START, "gen_outline")
graph.add_edge("gen_outline","gen_blog")
graph.add_edge("gen_blog", END)

# Compile the workflow
workflow = graph.compile()

# Get user query and run the workflow
query = input("Enter Query: ")
result = workflow.invoke({"question": query})

# Print the result
print(result)
Image(workflow.get_graph().draw_mermaid_png())