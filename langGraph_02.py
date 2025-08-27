# Simple Sequential LLM Chain Using LangGraph

from langgraph.graph import StateGraph,START,END
from langchain.prompts import ChatPromptTemplate
from typing import TypedDict
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# Making State
class qaState(TypedDict):
    question : str
    answer : str

# Initialize the LLM with OpenRouter and OpenAI GPT-3.5 Turbo
llm = ChatOpenAI(
    model_name="openai/gpt-3.5-turbo",
    base_url="https://openrouter.ai/api/v1",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0.7,
    max_tokens=500,
)

# Node function: fetches answer from LLM given a question
def fetch_answer(state: qaState) -> qaState:
    # fetching question from state
    query = state['question']
    # Create prompt for the LLM
    prompt = ChatPromptTemplate.from_template(f"Generate a answer in 4 line for the given query: {query}")
    # Chain prompt to LLM
    chain = prompt | llm
    # Invoke the chain and get response
    response = chain.invoke({"query": query})
    state['answer'] = response.content
    return state

# Create a state graph for the QA workflow
graph = StateGraph(qaState)

# Add fetch_answer node to the graph
graph.add_node("fetch_answer", fetch_answer)

# Define graph edges: start -> fetch_answer -> end
graph.add_edge(START, "fetch_answer")
graph.add_edge("fetch_answer", END)

# Compile the workflow
workflow = graph.compile()

# Get user query and run the workflow
query = input("Enter Query: ")
result = workflow.invoke({"question": query})

# Print the result
print(result)