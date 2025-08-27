# In this file, i have just a simple non-llm sequential chain just to understand how things work

from langgraph.graph import StateGraph,START,END
from typing import TypedDict
from IPython.display import Image


# Define State
class BMIState (TypedDict):
    weight_kg : float
    height_m : float
    bmi : float
    category: str
    

# Functions 
def calculate_bmi(state: BMIState) -> BMIState:
    
    weight = state["weight_kg"]
    height = state["height_m"]
    
    bmi = weight/(height**2)

    state["bmi"] = round(bmi, 2)
    
    return state

def label_bmi(state: BMIState) -> BMIState:
    bmi = state["bmi"]
    if (bmi > 20):
        state["category"] = "obese"
        return state
    
    state["category"] = "perfect"
    return state

# define the graph
graph = StateGraph(BMIState)

# add notes
graph.add_node("calculate_bmi",calculate_bmi)
graph.add_node("label_bmi",label_bmi)

# add edges
graph.add_edge(START,"calculate_bmi")
graph.add_edge("calculate_bmi","label_bmi")
graph.add_edge("label_bmi",END)

# compile graph
workflow = graph.compile()

# execute the graph
initial_state = {"weight_kg" : 80, "height_m" : 3}

final_state = workflow.invoke(initial_state)

print(final_state)
Image(workflow.get_graph().draw_mermaid_png())