# Prompt Chaining LLM Chain Using LangGraph

from langgraph.graph import StateGraph,START,END
from langchain.prompts import ChatPromptTemplate
from typing import TypedDict
from IPython.display import Image
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
from pydantic import BaseModel,Field
from langchain.output_parsers import PydanticOutputParser

load_dotenv()

# Making State

class GradeModel(BaseModel):
    text: str = Field(..., description="Summary of grading")
    grade: int = Field(..., description="Grades (1-10)")
    
class EssayState(TypedDict):
    original_text : str
    structure_grade : GradeModel
    language_grade : GradeModel
    creativity_grade : GradeModel
    summary : str

# Initialize the LLM with Google Gemini 2.5 Flash
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.7,
)

parser = PydanticOutputParser(pydantic_object=GradeModel)
# Functions / Nodes
def structureGrading(state : EssayState):
    original_text = state["original_text"]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a structure grading expert."),
        ("human", "Summarize and Grade the following text on the basis of its structure : {text}. Summarize should be of 2 3 lines. Return the response it the following structure : {structure}")
    ])

    chain = prompt | llm | parser
    
    response = chain.invoke({
        "text" : original_text,
        "structure" : parser.get_format_instructions()
    })
    
    return {"structure_grade" : response}
    
def languageGrading(state : EssayState):
    original_text = state["original_text"]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a grading expert on the basis of language."),
        ("human", "Summarize and Grade the following text on the basis of its language : {text}. Summarization should be of 2 3 lines. Return the response it the following structure : {structure}")
    ])

    chain = prompt | llm | parser
    
    response = chain.invoke({
        "text" : original_text,
        "structure" : parser.get_format_instructions()
    })
    
    # print ("response -->", response)
    
    return {"language_grade" : response}

def creativityGrading(state : EssayState):
    original_text = state["original_text"]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a creativity grading expert."),
        ("human", "Summarize and Grade the following text on the basis of its creativity : {text}. Summarization should be of 2 3 lines. Return the response it the following structure : {structure}")
    ])

    chain = prompt | llm | parser
    
    response = chain.invoke({
        "text" : original_text,
        "structure" : parser.get_format_instructions()
    })
    
    return {"creativity_grade" : response}


def gen_summary(state: EssayState):
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert at summarizing and evaluating essays based on structure, language, and creativity."),
        ("human", """Given the following grading results, write a concise summary of the essay's overall performance. Clearly mention the strengths and weaknesses in each area, and provide the grading scores in a readable format.

Structure Grade: {structure_grades}
Language Grade: {language_grades}
Creativity Grade: {creativity_grades}

Your response should include:
- A brief summary of the essay's overall quality.
- Specific comments on structure, language, and creativity.
- The grading scores for each category.
""")
    ])

    chain = prompt | llm
    
    response = chain.invoke({
        "structure_grades" : state["structure_grade"].model_dump_json(),
        "language_grades" : state["language_grade"].model_dump_json(),
        "creativity_grades" : state["creativity_grade"].model_dump_json()
    }).content

    return {"summary" : response}

graph = StateGraph(EssayState)

# Adding Nodes
graph.add_node("structure", structureGrading)
graph.add_node("language", languageGrading)
graph.add_node("creativity", creativityGrading)
graph.add_node("summarization", gen_summary)

# Adding egdes
graph.add_edge(START,"structure")
graph.add_edge(START,"language")
graph.add_edge(START,"creativity")

graph.add_edge("structure", "summarization")
graph.add_edge("language", "summarization")
graph.add_edge("creativity", "summarization")

graph.add_edge("summarization", END)

# Compiling
workflow = graph.compile()

# Executing
essay = {"original_text" : """ Essay: “The Role of Technology in Modern Education”
In today’s world, technology has become an inseparable part of education. Students now have access to online resources, video lectures, and interactive tools that make learning faster and more engaging. With just a few clicks, one can learn coding, mathematics, or even history from experts around the world.
However, over-reliance on technology can be harmful. Many students become distracted by social media while studying, and sometimes the depth of learning decreases when answers are instantly available online. Traditional methods like reading books and face-to-face discussions still hold great importance in building critical thinking.
In conclusion, technology is a powerful tool that can transform education if used wisely. A balance between digital learning and traditional approaches ensures that students not only gain knowledge but also learn discipline, focus, and deeper understanding. """
}

result = workflow.invoke(essay)
print(result)

Image(workflow.get_graph().draw_mermaid_png())