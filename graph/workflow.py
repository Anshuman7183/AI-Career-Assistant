from typing import TypedDict

from langgraph.graph import StateGraph
from langgraph.graph import END

from agents.skill_agent import skill_agent
from agents.job_agent import job_agent
from agents.interview_agent import interview_agent
from agents.resume_agent import resume_agent

from utils.llm import llm


# =========================
# STATE
# =========================

class GraphState(TypedDict):

    question: str

    skill_response: str

    job_response: str

    interview_response: str

    resume_response: str

    final_response: str


# =========================
# AGENT NODES
# =========================

def skills_node(state):

    question = state["question"]

    response = skill_agent(question)

    return {
        "skill_response": response
    }


def jobs_node(state):

    question = state["question"]

    response = job_agent(question)

    return {
        "job_response": response
    }


def interview_node(state):

    question = state["question"]

    response = interview_agent(question)

    return {
        "interview_response": response
    }


def resume_node(state):

    question = state["question"]

    response = resume_agent(question)

    return {
        "resume_response": response
    }


# =========================
# FINAL AGGREGATOR
# =========================

def final_node(state):

    prompt = f"""
    You are an AI Career Assistant.

    Combine all agent responses and generate
    one final accurate answer.

    Skills Agent:
    {state['skill_response']}

    Job Agent:
    {state['job_response']}

    Interview Agent:
    {state['interview_response']}

    Resume Agent:
    {state['resume_response']}
    """

    final_answer = llm.invoke(
    f"""
    Generate a clean final career guidance answer.

    Do NOT repeat the prompts.
    Do NOT repeat agent labels.

    User Question:
    {state["question"]}

    Agent Information:
    {prompt}
    """
)

    return {
        "final_response": final_answer
    }


# =========================
# BUILD GRAPH
# =========================

builder = StateGraph(GraphState)

builder.add_node("skills", skills_node)

builder.add_node("jobs", jobs_node)

builder.add_node("interview", interview_node)

builder.add_node("resume", resume_node)

builder.add_node("final", final_node)


# FLOW

builder.set_entry_point("skills")

builder.add_edge("skills", "jobs")

builder.add_edge("jobs", "interview")

builder.add_edge("interview", "resume")

builder.add_edge("resume", "final")

builder.add_edge("final", END)


# COMPILE GRAPH

graph = builder.compile()