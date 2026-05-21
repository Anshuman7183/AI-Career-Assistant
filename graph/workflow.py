from typing import TypedDict

from langgraph.graph import StateGraph
from langgraph.graph import END

from agents.skill_agent import skill_agent
from agents.job_agent import job_agent
from agents.interview_agent import interview_agent
from agents.resume_agent import resume_agent

from utils.llm import get_llm


# =========================
# STATE
# =========================

class GraphState(TypedDict):

    model_name: str

    question: str

    resume_text: str

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

    response = skill_agent(
        question,
        state["model_name"]
    )

    return {
        "skill_response": response
    }


def jobs_node(state):

    question = state["question"]

    response = job_agent(
        question,
        state["model_name"]
    )

    return {
        "job_response": response
    }


def interview_node(state):

    question = state["question"]

    response = interview_agent(
        question,
        state["model_name"]
    )

    return {
        "interview_response": response
    }


def resume_node(state):

    question = state["question"]

    resume_text = state.get(
        "resume_text",
        ""
    )

    response = resume_agent(
        question,
        resume_text,
        state["model_name"]
    )

    return {
        "resume_response": response
    }


# =========================
# FINAL AGGREGATOR
# =========================

def final_node(state):

    llm = get_llm(
        state["model_name"]
    )

    prompt = f"""
    User Question:
    {state["question"]}

    Skills Information:
    {state["skill_response"]}

    Job Information:
    {state["job_response"]}

    Interview Information:
    {state["interview_response"]}

    Resume Information:
    {state["resume_response"]}

    Generate a SHORT and CLEAN final career guidance answer.

    Give:
    - roadmap
    - skills
    - projects
    - interview preparation

    Keep response professional and concise.
    """

    final_answer = llm.invoke(prompt)

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


# =========================
# FLOW
# =========================

builder.set_entry_point("skills")

builder.add_edge("skills", "jobs")

builder.add_edge("jobs", "interview")

builder.add_edge("interview", "resume")

builder.add_edge("resume", "final")

builder.add_edge("final", END)


# =========================
# COMPILE GRAPH
# =========================

graph = builder.compile()