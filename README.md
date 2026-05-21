# AI Career Assistant

An Agentic AI based Career Guidance System built using LangChain, LangGraph, Ollama, ChromaDB, and Streamlit.

This project uses multiple AI agents with separate datasets and vector databases to provide accurate and context-aware career guidance.

---

# Features

- Multi-Agent AI Architecture
- RAG (Retrieval Augmented Generation)
- Multi-LLM Support using Ollama
- Resume Upload and Analysis
- ChromaDB Vector Database
- LangGraph Workflow Orchestration
- Streamlit Frontend
- ATS Resume Suggestions
- Career Roadmaps
- Interview Preparation Guidance
- Job Market Insights

---

# Agents Used

## 1. Skills Agent
Provides:
- Required skills
- Technologies
- Learning roadmap

Dataset:
- Career skills dataset

---

## 2. Job Agent
Provides:
- Hiring trends
- Company requirements
- Important technologies
- Job expectations

Dataset:
- Job market dataset

---

## 3. Interview Agent
Provides:
- Interview preparation
- Common interview questions
- Important topics
- Preparation strategies

Dataset:
- Interview questions dataset

---

## 4. Resume Agent
Provides:
- ATS suggestions
- Missing skills
- Resume improvements
- Project recommendations

Dataset:
- Resume dataset + uploaded resume

---

# Tech Stack

## Frontend
- Streamlit

## Backend
- Python

## Frameworks
- LangChain
- LangGraph

## Vector Database
- ChromaDB

## Embeddings
- HuggingFace Embeddings

## LLMs
- TinyLlama
- Phi
- Mistral

## Local LLM Runtime
- Ollama

---

# Project Architecture

```text
User Question
      ↓
Streamlit UI
      ↓
LangGraph Workflow
      ↓
Multiple AI Agents
      ↓
ChromaDB Retrieval
      ↓
Ollama LLMs
      ↓
Final Aggregated Response

Folder Structure
AI Career Assistant/
│
├── agents/
│   ├── skill_agent.py
│   ├── job_agent.py
│   ├── interview_agent.py
│   └── resume_agent.py
│
├── data/
│   ├── skills/
│   ├── jobs/
│   ├── interviews/
│   └── resumes/
│
├── graph/
│   └── workflow.py
│
├── utils/
│   ├── embeddings.py
│   ├── llm.py
│   ├── resume_parser.py
│   └── vectorstore.py
│
├── vectorstores/
│
├── .streamlit/
│   └── config.toml
│
├── app.py
├── requirements.txt
└── README.md