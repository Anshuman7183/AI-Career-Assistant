# AI Career Assistant

A Multi-Agent RAG-based AI Career Assistant built using LangGraph, LangChain, ChromaDB, HuggingFace, and Streamlit.

This system uses multiple AI agents with separate datasets and vector databases to provide accurate career guidance related to:
- Skills
- Jobs
- Resume Building
- Interview Preparation

---

# Features

- Multi-Agent Architecture
- Retrieval Augmented Generation (RAG)
- Semantic Search using Vector Databases
- Resume Guidance
- Interview Preparation
- Career Roadmaps
- Job Market Insights
- Skill Recommendations
- Streamlit Web Interface

---

# Tech Stack

- Python
- LangChain
- LangGraph
- ChromaDB
- HuggingFace Transformers
- Sentence Transformers
- Streamlit
- Pandas

---

# Project Architecture

User Question
↓
LangGraph Workflow
↓
Skills Agent
Jobs Agent
Interview Agent
Resume Agent
↓
Each Agent retrieves relevant context from its own Vector Database
↓
LLM combines all responses
↓
Final Career Guidance Response

---

# Agents Used

## 1. Skills Agent
Provides:
- required skills
- technologies
- learning roadmaps

Dataset:
- career_skills_dataset.csv

---

## 2. Job Agent
Provides:
- hiring trends
- company expectations
- required technologies

Dataset:
- jobs_dataset.csv

---

## 3. Interview Agent
Provides:
- interview questions
- preparation strategies
- important topics

Dataset:
- interview_questions_dataset.csv

---

## 4. Resume Agent
Provides:
- ATS resume tips
- resume improvements
- project suggestions

Dataset:
- sample resumes (.docx)

---

# Folder Structure

```text
AI-Career-Assistant/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── skills/
│   ├── jobs/
│   ├── interviews/
│   └── resumes/
│
├── vectorstores/
│
├── agents/
│   ├── skill_agent.py
│   ├── job_agent.py
│   ├── interview_agent.py
│   └── resume_agent.py
│
├── graph/
│   └── workflow.py
│
├── utils/
│   ├── embeddings.py
│   ├── llm.py
│   └── vectorstore.py