import pandas as pd

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.embeddings import embeddings

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# =========================
# SKILLS DATASET
# =========================

skills_df = pd.read_csv(
    "data/skills/career_skills_dataset.csv"
)

skills_docs = []

for _, row in skills_df.iterrows():

    text = f"""
    Role: {row['role']}

    Skills:
    {row['skills']}
    """

    skills_docs.append(
        Document(page_content=text)
    )

skills_chunks = splitter.split_documents(skills_docs)

skills_db = Chroma.from_documents(
    skills_chunks,
    embeddings,
    persist_directory="vectorstores/skills_db"
)

# =========================
# JOBS DATASET
# =========================

jobs_df = pd.read_csv(
    "data/jobs/jobs_dataset.csv"
)

job_docs = []

for _, row in jobs_df.iterrows():

    text = f"""
    Role: {row['Role']}

    Company:
    {row['Company']}

    Required Skills:
    {row['Required Skills']}

    Description:
    {row['Job Description']}
    """

    job_docs.append(
        Document(page_content=text)
    )

job_chunks = splitter.split_documents(job_docs)

jobs_db = Chroma.from_documents(
    job_chunks,
    embeddings,
    persist_directory="vectorstores/jobs_db"
)

# =========================
# INTERVIEW DATASET
# =========================

interview_df = pd.read_csv(
    "data/interviews/interview_questions_dataset.csv"
)

interview_docs = []

for _, row in interview_df.iterrows():

    text = f"""
    Role: {row['Role']}

    Question:
    {row['Interview Question']}
    """

    interview_docs.append(
        Document(page_content=text)
    )

interview_chunks = splitter.split_documents(
    interview_docs
)

interview_db = Chroma.from_documents(
    interview_chunks,
    embeddings,
    persist_directory="vectorstores/interview_db"
)

print("All vector databases created successfully.")

# =========================
# RESUME DATASET
# =========================

from langchain_community.document_loaders import Docx2txtLoader

import os

resume_docs = []

resume_folder = "data/resumes"

for file in os.listdir(resume_folder):

    if file.endswith(".docx"):

        path = os.path.join(
            resume_folder,
            file
        )

        loader = Docx2txtLoader(path)

        docs = loader.load()

        resume_docs.extend(docs)

resume_chunks = splitter.split_documents(
    resume_docs
)

resume_db = Chroma.from_documents(
    resume_chunks,
    embeddings,
    persist_directory="vectorstores/resume_db"
)