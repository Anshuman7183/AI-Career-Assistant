from langchain_community.vectorstores import Chroma

from utils.embeddings import embeddings
from utils.llm import llm


# Load Vector Database
db = Chroma(
    persist_directory="vectorstores/interview_db",
    embedding_function=embeddings
)


# Create Retriever
retriever = db.as_retriever(
    search_kwargs={"k": 3}
)


def interview_agent(question):

    docs = retriever.invoke(question)

    context = "\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
    You are an Interview Preparation Expert.

    Use ONLY the provided context.

    Context:
    {context}

    Question:
    {question}

    Give:
    - interview preparation guidance
    - common interview questions
    - important topics
    - preparation strategy
    """

    response = llm.invoke(prompt)

    return response