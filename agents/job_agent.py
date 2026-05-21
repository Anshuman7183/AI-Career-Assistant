from langchain_community.vectorstores import Chroma

from utils.embeddings import embeddings
from utils.llm import llm

# Load Vector Database
db = Chroma(
    persist_directory="vectorstores/jobs_db",
    embedding_function=embeddings
)

# Create Retriever
retriever = db.as_retriever(
    search_kwargs={"k": 3}
)

def job_agent(question):

    docs = retriever.invoke(question)

    context = "\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
    You are a Job Market Expert.

    Use ONLY the provided context.

    Context:
    {context}

    Question:
    {question}

    Give:
    - hiring trends
    - company requirements
    - important technologies
    - job expectations
    """

    response = llm.invoke(prompt)

    return response