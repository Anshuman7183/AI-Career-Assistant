from langchain_community.vectorstores import Chroma

from utils.embeddings import embeddings
from utils.llm import get_llm

# Load Vector Database
db = Chroma(
    persist_directory="vectorstores/skills_db",
    embedding_function=embeddings
)

# Create Retriever
retriever = db.as_retriever(
    search_kwargs={"k": 3}
)

def skill_agent(question, model_name):

    llm = get_llm(model_name)

    docs = retriever.invoke(question)

    context = "\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
    You are a Career Skills Expert.

    Answer briefly using the provided context.

    Context:
    {context}

    Question:
    {question}

    Give:
    - required skills
    - technologies
    - learning roadmap
    """

    response = llm.invoke(prompt)

    return response