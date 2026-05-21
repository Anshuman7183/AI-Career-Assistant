from langchain_community.vectorstores import Chroma

from utils.embeddings import embeddings
from utils.llm import get_llm


# Load Vector Database
db = Chroma(
    persist_directory="vectorstores/resume_db",
    embedding_function=embeddings
)


# Create Retriever
retriever = db.as_retriever(
    search_kwargs={"k": 3}
)


def resume_agent(
    question,
    resume_text="",
    model_name="tinyllama"
):

    llm = get_llm(model_name)

    docs = retriever.invoke(question)

    context = "\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
    You are a Resume Expert.

    Resume Content:
    {resume_text}

    Also analyze the uploaded resume and provide:
    - ATS suggestions
    - missing skills
    - resume improvements
    - project recommendations

    Answer briefly using the provided context.

    Context:
    {context}

    Question:
    {question}

    Give:
    - resume improvement suggestions
    - ATS optimization tips
    - important resume sections
    - project recommendations
    """

    response = llm.invoke(prompt)

    return response