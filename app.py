import streamlit as st
from utils.resume_parser import extract_resume_text
from utils.llm import get_llm
from graph.workflow import graph

st.title("AI Career Assistant")


selected_model = st.selectbox(
    "Choose LLM",
    [
        "tinyllama",
        "phi",
        "mistral"
    ]
)
st.write("Current Model:", selected_model)

question = st.text_input(
    "Ask your career question"
)

uploaded_resume = st.file_uploader(
    "Upload Resume",
    type=["pdf", "docx"]
)

resume_text = ""

if uploaded_resume:

    resume_text = extract_resume_text(uploaded_resume)

    st.success("Resume uploaded successfully!")

if st.button("Submit"):

    with st.spinner("Thinking..."):

        result = graph.invoke({

            "question": question,
            "resume_text": resume_text,
            "model_name": selected_model

        })

    st.subheader("Final Answer")

    response = result["final_response"]

    response = response.replace("Assistant:", "")
    response = response.replace("Answer:", "")

    st.write(response)