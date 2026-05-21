import streamlit as st

from graph.workflow import graph

st.title("AI Career Assistant")

question = st.text_input(
    "Ask your career question"
)

if st.button("Submit"):

    with st.spinner("Thinking..."):

        result = graph.invoke({

            "question": question

        })

    st.subheader("Final Answer")

    st.write(result["final_response"])