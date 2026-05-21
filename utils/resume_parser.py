from pypdf import PdfReader
from docx import Document


def extract_resume_text(uploaded_file):

    text = ""

    file_name = uploaded_file.name.lower()

    # PDF
    if file_name.endswith(".pdf"):

        pdf = PdfReader(uploaded_file)

        for page in pdf.pages:
            text += page.extract_text()

    # DOCX
    elif file_name.endswith(".docx"):

        doc = Document(uploaded_file)

        for para in doc.paragraphs:
            text += para.text + "\n"

    return text