import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.title("📄 PDF Chatbot with HuggingFace LLM")

# Sidebar for PDF upload
with st.sidebar:
    st.header("Upload Your PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file:
    # 1️⃣ Extract text from PDF
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    # 2️⃣ Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # 3️⃣ Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 4️⃣ Create FAISS vector store
    vector_store = FAISS.from_texts(chunks, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    # 5️⃣ Setup HuggingFace text-generation pipeline
    pipe = pipeline(
        task="text-generation",
        model="bigscience/bloomz-7b1-mt",  # use smaller model if CPU
        temperature=0.0,
        max_new_tokens=512
    )

    # 6️⃣ Wrap pipeline in LangChain LLM
    llm = HuggingFacePipeline(pipeline=pipe)

    # 7️⃣ User question input and button
    user_question = st.text_input("Enter your question about the PDF:")

    if st.button("Ask"):
        if user_question.strip() != "":
            with st.spinner("Thinking..."):
                # Retrieve relevant chunks
                relevant_docs = retriever.get_relevant_documents(user_question)
                context = "\n".join([doc.page_content for doc in relevant_docs])

                # Ask the LLM
                answer = llm.predict(
                    f"Answer the question based on the context below:\n\nContext:\n{context}\n\nQuestion: {user_question}"
                )
            st.write("**Answer:**")
            st.success(answer)
        else:
            st.warning("Please enter a question before clicking 'Ask'.")

else:
    st.info("👈 Please upload a PDF file to start asking questions.")
