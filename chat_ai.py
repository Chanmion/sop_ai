import streamlit as st
import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import torch
import re

# --------------------------- Paths ---------------------------
PDF_FOLDER = "./sop_pdfs"
TXT_FOLDER = "./sop_pdfs"
INDEX_FOLDER = "./faiss.index"
INDEX_NAME = "index"

# --------------------------- Load Documents ---------------------------
documents = []
document_sources = []

for pdf_file in Path(PDF_FOLDER).glob("*.pdf"):
    loader = PyPDFLoader(str(pdf_file))
    docs = loader.load()
    documents.extend(docs)
    document_sources.extend([pdf_file.name] * len(docs))

for txt_file in Path(TXT_FOLDER).glob("*.txt"):
    loader = TextLoader(str(txt_file), encoding="utf-8")
    docs = loader.load()
    documents.extend(docs)
    document_sources.extend([txt_file.name] * len(docs))

# --------------------------- Text Chunking ---------------------------
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
documents = text_splitter.split_documents(documents)
st.write(f"Loaded {len(documents)} chunks from PDFs and TXTs.")

# --------------------------- Embeddings ---------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --------------------------- FAISS ---------------------------
if os.path.exists(f"{INDEX_FOLDER}/{INDEX_NAME}.faiss"):
    vectorstore = FAISS.load_local(INDEX_FOLDER, embeddings, allow_dangerous_deserialization=True)
    if documents:
        vectorstore.add_documents(documents)
        vectorstore.save_local(INDEX_FOLDER, INDEX_NAME)
else:
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(INDEX_FOLDER, INDEX_NAME)

# --------------------------- Load HuggingFace Model ---------------------------
@st.cache_resource
def load_hf_model():
    model_name = "tiiuae/falcon-7b-instruct"  # can be changed to other HF models
    device = 0 if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto" if device == 0 else None)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
    return generator

generator = load_hf_model()

# --------------------------- Streamlit Chat UI ---------------------------
st.title("📄 SOP AI Assistant (Dynamic Positions + Typo-Tolerant)")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if user_input := st.chat_input("Ask a question about your SOPs... (typos OK)"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # --------------------------- Semantic Search ---------------------------
    docs = vectorstore.similarity_search(user_input.lower(), k=len(documents))

    seen_texts = set()
    unique_chunks = []
    for i, doc in enumerate(docs):
        text = doc.page_content.strip()
        if text not in seen_texts:
            seen_texts.add(text)
            unique_chunks.append(text)

    # --------------------------- Include Document Names ---------------------------
    context_list = []
    for i, chunk in enumerate(unique_chunks):
        doc_name = document_sources[i] if i < len(document_sources) else "Unknown"
        context_list.append(f"[Document: {doc_name}]\n{chunk}")
    context = "\n\n".join(context_list)

    st.write("### Retrieved Context (Debug)")
    st.write(context if context else "_No relevant chunks found._")

    # --------------------------- Enhanced Prompt ---------------------------
    full_prompt = f"""
You are an intelligent assistant that answers questions using ONLY the SOP documents below.
Even if the user has typos or uses different words, understand their intent.
Combine information across documents to provide accurate, concise, and professional answers.

Instructions:
- Reason and summarize naturally.
- If the user asks about positions, job openings, or roles:
  - List ALL positions as bullet points with document names in parentheses.
  - Include location, reporting manager, and key details if available.
- If information is missing, reply: 'This information is not available in the SOP documents.'

Context:
{context}

User Question:
{user_input}
"""

    # --------------------------- Generate Response ---------------------------
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        result = generator(full_prompt, max_new_tokens=512, do_sample=True, temperature=0.7)[0]["generated_text"]

        # --------------------------- Post-processing ---------------------------
        # Remove duplicates and empty lines
        lines = [line.strip() for line in result.split("\n") if line.strip()]
        lines = list(dict.fromkeys(lines))

        # Detect positions in the output (with parentheses)
        table_data = []
        for line in lines:
            matches = re.findall(r"(.*?)[(](.*?)[)]", line)
            for pos, doc in matches:
                table_data.append({"Position": pos.strip("- ").strip(), "Document": doc.strip()})

        # Streamed display
        streamed_text = ""
        for word in " ".join(lines).split():
            streamed_text += word + " "
            message_placeholder.markdown(streamed_text)

        # Show structured table if positions are detected
        if table_data:
            df = pd.DataFrame(table_data)
            st.write("### Open Positions Table")
            st.table(df)

    st.session_state.messages.append({"role": "assistant", "content": "\n".join(lines)})
