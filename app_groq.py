import os

import faiss
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="RAG Question Answering")
st.title("RAG-Based Question Answering System")


@st.cache_resource
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def get_api_key():
    key = os.environ.get("GROQ_API_KEY")
    if key:
        return key
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return None


def chunk_text(text, size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


embedder = load_embedder()

with st.sidebar:
    st.header("1. Upload documents")
    files = st.file_uploader("PDF files", type="pdf", accept_multiple_files=True)
    if st.button("Process documents"):
        if not files:
            st.warning("Pehle koi PDF upload karo.")
        else:
            chunks, meta = [], []
            for f in files:
                reader = PdfReader(f)
                for page_no, page in enumerate(reader.pages, start=1):
                    text = page.extract_text() or ""
                    for c in chunk_text(text):
                        if c.strip():
                            chunks.append(c)
                            meta.append((f.name, page_no))
            if not chunks:
                st.error("PDF se text nahi nikla (scanned PDF ho sakta hai).")
            else:
                vecs = embedder.encode(chunks, normalize_embeddings=True).astype("float32")
                index = faiss.IndexFlatIP(vecs.shape[1])
                index.add(vecs)
                st.session_state["index"] = index
                st.session_state["chunks"] = chunks
                st.session_state["meta"] = meta
                st.success(f"{len(chunks)} chunks ready.")

st.header("2. Ask a question")
question = st.text_input("Your question")

if st.button("Get answer") and question:
    if "index" not in st.session_state:
        st.warning("Pehle sidebar mein documents process karo.")
        st.stop()

    qv = embedder.encode([question], normalize_embeddings=True).astype("float32")
    scores, ids = st.session_state["index"].search(qv, 3)
    hits = [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]
    context = "\n\n".join(st.session_state["chunks"][i] for i, _ in hits)

    st.subheader("Answer")
    key = get_api_key()
    if not key:
        st.error("GROQ_API_KEY nahi mili. Key set karo (README dekho).")
    else:
        prompt = (
            "Neeche diye context ke basis par hi sawal ka jawab do. "
            "Agar jawab context mein nahi hai to kaho 'Document mein yeh nahi mila'. "
            "Jawab usi language mein do jisme sawal poocha gaya hai.\n\n"
            f"Context:\n{context}\n\nSawal: {question}"
        )
        try:
            client = Groq(api_key=key)
            r = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[{"role": "user", "content": prompt}],
            )
            st.write(r.choices[0].message.content)
        except Exception as e:
            st.error(f"Groq error: {e}")

    st.subheader("Supporting passages")
    for i, s in hits:
        name, page_no = st.session_state["meta"][i]
        with st.expander(f"{name} - page {page_no} (similarity {s:.2f})"):
            st.write(st.session_state["chunks"][i])
