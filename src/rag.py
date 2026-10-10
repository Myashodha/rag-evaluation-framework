from pathlib import Path

import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.llm import get_llm

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DOCS_DIR = ROOT / "data" / "documents"
NO_ANSWER = "I don't have enough information to answer that."


class RAGPipeline:
    def __init__(self, docs_dir=DEFAULT_DOCS_DIR, chunk_size=300, chunk_overlap=50,
                 top_k=3, collection_name="GubbisCart", llm=None):
        self.docs_dir = Path(docs_dir)
        self.top_k = top_k
        self.collection_name = collection_name
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.embedder = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.llm = llm or get_llm()
        self.client = chromadb.Client()   # in-memory; rebuilt in seconds
        self.collection = None

   
    def load_documents(self):
        return [{"source": f.name, "text": f.read_text(encoding="utf-8")}
                for f in sorted(self.docs_dir.glob("*.txt"))]

    def chunk(self, docs):
        chunks = []
        for d in docs:
            for i, piece in enumerate(self.splitter.split_text(d["text"])):
                chunks.append({"id": f"{d['source']}-{i}", "text": piece,
                               "source": d["source"]})
        return chunks

 
    def build_index(self):
        chunks = self.chunk(self.load_documents())
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )
        self.collection.add(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            embeddings=self.embedder.embed_documents([c["text"] for c in chunks]),
            metadatas=[{"source": c["source"]} for c in chunks],
        )
        return len(chunks)

  
    def retrieve(self, question):
        q_vec = self.embedder.embed_query(question)
        res = self.collection.query(query_embeddings=[q_vec], n_results=self.top_k)
        return [
            {"text": t, "source": m["source"], "similarity": round(1 - d, 3)}
            for t, m, d in zip(res["documents"][0], res["metadatas"][0],
                               res["distances"][0])
        ]

    def build_prompt(self, question, retrieved):
        context = "\n\n".join(f"[Source: {r['source']}]\n{r['text']}" for r in retrieved)
        return f"""You are a customer support assistant for Gubbiscart.
Answer the question using ONLY the context below.
If the context does not contain the answer, reply exactly: "{NO_ANSWER}"
Be concise.

Context:
{context}

Question: {question}
Answer:"""

    def ask(self, question):
        retrieved = self.retrieve(question)
        answer = self.llm.invoke(self.build_prompt(question, retrieved)).content.strip()
        return {
            "question": question,
            "answer": answer,
            "contexts": [r["text"] for r in retrieved],
            "sources": [r["source"] for r in retrieved],
            "retrieved": retrieved,
        }


    