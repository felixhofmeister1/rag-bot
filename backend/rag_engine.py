import os
import uuid
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

INDEX_PATH = "./faiss_index"

class RAGEngine:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        
        if os.path.exists(INDEX_PATH):
            self.vector_store = FAISS.load_local(
                INDEX_PATH, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
        else:
            self.vector_store = None

    def process_and_ingest(self, file_path: str, filename: str) -> int:
        ext = filename.lower().split('.')[-1]
        
        if ext == "pdf":
            loader = PyPDFLoader(file_path)
        elif ext in ["txt", "md"]:
            loader = TextLoader(file_path, autodetect_encoding=True)
        else:
            raise ValueError(f"Unsupported file format: .{ext}")
        
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            length_function=len
        )
        chunks = text_splitter.split_documents(docs)
        
        for idx, chunk in enumerate(chunks):
            chunk.metadata["source"] = filename
            chunk.metadata["chunk_id"] = str(uuid.uuid4())[:8]
            if "page" not in chunk.metadata:
                chunk.metadata["page"] = 1

        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        else:
            self.vector_store.add_documents(chunks)
            
        self.vector_store.save_local(INDEX_PATH)
        return len(chunks)

    def query(self, question: str, top_k: int = 4) -> Dict[str, Any]:
        if self.vector_store is None:
            return {
                "answer": "No documents uploaded yet.",
                "citations": []
            }
            
        results = self.vector_store.similarity_search_with_relevance_scores(question, k=top_k)
        
        if not results:
            return {
                "answer": "No relevant documents found in the knowledge base.",
                "citations": []
            }

        context_blocks = []
        citations = []
        
        for doc, score in results:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", 1)
            chunk_id = doc.metadata.get("chunk_id", "N/A")
            
            citations.append({
                "source": source,
                "page": page,
                "score": round(score, 3),
                "snippet": doc.page_content[:180] + "..."
            })
            
            context_blocks.append(
                f"[Source: {source} | Page: {page} | Chunk: {chunk_id}]\n{doc.page_content}"
            )

        formatted_context = "\n\n---\n\n".join(context_blocks)
        
        system_prompt = (
            "You are an expert enterprise research assistant. Answer the user's question using ONLY "
            "the provided context. Cite your sources clearly using [Source: <filename>, Page: <page>] notation.\n\n"
            f"Context:\n{formatted_context}"
        )
        
        response = self.llm.invoke([
            ("system", system_prompt),
            ("user", question)
        ])
        
        return {
            "answer": response.content,
            "citations": citations
        }

    def get_stats(self) -> Dict[str, Any]:
        total = self.vector_store.index.ntotal if self.vector_store else 0
        return {
            "total_chunks": total,
            "db_path": INDEX_PATH
        }

    def clear_database(self) -> bool:
        if os.path.exists(INDEX_PATH):
            import shutil
            shutil.rmtree(INDEX_PATH)
        self.vector_store = None
        return True