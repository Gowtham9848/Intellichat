import sys
if sys.platform == 'win32':
    import types
    sys.modules['readline'] = types.ModuleType('readline')

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from app.rag.config import Config

class VectorStoreManager:
    def __init__(self):
        self.pc = Pinecone(api_key=Config.PINECONE_API_KEY)
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        self.index_name = Config.PINECONE_INDEX_NAME
        self._ensure_index_exists()

    def _ensure_index_exists(self):
        existing_indexes = [i.name for i in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            print(f"Creating index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print(f"✅ Index '{self.index_name}' created")
        else:
            print(f"✅ Index '{self.index_name}' already exists")

    def store_documents(self, chunks):
        vector_store = PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            index_name=self.index_name
        )
        print(f"✅ Stored {len(chunks)} chunks in Pinecone")
        return vector_store

    def get_vector_store(self):
        return PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embeddings
        )