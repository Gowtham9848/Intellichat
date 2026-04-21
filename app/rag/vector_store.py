from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.rag.config import Config

class VectorStoreManager:
    def __init__(self):
        # Initializing Pinecone
        self.pc = Pinecone(api_key=Config.PINECONE_API_KEY)
        
        # Using HuggingFace embeddings for simplicity
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        self.index_name = Config.PINECONE_INDEX_NAME
        #check if index exists and create if not in the pinecone environment.
        self._ensure_index_exists()

    def _ensure_index_exists(self):
        """Create Pinecone index if it doesn't exist"""
        existing_indexes = [i.name for i in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            print(f"Creating index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=384,  # all-MiniLM-L6-v2 dimension
                metric="cosine",
                #Using cloud AWS for low latency and high availability for our application.
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print(f"✅ Index '{self.index_name}' created")
        else:
            print(f"✅ Index '{self.index_name}' already exists")

    def store_documents(self, chunks):
        """Store document chunks in Pinecone"""
        vector_store = PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            index_name=self.index_name
        )
        print(f"✅ Stored {len(chunks)} chunks in Pinecone")
        return vector_store
#Used when we want to retrieve the vector store for querying. This allows us to reuse the same index and embeddings for consistent retrieval.
    def get_vector_store(self):
        """Get existing vector store"""
        return PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embeddings
        )