from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.rag.config import Config
import os

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " "]
        )

    def load_document(self, file_path: str):
        """Load a document from file path"""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".txt":
            loader = TextLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        
        documents = loader.load()
        print(f"✅ Loaded {len(documents)} pages from {file_path}")
        return documents

    def split_documents(self, documents):
        """Split documents into chunks"""
        chunks = self.text_splitter.split_documents(documents)
        print(f"✅ Split into {len(chunks)} chunks")
        return chunks

    def process_file(self, file_path: str):
        """Full pipeline: load + split"""
        documents = self.load_document(file_path)
        chunks = self.split_documents(documents)
        return chunks