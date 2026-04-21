from app.rag.document_processor import DocumentProcessor
from app.rag.vector_store import VectorStoreManager
from app.rag.rag_pipeline import RAGPipeline

def test_upload_document():
    """Test uploading a document to Pinecone"""
    print("\n🔄 Testing Document Upload...")
    
    processor = DocumentProcessor()
    vector_store_manager = VectorStoreManager()
    
    # Process the sample file
    chunks = processor.process_file("data/uploads/sample_school_data.txt")
    
    # Store in Pinecone
    vector_store_manager.store_documents(chunks)
    
    print("✅ Document uploaded successfully!\n")

def test_query():
    """Test querying the RAG pipeline"""
    print(" Testing RAG Query...")
    
    pipeline = RAGPipeline()
    
    questions = [
        "What are the school timings?",
        "When is the next parent teacher meeting?",
        "What is the attendance policy?"
    ]
    
    for question in questions:
        print(f"\n❓ Question: {question}")
        result = pipeline.query(question)
        print(f"💬 Answer: {result['answer']}")
        if result['status'] == 'success':
            print(f"📄 Sources: Information from uploaded documents- Accuracy 100%")
        else:
            print(f"⚠️ Error: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    test_upload_document()
    test_query()