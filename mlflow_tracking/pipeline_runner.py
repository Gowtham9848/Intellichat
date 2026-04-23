import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag.document_processor import DocumentProcessor
from app.rag.vector_store import VectorStoreManager
from mlflow_tracking.experiment_tracker import ExperimentTracker

class PipelineRunner:
    def __init__(self):
        self.processor = DocumentProcessor()
        self.vector_store = VectorStoreManager()
        self.tracker = ExperimentTracker()
        print("✅ Pipeline Runner initialized")

    def scan_documents(self, upload_dir="data/uploads"):
        """Step 1 - Scan for new documents"""
        print("\n📂 Step 1: Scanning for documents...")
        
        supported = ['.pdf', '.txt', '.docx']
        files = []
        
        for filename in os.listdir(upload_dir):
            if any(filename.endswith(ext) for ext in supported):
                if filename != '.gitkeep':
                    files.append(os.path.join(upload_dir, filename))
        
        print(f"✅ Found {len(files)} documents: {files}")
        return files

    def process_and_store(self, files):
        """Step 2 - Process and store documents"""
        print("\n⚙️ Step 2: Processing documents...")
        
        all_chunks = []
        for file_path in files:
            start_time = time.time()
            
            try:
                chunks = self.processor.process_file(file_path)
                processing_time = time.time() - start_time
                
                # Track with MLflow
                self.tracker.track_document_processing(
                    file_path, chunks, processing_time
                )
                
                # Store in Pinecone
                self.vector_store.store_documents(chunks)
                all_chunks.extend(chunks)
                
                print(f"✅ Processed and stored: {file_path}")
            except Exception as e:
                print(f"❌ Failed: {file_path} — {e}")
        
        return all_chunks

    def track_quality(self, all_chunks):
        """Step 3 - Track embedding quality"""
        print("\n📊 Step 3: Tracking embedding quality...")
        self.tracker.track_embedding_quality(all_chunks)
        print("✅ Quality metrics saved to MLflow")

    def run(self):
        """Run the full pipeline"""
        print("\n🚀 Starting IntelliChat Pipeline...")
        print("=" * 50)
        
        # Step 1
        files = self.scan_documents()
        if not files:
            print("⚠️ No documents found in data/uploads/")
            return
        
        # Step 2
        all_chunks = self.process_and_store(files)
        
        # Step 3
        self.track_quality(all_chunks)
        
        print("\n" + "=" * 50)
        print("✅ Pipeline completed successfully!")
        print(f"📊 Total chunks processed: {len(all_chunks)}")
        print("\nRun 'mlflow ui' to see the dashboard!")

if __name__ == "__main__":
    runner = PipelineRunner()
    runner.run()