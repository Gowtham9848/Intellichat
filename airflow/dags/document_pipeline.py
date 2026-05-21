from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

# Default arguments for the DAG
default_args = {
    'owner': 'intellichat',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

# Defining the dag where the tasks will be done.
dag = DAG(
    'document_processing_pipeline',
    default_args=default_args,
    description='Automatically process new school documents and store in Pinecone',
    schedule_interval=timedelta(hours=24),  # runs every 24 hours
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['intellichat', 'rag', 'documents']
)

def scan_for_new_documents(**context):
    """Scan uploads folder for new documents"""
    import json
    
    upload_dir = "data/uploads"
    supported_extensions = ['.pdf', '.txt', '.docx']
    
    new_files = []
    for filename in os.listdir(upload_dir):
        #checking for supporting file formats and ignoring .gitkeep
        if any(filename.endswith(ext) for ext in supported_extensions):
            if filename != '.gitkeep':
                file_path = os.path.join(upload_dir, filename)
                new_files.append(file_path)
    
    print(f"✅ Found {len(new_files)} documents to process")
    
    # Push file list to next task via XCom
    context['task_instance'].xcom_push(
        key='new_files',
        value=json.dumps(new_files)
    )
    return new_files

def process_documents(**context):
    """Process and chunk documents"""
    import json
    from app.rag.document_processor import DocumentProcessor
    
    # Get files from previous task
    new_files = json.loads(
        context['task_instance'].xcom_pull(
            task_ids='scan_documents',
            key='new_files'
        )
    )
    
    processor = DocumentProcessor()
    all_chunks = []
    
    for file_path in new_files:
        try:
            chunks = processor.process_file(file_path)
            all_chunks.extend(chunks)
            print(f"✅ Processed: {file_path} → {len(chunks)} chunks")
        except Exception as e:
            print(f"❌ Failed to process {file_path}: {e}")
    
    print(f"✅ Total chunks created: {len(all_chunks)}")
    
    # Push chunks count to next task
    context['task_instance'].xcom_push(
        key='total_chunks',
        value=len(all_chunks)
    )
    return len(all_chunks)

def store_embeddings(**context):
    """Generate embeddings and store in Pinecone"""
    import json
    from app.rag.document_processor import DocumentProcessor
    from app.rag.vector_store import VectorStoreManager
    
    new_files = json.loads(
        context['task_instance'].xcom_pull(
            task_ids='scan_documents',
            key='new_files'
        )
    )
    
    processor = DocumentProcessor()
    vector_store_manager = VectorStoreManager()
    
    total_stored = 0
    for file_path in new_files:
        try:
            chunks = processor.process_file(file_path)
            vector_store_manager.store_documents(chunks)
            total_stored += len(chunks)
            print(f"✅ Stored embeddings for: {file_path}")
        except Exception as e:
            print(f"❌ Failed to store {file_path}: {e}")
    
    print(f"✅ Total embeddings stored: {total_stored}")
    return total_stored

def track_pipeline_metrics(**context):
    """Track pipeline metrics with MLflow"""
    import mlflow
    import json
    
    total_chunks = context['task_instance'].xcom_pull(
        task_ids='process_documents',
        key='total_chunks'
    )
    
    new_files = json.loads(
        context['task_instance'].xcom_pull(
            task_ids='scan_documents',
            key='new_files'
        )
    )
    
    with mlflow.start_run(run_name="document_pipeline"):
        mlflow.log_param("chunk_size", 1000)
        mlflow.log_param("chunk_overlap", 200)
        mlflow.log_param("embedding_model", "all-MiniLM-L6-v2")
        mlflow.log_metric("total_documents", len(new_files))
        mlflow.log_metric("total_chunks", total_chunks or 0)
        mlflow.log_metric("pipeline_status", 1)  # 1 = success
        print("✅ Metrics tracked in MLflow")

# Define tasks
scan_task = PythonOperator(
    task_id='scan_documents',
    python_callable=scan_for_new_documents,
    dag=dag,
)

process_task = PythonOperator(
    task_id='process_documents',
    python_callable=process_documents,
    dag=dag,
)

embed_task = PythonOperator(
    task_id='store_embeddings',
    python_callable=store_embeddings,
    dag=dag,
)

track_task = PythonOperator(
    task_id='track_metrics',
    python_callable=track_pipeline_metrics,
    dag=dag,
)

# Define task order
scan_task >> process_task >> embed_task >> track_task