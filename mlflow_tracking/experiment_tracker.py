import mlflow
import os
from datetime import datetime

class ExperimentTracker:
    def __init__(self, experiment_name="intellichat_rag"):
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name
        print(f"✅ MLflow initialized")

    def track_document_processing(self, file_path, chunks, processing_time):
        """Track document processing metrics"""
        run_name = f"doc_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_param("file", os.path.basename(file_path))
            mlflow.log_metric("chunks", len(chunks))
            mlflow.log_metric("time_sec", processing_time)
            print(f"✅ Logged metrics to run: {run.info.run_id}")

    def track_query_performance(self, question, answer, retrieval_time, num_sources):
        """Track query performance"""
        with mlflow.start_run(run_name="query_performance"):
            mlflow.log_metric("retrieval_time", retrieval_time)
            mlflow.log_metric("answer_length", len(answer))
            mlflow.log_metric("num_sources", num_sources)

    def track_embedding_quality(self, chunks, run_name="embedding_quality"):
        """Track embedding quality"""
        with mlflow.start_run(run_name=run_name):
            mlflow.log_metric("total_chunks", len(chunks))
            if chunks:
                avg_len = sum(len(c.page_content) for c in chunks) / len(chunks)
                mlflow.log_metric("avg_chunk_length", avg_len)