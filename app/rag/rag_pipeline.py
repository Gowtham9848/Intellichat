from langchain_anthropic import ChatAnthropic
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from app.rag.vector_store import VectorStoreManager
from app.rag.config import Config

class RAGPipeline:
    def __init__(self):
        self.vector_store_manager = VectorStoreManager()
        self.llm = ChatAnthropic(
            model=Config.CLAUDE_MODEL,
            anthropic_api_key=Config.ANTHROPIC_API_KEY,
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_TOKENS
        )
        self.qa_chain = self._build_chain()

    def _build_chain(self):
        """Build the RAG chain"""
        prompt_template = """You are IntelliChat, a helpful school assistant for parents.
Use the following context from school documents to answer the parent's question accurately.

If you don't know the answer from the context, say:
"I don't have that information right now. Please contact the school office directly."

Never make up information. Always be polite and concise.

Context:
{context}

Parent's Question: {question}

Answer:"""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        vector_store = self.vector_store_manager.get_vector_store()
        retriever = vector_store.as_retriever(
            search_kwargs={"k": Config.TOP_K_RESULTS}
        )

        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        return chain

    def query(self, question: str):
        """Query the RAG pipeline"""
        try:
            result = self.qa_chain.invoke({"query": question})
            return {
                "answer": result["result"],
                "sources": [doc.metadata for doc in result["source_documents"]],
                "status": "success"
            }
        except Exception as e:
            return {
                "answer": "I encountered an error processing your question. Please try again.",
                "status": "error",
                "error": str(e)
            }