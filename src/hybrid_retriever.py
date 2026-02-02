from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from rank_bm25 import BM25Okapi  

class HybridRetriever:
    """
    Combines BM25 (keyword) + Semantic (vector) search using Reciprocal Rank Fusion.
    """
    def __init__(self, documents, vector_retriever):
        self.bm25_retriever = BM25Retriever.from_documents(
            documents=documents,
            k=10
        )
        
        self.vector_retriever = vector_retriever

        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.vector_retriever],
            weights=[0.5, 0.5] 
        )

    def as_retriever(self):
        return self.ensemble_retriever