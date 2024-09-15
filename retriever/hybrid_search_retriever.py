from typing import List
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever

from FlagEmbedding import BGEM3FlagModel
from pinecone import Index

class HybridSearchRetriever(BaseRetriever):

    pinecone_index:Index
    embedding_model:BGEM3FlagModel
    alpha:float
    top_k: int
    min_score:float

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[str]:
        """Sync implementations for retriever."""
        user_query_emb = self.embedding_model.encode(query, return_dense=True, return_sparse=True, return_colbert_vecs=False) #dense, sparse 둘 다 반환함
        
        query_dense_vector = user_query_emb['dense_vecs'].tolist()
        user_query_sparse = user_query_emb['lexical_weights']
        query_sparse_vector = {
            'indices': [int(k) for k in user_query_sparse.keys() if k.isdigit()], #isdigit() 안하면 에러뜨더라
            'values': [float(v) for k, v in user_query_sparse.items() if k.isdigit()]
        }

        hdense, hsparse = self._hybrid_score_norm(query_dense_vector, query_sparse_vector, alpha=self.alpha)

        hybrid_query_response = self.pinecone_index.query(
            top_k=self.top_k,
            vector=hdense,
            sparse_vector=hsparse,
            include_metadata=True,
        )
        
        documents = [
            f"{match['metadata']['answer_intro']}\n"
            f"{match['metadata']['answer_body']}\n"
            f"{match['metadata']['answer_conclusion']}"
            for match in hybrid_query_response['matches']
            if match['score'] >= self.min_score
        ]
        return documents
    
    def _hybrid_score_norm(self, dense, sparse, alpha: float):
        """Hybrid score using a convex combination

        alpha * dense + (1 - alpha) * sparse

        Args:
            dense: Array of floats representing
            sparse: a dict of `indices` and `values`
            alpha: scale between 0 and 1
        """
        if alpha < 0 or alpha > 1:
            raise ValueError("Alpha must be between 0 and 1")
        hs = {
            'indices': sparse['indices'],
            'values':  [v * (1 - alpha) for v in sparse['values']]
        }
        return [v * alpha for v in dense], hs