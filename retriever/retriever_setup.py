import os
from pinecone import Pinecone
from FlagEmbedding import BGEM3FlagModel
from . import HybridSearchRetriever



def setup_pinecone_index():
    """Pinecone 인덱스를 설정하고 연결"""
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    return pc.Index(os.getenv("PINECONE_INDEX"))


def setup_retriever(embedding_model: BGEM3FlagModel, 
                    alpha: float = 0.95, 
                    top_k: int = 3, 
                    min_score: float = 0.55) -> HybridSearchRetriever:
    """
    HybridSearchRetriever를 설정하여 검색을 수행할 수 있는 인스턴스를 생성합니다.

    Args:
        embedding_model (Any): 검색을 위한 임베딩 모델 객체. 텍스트 임베딩을 생성하는 데 사용됩니다.
        alpha (float, optional): 검색 시 가중치를 조정하는 매개변수로, Pinecone 검색과 임베딩 모델 간의 
            가중치를 조절합니다. 기본값은 0.95입니다.
        top_k (int, optional): 검색 결과에서 상위 몇 개의 문서를 반환할지 결정하는 파라미터입니다. 
            기본값은 3입니다.
        min_score (float, optional): 검색 결과의 최소 유사도 점수를 설정하여, 이 값보다 낮은 점수의 
            결과는 무시합니다. 기본값은 0.55입니다.

    Returns:
        HybridSearchRetriever: 설정된 매개변수를 바탕으로 생성된 HybridSearchRetriever 객체로, 
        문서 검색을 수행하는 데 사용됩니다.
    """
    return HybridSearchRetriever(
        pinecone_index=setup_pinecone_index(),
        embedding_model=embedding_model,
        alpha=alpha,
        top_k=top_k,
        min_score=min_score
    )
