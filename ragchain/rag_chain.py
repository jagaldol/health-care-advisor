from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import models, ragchain, retriever, template

class RAGChian:
    def __init__(self):
        # 모델 및 토크나이저 로드
        self.gemma_2_model, self.gemma_2_tokenizer = models.load_gemma_model("google/gemma-2-2b-it")
        self.embedding_model = models.load_embedding_model('BAAI/bge-m3')

    def chaining(self, 
                 query: str,
                 history: list[str] = None,
                 retriever_kwargs: dict = None,
                 llm_kwargs: dict = None) -> str:
        """
        입력된 쿼리에 대해 RAG (Retrieval-Augmented Generation) 체인을 구성하여 응답을 생성합니다.

        Args:
            query (str): 사용자가 입력한 질문이나 요청 텍스트입니다.
            retriever_kwargs (dict, optional): Retriever 설정에 필요한 하이퍼파라미터 딕셔너리입니다.
                가능한 키로는 'alpha', 'top_k', 'min_score'가 있습니다. 기본값은 None입니다.
            llm_kwargs (dict, optional): 텍스트 생성 파이프라인의 하이퍼파라미터 딕셔너리입니다.
                가능한 키로는 'temperature', 'top_p', 'repetition_penalty'가 있습니다. 기본값은 None입니다.

        Returns:
            str: 입력 쿼리에 대한 생성된 응답 텍스트를 반환합니다.
        """

        if history is None:
            history = []

        # retriever_kwargs와 llm_kwargs 기본값 설정
        if retriever_kwargs is None:
            retriever_kwargs = {}

        if llm_kwargs is None:
            llm_kwargs = {}

        prev_chat = [{"role": "user" if idx % 2 == 0 else "assistant", "content": message} for idx, message in enumerate(history)]

        # Retriever 설정
        rag_retriever = retriever.setup_retriever(self.embedding_model, **retriever_kwargs)

        # 프롬프트 생성
        prompt = template.create_chat_prompt_template(self.gemma_2_tokenizer, prev_chat=prev_chat)
        
        # 텍스트 생성 파이프라인 설정
        llm = ragchain.setup_text_generation_pipeline(self.gemma_2_model, self.gemma_2_tokenizer, **llm_kwargs)


        # RAG 체인 설정 및 답변 생성
        rag_chain = ({"documents": rag_retriever, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser())
        
        return rag_chain.stream(query)
