from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import models, ragchain, retriever, template, utils


def main():
    # 환경 설정 로드
    utils.load_environment()

    # 모델 및 토크나이저 로드
    gemma_2_model, gemma_2_tokenizer = models.load_gemma_model("google/gemma-2-2b-it")
    embedding_model = models.load_embedding_model('BAAI/bge-m3')

    # Retriever 설정
    rag_retriever = retriever.setup_retriever(embedding_model)

    # 프롬프트 생성
    prompt = template.create_chat_prompt_template(gemma_2_tokenizer)
    
    # 텍스트 생성 파이프라인 설정
    llm = ragchain.setup_text_generation_pipeline(gemma_2_model, gemma_2_tokenizer)


    # RAG 체인 설정 및 답변 생성
    rag_chain = ({"documents": rag_retriever, "question": RunnablePassthrough()}
                 | prompt
                 | llm
                 | StrOutputParser())
    
    answer = rag_chain.invoke("머리 아프다")

if __name__ == "__main__":
    main()