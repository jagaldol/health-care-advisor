import utils
from ragchain import RAGChian

def main():
    # 환경 설정 로드
    utils.load_environment()

    chain = RAGChian()

    chain.chaining(
        "머리아프다.",
        retriever_kwargs={
            "alpha": 0.95,
            "top_k": 3,
            "min_score": 0.55
        },
        llm_kwargs={
            'temperature': 0.5,
            'top_p': 0.7,
            'repetition_penalty': 1.1
        }
    )

if __name__ == "__main__":
    main()