import utils
from ragchain import RAGChian
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run RAGChian inference with a query.")
    parser.add_argument(
        'query', 
        type=str, 
        help='The input query string to be processed by RAGChian'
    )
    args = parser.parse_args()


    # 환경 설정 로드
    utils.load_environment()

    chain = RAGChian()

    chain.chaining(
        args.query,
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