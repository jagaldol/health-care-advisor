from typing import Optional
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import uvicorn

import utils
from ragchain import RAGChian

chain = {}

# 앱 시작 시 한 번 실행되는 초기화 함수
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 환경 설정 로드
    utils.load_environment()
    chain["rag chain"] = RAGChian()
    yield

app = FastAPI(lifespan=lifespan)

origins = [
    "https://undoc.vercel.app",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 요청 바디에서 받을 데이터 모델 정의
class RequestBody(BaseModel):
    query: str  # 필수 인자
    history: Optional[list[str]] = None
    rt_alpha: Optional[float] = 0.95
    rt_top_k: Optional[int] = 3 
    rt_min_score: Optional[float] = 0.55
    lm_temperature: Optional[float] = 0.5
    lm_top_p: Optional[float] = 0.7
    lm_repetition_penalty: Optional[float] = 1.1


@app.post("/inference")
async def run_inference(body: RequestBody):
    # 요청에서 query와 hyperparameters를 추출
    query = body.query
    history = body.history if body.history is not None else []
    retriever_kwargs = {
        "alpha": body.rt_alpha,
        "top_k": body.rt_top_k,
        "min_score": body.rt_min_score
    }

    llm_kwargs = {
        "temperature": body.lm_temperature,
        "top_p": body.lm_top_p,
        "repetition_penalty": body.lm_repetition_penalty
    }

    # chaining 함수 호출 및 generator 생성
    response = chain["rag chain"].chaining(
        query,
        history=history,
        retriever_kwargs=retriever_kwargs,
        llm_kwargs=llm_kwargs
    )

    # 단어별로 데이터를 스트리밍하기 위한 generator
    def word_streamer():
        for word in response:  # response가 generator일 때 사용
            escaped_word = word.replace('\n', '<enter_token>')
            escaped_word = escaped_word.replace(' ', '<space_token>')
            yield f"data: {escaped_word}\n\n"  # SSE 포맷 유지 또는 필요시 수정
    
    return StreamingResponse(word_streamer(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0")
