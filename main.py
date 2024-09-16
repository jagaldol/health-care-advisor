from typing import Optional
from pydantic import BaseModel
from fastapi import FastAPI
from contextlib import asynccontextmanager

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

# 요청 바디에서 받을 데이터 모델 정의
class RequestBody(BaseModel):
    query: str  # 필수 인자
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

    response = chain["rag chain"].chaining(
        query,
        retriever_kwargs=retriever_kwargs,
        llm_kwargs=llm_kwargs
    )

    # Convert the generator chunks to a single string
    response_string = ''.join(str(chunk) for chunk in response)

    return response_string
