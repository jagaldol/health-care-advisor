import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from FlagEmbedding import BGEM3FlagModel

def load_gemma_model(model_id: str):

    bnb_config = BitsAndBytesConfig(
        # 입력값을 4 bit로 변환
        load_in_4bit=True,
        # 모델을 4 bit으로 양자화
        bnb_4bit_quant_type="nf4",
        # 4 bit 계산에 사용될 데이터 유형, 4비트 부동소수점(bfloat16), 4비트 정수(uint8)
        bnb_4bit_compute_dtype=torch.float16
)


    """모델과 토크나이저 로드"""
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer

def load_embedding_model(model_id: str, device: str = 'cuda'):
    """임베딩 모델 로드"""
    return BGEM3FlagModel(model_id, device=device)