import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from FlagEmbedding import BGEM3FlagModel

def load_gemma_model(model_id: str):

    """모델과 토크나이저 로드"""
    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer

def load_embedding_model(model_id: str, device: str = 'cuda'):
    """임베딩 모델 로드"""
    return BGEM3FlagModel(model_id, device=device)