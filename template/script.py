from typing import List, Dict
from transformers import PreTrainedTokenizer

from langchain_core.prompts import PromptTemplate
from . import generation_prompt

def create_chat_prompt_template(prev_chat: List[Dict], tokenizer: PreTrainedTokenizer):
    """
    이전 채팅 기록과 생성 프롬프트를 결합하여 새로운 프롬프트 템플릿을 생성합니다.

    Args:
        prev_chat (List[Dict]): 이전 채팅 기록의 리스트입니다. 각 기록은 딕셔너리 형태로,
            예를 들어 {"role": "user", "content": "질문 내용"}과 같은 구조를 가집니다.
        tokenizer (PreTrainedTokenizer): Hugging Face에서 제공하는 Tokenizer로,
            채팅 템플릿에 적용될 수 있는 토크나이저입니다. 주로 GPT 계열 모델의 토크나이저를 사용합니다.

    Returns:
        PromptTemplate: 'question'과 'documents'를 포함하는 새로운 프롬프트 템플릿을 반환합니다.
    """
    prev_chat = []

    chat = [*prev_chat,
            { "role": "user", "content": generation_prompt.template}
            ]
    
    prompt_template = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    prompt = PromptTemplate(input_variables=["question", "documents"], template=prompt_template)

    return prompt