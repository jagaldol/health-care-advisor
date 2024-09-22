from transformers import pipeline
from .huggingface_pipeline import HuggingFacePipeline
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

def setup_text_generation_pipeline(model: PreTrainedModel, 
                                   tokenizer: PreTrainedTokenizer, 
                                   temperature: float = 0.5, 
                                   top_p: float = 0.7, 
                                   repetition_penalty: float = 1.1) -> HuggingFacePipeline:
    """
    텍스트 생성 파이프라인을 설정하고 HuggingFacePipeline 객체를 반환합니다.

    Args:
        model (PreTrainedModel): Hugging Face에서 제공하는 사전 학습된 텍스트 생성 모델.
        tokenizer (PreTrainedTokenizer): 모델과 함께 사용하는 토크나이저로, 입력 텍스트를 토큰화하고 
            모델에 맞게 변환하는 역할을 합니다.
        temperature (float, optional): 샘플링 시 생성된 텍스트의 다양성을 조절하는 매개변수입니다. 
            낮은 값은 더 예측 가능한 텍스트를, 높은 값은 더 창의적인 텍스트를 생성합니다. 기본값은 0.5입니다.
        top_p (float, optional): Nucleus 샘플링을 위한 매개변수로, 상위 p 백분위수에 속하는 텍스트 
            토큰을 선택합니다. 기본값은 0.7입니다.
        repetition_penalty (float, optional): 텍스트 생성 시 반복을 줄이기 위한 패널티를 적용합니다. 
            기본값은 1.1입니다.

    Returns:
        Pipeline: 텍스트 생성 모델과 설정된 옵션을 기반으로 하는 HuggingFacePipeline 객체를 반환합니다. 
        이를 통해 텍스트 생성이 수행됩니다.
    """
    return HuggingFacePipeline(pipeline= 
                               pipeline(task='text-generation',
                                        model=model,
                                        tokenizer=tokenizer,
                                        max_new_tokens=1024,
                                        temperature=temperature,
                                        top_p=top_p,
                                        repetition_penalty=repetition_penalty,
                                        do_sample=True,
                                        )
                               )