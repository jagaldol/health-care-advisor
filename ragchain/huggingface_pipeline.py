from __future__ import annotations  # type: ignore[import-not-found]

from typing import Any, Iterator, List, Mapping, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from pydantic import ConfigDict

DEFAULT_MODEL_ID = "gpt2"
DEFAULT_TASK = "text-generation"
VALID_TASKS = (
    "text2text-generation",
    "text-generation",
    "summarization",
    "translation",
)
DEFAULT_BATCH_SIZE = 4



class HuggingFacePipeline(BaseLLM):
    pipeline: Any = None  #: :meta private:
    model_id: str = DEFAULT_MODEL_ID
    """Model name to use."""
    model_kwargs: Optional[dict] = None
    """Keyword arguments passed to the model."""
    pipeline_kwargs: Optional[dict] = None
    """Keyword arguments passed to the pipeline."""
    batch_size: int = DEFAULT_BATCH_SIZE
    """Batch size to use when passing multiple documents to generate."""

    model_config = ConfigDict(
        extra="forbid",
    )

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_id": self.model_id,
            "model_kwargs": self.model_kwargs,
            "pipeline_kwargs": self.pipeline_kwargs,
        }

    @property
    def _llm_type(self) -> str:
        return "huggingface_pipeline"

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        # List to hold all results
        text_generations: List[str] = []
        pipeline_kwargs = kwargs.get("pipeline_kwargs", {})
        skip_prompt = kwargs.get("skip_prompt", False)

        for i in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[i : i + self.batch_size]

            # Process batch of prompts
            responses = self.pipeline(
                batch_prompts,
                **pipeline_kwargs,
            )

            # Process each response in the batch
            for j, response in enumerate(responses):
                if isinstance(response, list):
                    # if model returns multiple generations, pick the top one
                    response = response[0]

                if self.pipeline.task == "text-generation":
                    text = response["generated_text"]
                elif self.pipeline.task == "text2text-generation":
                    text = response["generated_text"]
                elif self.pipeline.task == "summarization":
                    text = response["summary_text"]
                elif self.pipeline.task in "translation":
                    text = response["translation_text"]
                else:
                    raise ValueError(
                        f"Got invalid task {self.pipeline.task}, "
                        f"currently only {VALID_TASKS} are supported"
                    )
                if skip_prompt:
                    text = text[len(batch_prompts[j]) :]
                # Append the processed text to results
                text_generations.append(text)

        return LLMResult(
            generations=[[Generation(text=text)] for text in text_generations]
        )

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        from threading import Thread

        import torch
        from transformers import (
            StoppingCriteria,
            StoppingCriteriaList,
            TextIteratorStreamer,
        )

        pipeline_kwargs = kwargs.get("pipeline_kwargs", {})
        skip_prompt = kwargs.get("skip_prompt", True)

        if stop is not None:
            stop = self.pipeline.tokenizer.convert_tokens_to_ids(stop)
        stopping_ids_list = stop or []

        class StopOnTokens(StoppingCriteria):
            def __call__(
                self,
                input_ids: torch.LongTensor,
                scores: torch.FloatTensor,
                **kwargs: Any,
            ) -> bool:
                for stop_id in stopping_ids_list:
                    if input_ids[0][-1] == stop_id:
                        return True
                return False

        stopping_criteria = StoppingCriteriaList([StopOnTokens()])

        streamer = TextIteratorStreamer(
            self.pipeline.tokenizer,
            timeout=60.0,
            skip_prompt=skip_prompt,
            skip_special_tokens=True,
        )
        generation_kwargs = dict(
            text_inputs= prompt,
            streamer=streamer,
            stopping_criteria=stopping_criteria,
            **pipeline_kwargs,
        )
        t1 = Thread(target=self.pipeline, kwargs=generation_kwargs)
        t1.start()

        for char in streamer:
            chunk = GenerationChunk(text=char)
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)

            yield chunk