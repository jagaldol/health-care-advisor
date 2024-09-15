from FlagEmbedding import BGEM3FlagModel
from pinecone import Pinecone

from langchain_core.prompts import PromptTemplate

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, TextStreamer
from langchain_huggingface import HuggingFacePipeline

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
import os

from retriever import HybridSearchRetriever

load_dotenv()



template = """
Instructions:
- If the question involves a health-related issue, suggest possible causes and basic steps the user can take for relief, if applicable.
- You should explain in as much detail as possible what you know from the bottom of your heart to the user's questions.
- You can refer to the contents of the documents to create a response.
- Only use information that is directly related to the question.
- If no information is found in the documents, provide an answer based on general knowledge without fabricating details.
- You MUST answer in Korean.


Documents: {documents}

Question: {question}
"""

# Load Models
model_id="google/gemma-2-2b-it"
gemma_2_model = AutoModelForCausalLM.from_pretrained(model_id)
gemma_2_tokenizer = AutoTokenizer.from_pretrained(model_id)

embedding_model = BGEM3FlagModel('BAAI/bge-m3',use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation

# Connect to the existing Pinecone index
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pc.Index("health-care")
    
retriever = HybridSearchRetriever(pinecone_index=pinecone_index,
                                  embedding_model=embedding_model,
                                  alpha=0.95,
                                  top_k=3,
                                  min_score=0.55,
                                  )

streamer = TextStreamer(gemma_2_tokenizer, skip_prompt=True, skip_special_tokens=True)

gen = pipeline(task='text-generation',
               model=gemma_2_model,
               tokenizer=gemma_2_tokenizer,
               max_new_tokens=1024,
               streamer=streamer,
               device=0 if torch.cuda.is_available() else -1,
               temperature=.5,
               top_p=0.7,
               repetition_penalty=1.1,
               do_sample=True,
               )

llm = HuggingFacePipeline(pipeline=gen)

prev_chat = []

chat = [*prev_chat,
        { "role": "user", "content": template}
        ]

prompt_template = gemma_2_tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

prompt = PromptTemplate(input_variables=["question", "context"], template=prompt_template)

rag_chain = ({"documents": retriever, "question": RunnablePassthrough()}
             | prompt
             | llm
             | StrOutputParser()
             )

answer = rag_chain.invoke("머리 아프다")
