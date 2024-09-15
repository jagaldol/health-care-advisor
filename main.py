from typing import List
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever

from FlagEmbedding import BGEM3FlagModel
from pinecone import Pinecone, Index

from langchain_core.prompts import PromptTemplate

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, TextStreamer
from langchain_huggingface import HuggingFacePipeline

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
import os

load_dotenv()

class HybridSearchRetriever(BaseRetriever):

    pinecone_index:Index
    embedding_model:BGEM3FlagModel
    alpha:float
    top_k: int
    min_score:float

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[str]:
        """Sync implementations for retriever."""
        user_query_emb = self.embedding_model.encode(query, return_dense=True, return_sparse=True, return_colbert_vecs=False) #dense, sparse 둘 다 반환함
        
        query_dense_vector = user_query_emb['dense_vecs'].tolist()
        user_query_sparse = user_query_emb['lexical_weights']
        query_sparse_vector = {
            'indices': [int(k) for k in user_query_sparse.keys() if k.isdigit()], #isdigit() 안하면 에러뜨더라
            'values': [float(v) for k, v in user_query_sparse.items() if k.isdigit()]
        }

        hdense, hsparse = self._hybrid_score_norm(query_dense_vector, query_sparse_vector, alpha=self.alpha)

        hybrid_query_response = self.pinecone_index.query(
            top_k=self.top_k,
            vector=hdense,
            sparse_vector=hsparse,
            include_metadata=True,
        )
        
        documents = [
            f"{match['metadata']['answer_intro']}\n"
            f"{match['metadata']['answer_body']}\n"
            f"{match['metadata']['answer_conclusion']}"
            for match in hybrid_query_response['matches']
            if match['score'] >= self.min_score
        ]
        return documents
    
    def _hybrid_score_norm(self, dense, sparse, alpha: float):
        """Hybrid score using a convex combination

        alpha * dense + (1 - alpha) * sparse

        Args:
            dense: Array of floats representing
            sparse: a dict of `indices` and `values`
            alpha: scale between 0 and 1
        """
        if alpha < 0 or alpha > 1:
            raise ValueError("Alpha must be between 0 and 1")
        hs = {
            'indices': sparse['indices'],
            'values':  [v * (1 - alpha) for v in sparse['values']]
        }
        return [v * alpha for v in dense], hs

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
