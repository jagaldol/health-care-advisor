# health-care-advisor

[![banner](/docs/banner.png)](https://undoc.vercel.app)

The chatbot based on the Gemma 2 model, delivering professional healthcare information to patients quickly and accurately.

## Demo

[![demo](/docs/demo.png)](https://undoc.vercel.app/)

- Demo site: https://undoc.vercel.app
- Demo view Repo: https://github.com/jagaldol/undoc

> You can view our demo page on [this site](https://undoc.vercel.app/).
>
> The repository for the Demo page implementation can be found [here](https://github.com/jagaldol/undoc).
> you can see our demo page in this site

## Model

We use the RAFT-finetuned `Gemma-2-2b-it` model with RAG for healthcare datasets.

> [Huggingface Model Card](https://huggingface.co/devlim/Korea-HealthCare-RAFT-float16)

### Model Performance Comparison

![performance](/docs/performance.jpg)

| Model             |   Median |
| :---------------- | -------: |
| Gemma-2-2b-it     |     0.80 |
| Gemma-2-2b-it+RAG |     0.89 |
| DSF unit8         |     0.88 |
| DSF unit8 + RAG   |     0.93 |
| DSF               |     0.90 |
| DSF + RAG         |     0.94 |
| RAFT unit8        |     0.86 |
| RAFT unit8 + RAG  |     0.93 |
| RAFT              |     0.88 |
| **RAFT + RAG**    | **0.96** |

**We achieved a 16% performance increase from the base model!**

> we use [RAGAS Semantic Similarity](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/semantic_similarity/) to measure performance.

## How to Start

Follow the steps below to set up your development environment and get started with the project:

### Set Up Environment Variables

Copy the .env.example file to .env and update the environment variables as needed:

```sh
$ cp .env.example .env
```

Open the .env file and update the variables with your specific configuration:

```text
HF_TOKEN=your huggingface token
PINECONE_API_KEY=your pinecone api key
PINECONE_INDEX=your pinecone index name
```

### Install Dependencies with Pipenv

Use pipenv to install the required dependencies and set up the virtual environment:

```sh
$ pipenv sync
```

This command will create a virtual environment if one does not already exist and install all dependencies as specified in the Pipfile.

### Activate the Virtual Environment

To activate the virtual environment, run:

```sh
$ pipenv shell
```

### Run the Application

After setting up the environment, you can start the application with the following command:

```sh
$ python main.py
```

Once the server is running, you can access the API at http://localhost:8000.

## Inference

If you want a single response, run the inference.py script with a query to get a direct answer from the model:

```sh
$ python inference.py "배탈 난거 같은데 어떻게 해?"
```

This script takes a user query as input and generates a relevant response based on the AI model.

## Sturctures

```text
health-care-advisor
├── main.py
├── models
│   └── model_loader.py
├── notebooks
├── ragchain
│   ├── pipeline_setup.py
│   └── rag_chain.py
├── retriever
│   ├── hybrid_search_retriever.py
│   └── retriever_setup.py
├── template
│   ├── generation_prompt.py
│   └── script.py
└── utils
    └── environment.py
```

## System Structure

![structure](/docs/structure.jpg)

## Citation

- 초거대 AI 헬스케어 질의응답 데이터: AI 허브, 초거대 AI 헬스케어 질의응답 데이터
- Gemma 2 모델: "Gemma 2: Improving Open Language Models at a Practical Size", 2023.
- RAFT 방법론: "Adapting Language Model to Domain Specific RAG", arXiv preprint arXiv:2403.10131, 2023.
- RAGAS 평가 방법론: "RAGAS: Automated Evaluation of Retrieval Augmented Generation", 2023.

## Collaborators

|          [임영윤](https://github.com/Lim-YoungYoon)          |          [안혜준](https://github.com/jagaldol)          |
| :----------------------------------------------------------: | :-----------------------------------------------------: |
| <img src="https://github.com/Lim-YoungYoon.png" width="100"> | <img src="https://github.com/jagaldol.png" width="100"> |
