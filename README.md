# health-care-advisor

The chatbot based on the Gemma 2 model, delivering professional healthcare information to patients quickly and accurately.

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

## Collaborators

|          [임영윤](https://github.com/Lim-YoungYoon)          |          [안혜준](https://github.com/jagaldol)          |
| :----------------------------------------------------------: | :-----------------------------------------------------: |
| <img src="https://github.com/Lim-YoungYoon.png" width="100"> | <img src="https://github.com/jagaldol.png" width="100"> |
