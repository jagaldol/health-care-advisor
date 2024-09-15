# health-care-advisor

The chatbot based on the Gemma 2 model, delivering professional healthcare information to patients quickly and accurately.

## How to Start

Follow the steps below to set up your development environment and get started with the project:

### Set Up Environment Variables

Copy the .env.example file to .env and update the environment variables as needed:

bash

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

Once the environment is set up, you can start the application using the appropriate command, such as:

```sh
$ python main.py
```
