# rag-agent
RAG Agent for answering questions from an input PDF

# Setup
The required setup for this code is as follows:

1. Install the required dependencies by running:
```bash
pip install -r requirements.txt
```
This code is tested on Python3.10

2. To use OpenAI API, you need to set up your OpenAI API key in the environment variable `OPENAI_API_KEY`, either as an environment variable or in a `.env` file.
You may optionally set the `SLACK_ACCESS_TOKEN` variable and configure the Slack channel in `main.py` to send the responses to Slack (not fully tested).

3. Optionally, you may also install Redis by following the installation steps [here](https://redis.io/docs/latest/operate/oss_and_stack/install/install-redis/install-redis-on-linux/)

Verify redis service is running by
```bash
$ redis-cli ping
PONG
```

# Usage
To run the code, simply execute `main.py` file by running
```bash
python main.py
```
This will run the agent on the input PDF file `sample/handbook.pdf` with the questions defined in `sample/input_questions.txt` and generate a JSON file `output/output_{timestamp}.json` with a JSON blob of Question and Answer pairs.

# Workflow
The RAG Agent works using the following flow:
1. Reads the PDF file and converts it to Markdown
2. Chunks the PDF Markdown text into smaller chunks based on a recursive splitter
3. For each chunk (and the query), calculate the embedding vectors, either using OpenAI Embedding models, or a custom Pytorch based model
4. Calculate the cosine similarity of the query embedding vector with the chunk embedding vector
5. Accumulate the top maching text chunks based on the similarity scores
6. Optionally, rerank the text chunks again with a reranker model
7. Call the OpenAI API to generate the response