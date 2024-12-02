# Project Overview

The project aims to use **LLM (Large Language Models)** and **OCR (Optical Character Recognition)** to create a **Question Answering (QA) system** over a document.

## Files

There are 4 main files in the project:

1. **ingestion.py**: Responsible for performing OCR, processing, and creating a vector store.
2. **generate/chatbot.py**: Implements the **RAG (Retrieval-Augmented Generation)** functionality.
3. **main.py**: Implements the user interface using **Streamlit**.
4. **.env**: Please load your environment variables here (e.g., API keys).

## Deployment

While this project is not currently deployed, it can be deployed by:

1. Uploading the project to **GitHub**.
2. Connecting it to **Streamlit Cloud** for deployment.

## Environment Variables

Please make sure to include the following API keys in the `.env` file:

- **OPENAI API KEY**
- **PINECONE API KEY**

## Requirements

All the necessary dependencies are listed in the `requirements.txt` file. You can install them using:

```bash
pip install -r requirements.txt
```
To run the code do the following:
'''bash
streamlit run main.py
```
Note : The page will be slow to read, as the processing and indexing will be done before that.
