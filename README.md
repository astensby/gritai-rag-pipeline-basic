# GritAI - Simple RAG Pipelines

## Overview

This project provides a collection of example RAG (Retrieval Augmented Generation) pipelines, demonstrating various configurations and complexities used as part of our GritAI RAG & MCP Training Courses. It serves as a learning resource for understanding and implementing RAG systems, from basic in-memory setups to more advanced versions using local vector databases (pgvector) and cloud-based embedding services (OpenAI, AWS Bedrock).


## Coding with the help of AI in Cursor

The majority of code is this project was built through prompting with Cursor. 

Be specific. Break things into steps. And iterate from there. 
The core RAG pipeline was initially built with a fairly simple prompt, included here for inspiration:

    ```text    
    Create a simple local RAG pipeline in Python with clearly defined functions for each step so that the code is easy to read and follow and that we can later replace the approach in each step with more sophisticated approaches.

    Use the following naive RAG structure:
    1. Ingestion step: Read files in a dedicated folder inside of our project - this can then later be replaced with for instance a AWS S3 bucket
    2. Chunking step: For each document ingested, we need to chunk the document. Please reccommend a simple approach to chunking to get us started.
    3. Embedding step: We need to create embeddings (for simplicity, let us use OpenAI embeddings for this step)
    4. Storage step: Let us for simplicity use an in memory vector db (for instance persisting it to a .json file or similar. This will later be replaced by a proper vector database.
    5. Retrieval step: Let us also create a retrieval function for testing using a basic cos similarity measure between a user query and the vector storage. We should also include a re-ranking function that can be empty for now, but that we can later implement.
    6. Evaluation step: Add a simple function for evaluation

    We should be able to run steps 1-4 as one function (for ingesting and creating our knowledge base). And then subject to an established knowledge base, we should be able to run and test steps 5 and 6 separately.
    ```

## Features

The repository includes the following example pipelines:

1.  **`rag_pipeline.py` (Core RAG Pipeline):**
    *   Reads `.txt` files from the `data/` directory.
    *   Implements a simple character-based chunking strategy with configurable overlap.
    *   Generates embeddings using the OpenAI API (requires `OPENAI_API_KEY`).
    *   Stores chunks and their embeddings in an in-memory vector database, persisted to a JSON file (`vector_db/vector_db.json`).
    *   Performs brute-force retrieval using cosine similarity.
    *   Includes a basic query answering and evaluation mechanism.

2.  **`examples/01_extended_pipeline/rag_pipeline_advanced.py` (Extended Pipeline):**
    *   Extends the core pipeline with support for reading `.pdf` files (using PyMuPDF/fitz).
    *   Supports embedding generation via both OpenAI API and AWS Bedrock (e.g., Titan models).
    *   Allows selection of the embedding provider.
    *   Persists the vector database to a provider-specific JSON file (e.g., `vector_db/vector_db_openai.json`, `vector_db/vector_db_aws.json`).

3.  **`examples/02_aws_complete_pipeline/rag_pipeline_advanced_aws.py` (AWS-focused Pipeline):**
    *   A streamlined version focused exclusively on using AWS Bedrock for embeddings.
    *   Reads `.txt` and `.pdf` files.
    *   Stores data in a JSON-based vector database (`vector_db/vector_db_aws.json`).

4.  **`examples/03_local_pipeline/` (Local VectorDB - PostgreSQL with pgvector):**
    *   **`rag_pipeline_localdb_complete.py`:**
        *   Integrates with a PostgreSQL database using the `pgvector` extension for efficient vector storage and similarity search (ANN).
        *   Uses AWS Bedrock for generating embeddings.
        *   Handles schema creation, data ingestion, chunking, embedding, and storage in PostgreSQL.
        *   Retrieves relevant chunks using cosine similarity directly in the database.
    *   **`rag_retrieval_only_localdb.py`:**
        *   A retrieval-only example demonstrating how to query an existing pgvector database populated by `rag_pipeline_localdb_complete.py`.
        *   Useful for scenarios where the knowledge base is pre-built and you only need to perform lookups (e.g., in an MCP server).
        *   Uses AWS Bedrock to embed the query before searching the database.

5.  **`examples/04_local_ollama/rag_pipeline_ollama.py` (Local Ollama Pipeline):**
    *   Runs 100% locally with no API keys or cloud dependencies.
    *   Uses Ollama for generating embeddings with the `embeddinggemma` model.
    *   Reads `.txt` files from the `data/` directory.
    *   Stores data in a JSON-based vector database (`vector_db/vector_db_ollama.json`).
    *   Perfect for privacy-sensitive applications or offline development.

6.  **`examples/05_openai_vector_store/` (OpenAI Vector Store Pipeline):**
    *   **`rag_pipeline_openai_vectorstore.py`:**
        *   Uses OpenAI's managed Vector Store API with the Assistants API.
        *   Automatically uploads documents where OpenAI handles chunking and embedding.
        *   Creates an Assistant configured with the `file_search` tool.
        *   Stores configuration (vector store ID, assistant ID) for later retrieval.
        *   Supports `.txt`, `.pdf`, `.md`, `.doc`, and `.docx` files.
    *   **`rag_retrieval_openai_vectorstore.py`:**
        *   Queries the pre-built OpenAI vector store using the Assistant.
        *   Supports interactive mode, batch queries, and command-line queries.
        *   Returns responses with automatic source citations from the knowledge base.
        *   Ideal for production deployments leveraging OpenAI's managed infrastructure.

## Directory Structure

```
.
├── data/                     # Sample .txt (and .pdf) files for ingestion
├── examples/
│   ├── 01_extended_pipeline/ # Advanced RAG with OpenAI & AWS
│   ├── 02_aws_complete_pipeline/ # AWS Bedrock focused RAG
│   ├── 03_local_pipeline/    # RAG with PostgreSQL/pgvector
│   ├── 04_local_ollama/      # Local RAG with Ollama embeddings
│   └── 05_openai_vector_store/ # RAG with OpenAI Vector Store
├── vector_db/                # Default location for JSON-based vector stores
├── .gitignore
├── rag_pipeline.py           # Core RAG pipeline script
├── README.md                 # This file
└── requirements.txt          # Python package dependencies
```

## Prerequisites

*   **Python:** Version 3.8 or higher is recommended.
*   **pip:** Python package installer.
*   **OpenAI API Key:** Required for examples using OpenAI embeddings. Set the `OPENAI_API_KEY` environment variable.
*   **AWS Credentials:** Required for examples using AWS Bedrock. Configure your AWS credentials (e.g., via `~/.aws/credentials`, environment variables, or IAM roles) so that `boto3` can access them. The scripts also look for an optional `AWS_PROFILE_NAME` and `AWS_REGION` environment variable or use defaults. 

boto3 works well with a simple SSO workflow.
You will need to create an SSO profile, and make sure you are logged in with
aws sso login --profile <your-profile-name>

aws configure sso is a one‑time wizard; afterwards only aws sso login is needed to renew the token.

*   **PostgreSQL with pgvector:** Required for the `03_local_pipeline` examples.
    *   Install PostgreSQL (e.g., version 13+).
    *   Install the [pgvector extension](https://github.com/pgvector/pgvector).
    *   Ensure your PostgreSQL server is running.

*   **Ollama:** Required for the `04_local_ollama` example.
    *   Install Ollama from [https://ollama.com/download](https://ollama.com/download).
    *   Start the Ollama service (typically runs automatically on macOS after installation, or run `ollama serve`).
    *   Pull the embedding model: `ollama pull nomic-embed-text`

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-name>
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Data to ingest
There are a few very simple files in the data folder. You can add .txt files and .pdf files to this folder. You can also use AI-coding assistants like Cursor or Claude to help you generate more data for testing! Simplay prompt the agent to create X additional .txt files in the data directory. You can also ask it to create significantly larger files for testing.

## Configuration

### 1. OpenAI API Key
Set the `OPENAI_API_KEY` environment variable:
```bash
export OPENAI_API_KEY="your_openai_api_key_here"
```
You might want to add this to your shell's configuration file (e.g., `.bashrc`, `.zshrc`) for persistence. 

Or use a .env file in the root of this project.

### 2. AWS Credentials & Region
Ensure your AWS credentials are configured for `boto3`. You can also set the following environment variables if needed by specific scripts (though they often have defaults or allow profile names in code):
```bash
export AWS_REGION="your-aws-region" # e.g., us-west-1
# If using a specific AWS profile (some scripts might look for AWS_PROFILE_NAME or have it hardcoded as "my-mcp-server-profile"):
# export AWS_PROFILE_NAME="your-profile-name"
```

### 3. PostgreSQL Connection (for `03_local_pipeline`)
The scripts in `examples/03_local_pipeline/` use the following environment variables to connect to your PostgreSQL database. Set them according to your local setup:
```bash
export PGVECTOR_DB_NAME="vectordb"         # Your database name
export PGVECTOR_DB_USER="your_username"    # Your PostgreSQL username
export PGVECTOR_DB_PASSWORD="your_password"  # Your PostgreSQL password
export PGVECTOR_DB_HOST="localhost"        # Database host
export PGVECTOR_DB_PORT="5432"             # Database port
```
The `rag_pipeline_localdb_complete.py` script will attempt to create the necessary schema (`embeddings`) and table (`documents`) if they don't exist, and also tries to enable the `vector` extension.

## Running the Examples

Ensure you have configured the necessary API keys and environment variables as described above.

### 1. Core RAG Pipeline (`rag_pipeline.py`)
This script will:
1.  Ingest documents from the `data/` directory.
2.  Chunk them.
3.  Create embeddings using OpenAI.
4.  Store them in `vector_db/vector_db.json`.
5.  Then, it enters a loop to answer your queries based on the built knowledge base.

```bash
python rag_pipeline.py
```
You can place your `.txt` files in the `data/` directory before running.

### 2. Extended Pipeline (`examples/01_extended_pipeline/rag_pipeline_advanced.py`)
This script can use either OpenAI or AWS for embeddings. It will also process `.pdf` files in the `data/` directory.
```bash
# The script defaults to OpenAI if available, then AWS.
# It will create a vector_db_openai.json or vector_db_aws.json
python examples/01_extended_pipeline/rag_pipeline_advanced.py
```
Check the script's `DEFAULT_EMBEDDING_PROVIDER` and how to potentially modify it if needed (e.g., by changing the variable or adapting the script to take a command-line argument).

### 3. AWS-focused Pipeline (`examples/02_aws_complete_pipeline/rag_pipeline_advanced_aws.py`)
This script uses AWS Bedrock for embeddings.
```bash
python examples/02_aws_complete_pipeline/rag_pipeline_advanced_aws.py
```
It will create `vector_db/vector_db_aws.json`.

### 4. Local VectorDB Pipeline (`examples/03_local_pipeline/`)

**a) Building the Knowledge Base (`rag_pipeline_localdb_complete.py`):**
First, run this script to ingest documents, create embeddings (using AWS Bedrock), and store them in your PostgreSQL/pgvector database.
```bash
# Ensure PostgreSQL is running and environment variables are set
python examples/03_local_pipeline/rag_pipeline_localdb_complete.py
```

**b) Querying the Knowledge Base (`rag_retrieval_only_localdb.py`):**
After populating the database, you can use this script to ask questions. It will embed your query using AWS Bedrock and retrieve relevant chunks from PostgreSQL.
```bash
python examples/03_local_pipeline/rag_retrieval_only_localdb.py
```

### 5. Local Ollama Pipeline (`examples/04_local_ollama/rag_pipeline_ollama.py`)
This script runs 100% locally with no API keys required. It uses Ollama for embeddings.
```bash
# Ensure Ollama is installed and running, and the model is pulled
ollama pull nomic-embed-text
python examples/04_local_ollama/rag_pipeline_ollama.py
```
It will create `vector_db/vector_db_ollama.json`.

### 6. OpenAI Vector Store Pipeline (`examples/05_openai_vector_store/`)

**a) Building the Knowledge Base (`rag_pipeline_openai_vectorstore.py`):**
This script uploads your documents to OpenAI's Vector Store, where OpenAI automatically handles chunking and embedding. It also creates an Assistant configured with file search capabilities.

```bash
# Ensure OPENAI_API_KEY is set
python examples/05_openai_vector_store/rag_pipeline_openai_vectorstore.py
```

The script will create a configuration file at `examples/05_openai_vector_store/vectorstore_config.json` containing your vector store ID and assistant ID.

**Useful options:**
- `--rebuild`: Force rebuild even if configuration exists
- `--cleanup`: Delete the vector store and assistant (WARNING: permanent deletion!)

```bash
# Rebuild the knowledge base
python examples/05_openai_vector_store/rag_pipeline_openai_vectorstore.py --rebuild

# Clean up resources (will prompt for confirmation)
python examples/05_openai_vector_store/rag_pipeline_openai_vectorstore.py --cleanup
```

**b) Querying the Knowledge Base (`rag_retrieval_openai_vectorstore.py`):**
After building the knowledge base, use this script to query it. The Assistant will search the vector store and provide answers with source citations.

```bash
# Run with default test queries
python examples/05_openai_vector_store/rag_retrieval_openai_vectorstore.py

# Interactive mode - ask multiple questions
python examples/05_openai_vector_store/rag_retrieval_openai_vectorstore.py --interactive

# Command-line query
python examples/05_openai_vector_store/rag_retrieval_openai_vectorstore.py "What is machine learning?"

# Multiple queries at once
python examples/05_openai_vector_store/rag_retrieval_openai_vectorstore.py "Query 1" "Query 2" "Query 3"
```

**Key Features:**
- Fully managed by OpenAI (no manual chunking/embedding management)
- Automatic source citations in responses
- Supports multiple document formats (.txt, .pdf, .md, .doc, .docx)
- Interactive query mode for exploration
- Ideal for production use with OpenAI's infrastructure

## Data
Place your `.txt` and `.pdf` (for relevant pipelines) files into the `data/` directory. The ingestion scripts will pick them up from there.