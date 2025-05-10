import os
import json
import numpy as np
from openai import OpenAI
import logging
from typing import List, Dict, Any, Literal
import fitz # PyMuPDF
import boto3 # Added for AWS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Ensure OPENAI_API_KEY environment variable is set
# Or configure AWS credentials (e.g., via environment variables or IAM role)
try:
    openai_client = OpenAI() # Reads OPENAI_API_KEY from environment variable
    openai_available = True
except Exception as e:
    logging.warning(f"Failed to initialize OpenAI client. OpenAI embeddings unavailable. Error: {e}")
    openai_client = None
    openai_available = False

# AWS Bedrock Configuration
AWS_REGION_NAME = os.environ.get("AWS_REGION", "eu-west-1") # Or your preferred region
AWS_EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
AWS_PROFILE_NAME = "my-mcp-server-profile" # Set the profile name directly

try:
    # Create a session with the specific profile
    logging.info(f"Using AWS Profile: {AWS_PROFILE_NAME}")
    session = boto3.Session(profile_name=AWS_PROFILE_NAME, region_name=AWS_REGION_NAME)

    # Use the session to create the Bedrock client
    bedrock_runtime = session.client(service_name='bedrock-runtime')
    aws_available = True
    logging.info(f"Successfully initialized AWS Bedrock runtime client in region '{AWS_REGION_NAME}' using profile '{AWS_PROFILE_NAME}'.")
except Exception as e:
    logging.warning(f"Failed to initialize AWS Bedrock runtime client using profile '{AWS_PROFILE_NAME}'. AWS embeddings unavailable. Error: {e}")
    bedrock_runtime = None
    aws_available = False


DEFAULT_EMBEDDING_PROVIDER: Literal['openai', 'aws'] = 'openai' if openai_available else ('aws' if aws_available else 'none')
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
VECTOR_DB_PATH = "vector_db/vector_db_{provider}.json" # Provider-specific DB
DATA_DIR = "data"
MAX_OPENAI_EMBEDDING_BATCH_SIZE = 2000 # Keep slightly below OpenAI's limit of 2048 for safety
# Note: AWS Bedrock Titan V2 currently has a limit of 1 input text per request. Batching needs sequential calls.
MAX_AWS_EMBEDDING_BATCH_SIZE = 1 # For Titan V2, process one by one in a loop


# --- 1. Ingestion Step ---
def ingest_documents(data_dir: str) -> List[Dict[str, str]]:
    """
    Reads text files from a specified directory.

    Args:
        data_dir: The path to the directory containing the text files.

    Returns:
        A list of dictionaries, where each dictionary represents a document
        with 'filename' and 'content'. Returns empty list if dir not found.
    """
    documents = []
    if not os.path.isdir(data_dir):
        logging.warning(f"Data directory '{data_dir}' not found.")
        return documents
    try:
        for filename in os.listdir(data_dir):
            if filename.endswith(".txt"): # Simple filter for text files
                file_path = os.path.join(data_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        documents.append({"filename": filename, "content": content})
                        logging.info(f"Successfully ingested '{filename}'.")
                except Exception as e:
                    logging.error(f"Failed to read or decode '{filename}': {e}")
            elif filename.endswith(".pdf"):
                file_path = os.path.join(data_dir, filename)
                try:
                    doc = fitz.open(file_path)
                    content = ""
                    for page_num in range(len(doc)):
                        page = doc.load_page(page_num)
                        content += page.get_text()
                    doc.close()
                    if content.strip(): # Ensure content was extracted
                        documents.append({"filename": filename, "content": content})
                        logging.info(f"Successfully ingested and extracted text from PDF '{filename}'.")
                    else:
                        logging.warning(f"Extracted empty content from PDF '{filename}'.")
                except Exception as e:
                    logging.error(f"Failed to read or process PDF '{filename}': {e}")
            # Add more file types (e.g., .docx) handling here later
    except Exception as e:
        logging.error(f"Error listing directory '{data_dir}': {e}")
    return documents

# --- 2. Chunking Step ---
def chunk_documents(documents: List[Dict[str, str]], chunk_size: int = 1000, overlap: int = 100) -> List[Dict[str, Any]]:
    """
    Splits documents into smaller chunks using a fixed-size strategy with overlap.

    Args:
        documents: A list of document dictionaries (from ingest_documents).
        chunk_size: The desired character length of each chunk.
        overlap: The number of characters to overlap between consecutive chunks.

    Returns:
        A list of chunk dictionaries, each containing 'filename', 'chunk_id',
        and 'text'.
    """
    chunks = []
    chunk_id_counter = 0
    for doc in documents:
        content = doc['content']
        filename = doc['filename']
        start = 0
        doc_chunk_index = 0
        while start < len(content):
            end = start + chunk_size
            chunk_text = content[start:end]

            # Ensure we don't create empty chunks if content is smaller than chunk_size
            if not chunk_text.strip():
                start += chunk_size - overlap # Move to next potential chunk start
                continue

            chunks.append({
                "filename": filename,
                "chunk_id": f"{filename}_{doc_chunk_index}", # Unique ID per chunk
                "text": chunk_text
            })
            chunk_id_counter += 1
            doc_chunk_index += 1

            # Move the start position for the next chunk
            start += chunk_size - overlap

            # If the next chunk would start beyond the content length, break
            if start >= len(content):
                break

        logging.info(f"Chunked '{filename}' into {doc_chunk_index} chunks.")

    logging.info(f"Total chunks created: {len(chunks)}")
    return chunks

# --- 3. Embedding Step ---

def _create_embeddings_openai(chunks: List[Dict[str, Any]], model: str, batch_size: int) -> List[np.ndarray]:
    """Creates embeddings using OpenAI."""
    if not openai_client:
        logging.error("OpenAI client not initialized. Cannot create OpenAI embeddings.")
        return []
    if not chunks:
        logging.warning("No chunks provided to create_embeddings_openai.")
        return []

    all_embeddings = []
    total_chunks = len(chunks)
    logging.info(f"Creating OpenAI embeddings for {total_chunks} chunks using model '{model}' in batches of {batch_size}...")

    for i in range(0, total_chunks, batch_size):
        batch_chunks = chunks[i:min(i + batch_size, total_chunks)]
        texts_to_embed = [chunk['text'] for chunk in batch_chunks]

        try:
            logging.info(f"Processing OpenAI batch {i // batch_size + 1}/{(total_chunks + batch_size - 1) // batch_size} ({len(batch_chunks)} chunks)")
            response = openai_client.embeddings.create(input=texts_to_embed, model=model)
            batch_embeddings = [np.array(item.embedding) for item in response.data]
            all_embeddings.extend(batch_embeddings)
            # logging.info(f"Successfully created embeddings for OpenAI batch.") # Reduce verbosity

        except Exception as e:
            logging.error(f"OpenAI API call failed for batch starting at index {i}: {e}")
            logging.error("Aborting OpenAI embedding creation due to API error.")
            return [] # Stop on error

    # Default embedding dimension for OpenAI text-embedding-3-small = 1536

    logging.info(f"Finished creating {len(all_embeddings)} OpenAI embeddings.")
    return all_embeddings


def _create_embeddings_aws(chunks: List[Dict[str, Any]], model_id: str) -> List[np.ndarray]:
    """Creates embeddings using AWS Bedrock Titan V2."""
    if not bedrock_runtime:
        logging.error("AWS Bedrock runtime client not initialized. Cannot create AWS embeddings.")
        return []
    if not chunks:
        logging.warning("No chunks provided to create_embeddings_aws.")
        return []

    all_embeddings = []
    total_chunks = len(chunks)
    logging.info(f"Creating AWS embeddings for {total_chunks} chunks using model '{model_id}' (one by one)...")

    for i, chunk in enumerate(chunks):
        text_to_embed = chunk['text']
        # Ensure text is not empty, as Bedrock might error
        if not text_to_embed.strip():
            logging.warning(f"Skipping empty chunk {chunk.get('chunk_id', i)}. Adding zero vector.")
            # Default embedding dimension for Titan V2 (e.g., 1024)
            embedding_dim = 1024 
            all_embeddings.append(np.zeros(embedding_dim))
            continue

        try:
            if (i + 1) % 100 == 0 or i == total_chunks - 1: # Log progress periodically
                 logging.info(f"Processing AWS chunk {i + 1}/{total_chunks}")

            request_body = json.dumps({"inputText": text_to_embed})
            response = bedrock_runtime.invoke_model(
                body=request_body,
                modelId=model_id,
                accept='application/json',
                contentType='application/json'
            )
            response_body = json.loads(response.get('body').read())
            embedding = np.array(response_body.get('embedding'))
            all_embeddings.append(embedding)

        except Exception as e:
            logging.error(f"AWS Bedrock API call failed for chunk index {i} (ID: {chunk.get('chunk_id', 'N/A')}): {e}")
            # Decide on error handling: add zero vector, stop, etc. Let's stop for now.
            logging.error("Aborting AWS embedding creation due to API error.")
            return []

    logging.info(f"Finished creating {len(all_embeddings)} AWS embeddings.")
    return all_embeddings


def create_embeddings(
    chunks: List[Dict[str, Any]],
    provider: Literal['openai', 'aws'] = DEFAULT_EMBEDDING_PROVIDER,
    openai_model: str = DEFAULT_OPENAI_EMBEDDING_MODEL,
    aws_model_id: str = AWS_EMBEDDING_MODEL_ID,
    openai_batch_size: int = MAX_OPENAI_EMBEDDING_BATCH_SIZE
) -> List[np.ndarray]:
    """
    Creates embeddings for text chunks using the specified provider (OpenAI or AWS).

    Args:
        chunks: A list of chunk dictionaries (from chunk_documents).
        provider: The embedding service to use ('openai' or 'aws').
        openai_model: The OpenAI embedding model to use (if provider is 'openai').
        aws_model_id: The AWS Bedrock model ID to use (if provider is 'aws').
        openai_batch_size: The batch size for OpenAI API calls.

    Returns:
        A list of embeddings (as numpy arrays), corresponding to the input chunks.
        Returns empty list if the selected provider is not available or API call fails.
    """
    if provider == 'openai':
        if not openai_available:
            logging.error("OpenAI provider selected, but client is not available.")
            return []
        return _create_embeddings_openai(chunks, model=openai_model, batch_size=openai_batch_size)
    elif provider == 'aws':
        if not aws_available:
            logging.error("AWS provider selected, but client is not available.")
            return []
        # AWS Titan V2 currently requires sequential processing (batch size 1) via API
        return _create_embeddings_aws(chunks, model_id=aws_model_id)
    else:
        logging.error(f"Invalid or unavailable embedding provider specified: {provider}. Choose 'openai' or 'aws'.")
        return []


# --- 4. Storage Step ---
def store_embeddings(chunks: List[Dict[str, Any]], embeddings: List[np.ndarray], output_path: str):
    """
    Stores chunks and their corresponding embeddings in a JSON file.

    Args:
        chunks: The list of chunk dictionaries.
        embeddings: The list of corresponding embeddings (numpy arrays).
        output_path: The file path to save the data.
    """
    if len(chunks) != len(embeddings):
        logging.error("Mismatch between number of chunks and embeddings. Cannot store.")
        return

    vector_db = []
    for chunk, embedding in zip(chunks, embeddings):
        # Ensure embedding is JSON serializable (list of floats)
        vector_db.append({
            "chunk_id": chunk["chunk_id"],
            "filename": chunk["filename"],
            "text": chunk["text"],
            "embedding": embedding.tolist() # Convert numpy array to list
        })

    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(vector_db, f, indent=4)
        logging.info(f"Successfully stored {len(vector_db)} chunk embeddings in '{output_path}'.")
    except Exception as e:
        logging.error(f"Failed to save vector database to '{output_path}': {e}")

# --- Combined Ingestion Pipeline ---
def build_knowledge_base(data_dir: str = DATA_DIR, output_path: str = VECTOR_DB_PATH, provider: Literal['openai', 'aws'] = DEFAULT_EMBEDDING_PROVIDER):
    """
    Runs the complete ingestion pipeline: Ingest -> Chunk -> Embed -> Store.
    """
    logging.info("Starting knowledge base build process...")
    documents = ingest_documents(data_dir)
    if not documents:
        logging.warning("No documents ingested. Stopping knowledge base build.")
        return

    chunks = chunk_documents(documents)
    if not chunks:
        logging.warning("No chunks created. Stopping knowledge base build.")
        return

    embeddings = create_embeddings(chunks, provider=provider)
    if not embeddings or len(embeddings) != len(chunks):
        logging.error("Embedding creation failed or resulted in mismatch. Stopping knowledge base build.")
        return

    store_embeddings(chunks, embeddings, output_path.format(provider=provider))
    logging.info("Knowledge base build process completed.")

# --- 5. Retrieval Step ---
def load_vector_db(db_path: str) -> List[Dict[str, Any]]:
    """Loads the vector database from the JSON file."""
    if not os.path.exists(db_path):
        logging.error(f"Vector database file not found at '{db_path}'.")
        return []
    try:
        with open(db_path, 'r', encoding='utf-8') as f:
            vector_db = json.load(f)
        # Convert embeddings back to numpy arrays for calculations
        for item in vector_db:
            item['embedding'] = np.array(item['embedding'])
        logging.info(f"Successfully loaded vector database from '{db_path}'.")
        return vector_db
    except Exception as e:
        logging.error(f"Failed to load or parse vector database from '{db_path}': {e}")
        return []

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculates cosine similarity between two vectors."""
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0 # Avoid division by zero
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def retrieve_chunks(
    query: str,
    vector_db: List[Dict[str, Any]],
    top_n: int = 5,
    provider: Literal['openai', 'aws'] = DEFAULT_EMBEDDING_PROVIDER,
    openai_model: str = DEFAULT_OPENAI_EMBEDDING_MODEL,
    aws_model_id: str = AWS_EMBEDDING_MODEL_ID
) -> List[Dict[str, Any]]:
    """
    Retrieves the top_n most relevant chunks for a given query using the specified provider.

    Args:
        query: The user's query string.
        vector_db: The loaded vector database (list of dicts).
        top_n: The number of top chunks to retrieve.
        provider: The embedding service ('openai' or 'aws').
        openai_model: The OpenAI model (if provider is 'openai').
        aws_model_id: The AWS model (if provider is 'aws').

    Returns:
        A list of dictionaries, each containing 'score' and 'chunk' data,
        sorted by similarity score in descending order.
        Returns empty list on failure or if DB is empty.
    """
    if not vector_db:
        logging.warning("Vector database is empty. Cannot retrieve chunks.")
        return []
    if not query:
        logging.warning("Query is empty. Cannot retrieve chunks.")
        return []

    query_embedding = None
    try:
        # 1. Embed the query using the specified provider
        logging.info(f"Creating query embedding using provider: {provider}")
        if provider == 'openai':
            if not openai_client:
                logging.error("OpenAI client not available for query embedding.")
                return []
            response = openai_client.embeddings.create(input=[query], model=openai_model)
            query_embedding = np.array(response.data[0].embedding)
        elif provider == 'aws':
            if not bedrock_runtime:
                logging.error("AWS Bedrock client not available for query embedding.")
                return []
            request_body = json.dumps({"inputText": query})
            response = bedrock_runtime.invoke_model(
                body=request_body, modelId=aws_model_id, accept='application/json', contentType='application/json'
            )
            response_body = json.loads(response.get('body').read())
            query_embedding = np.array(response_body.get('embedding'))
        else:
            logging.error(f"Invalid or unavailable provider '{provider}' for query embedding.")
            return []

        if query_embedding is None:
             logging.error(f"Failed to create query embedding with provider {provider}.")
             return []
        logging.info(f"Successfully created embedding for query using {provider}.")

        # 2. Calculate similarities
        similarities = []
        for item in vector_db:
            sim = cosine_similarity(query_embedding, item['embedding'])
            similarities.append((sim, item)) # Store similarity score with the item

        # 3. Sort by similarity (descending)
        similarities.sort(key=lambda x: x[0], reverse=True)

        # 4. Get top N results formatted with score and chunk
        top_results = [{'score': score, 'chunk': item} for score, item in similarities[:top_n]]
        logging.info(f"Retrieved top {len(top_results)} chunks for the query.")
        return top_results

    except Exception as e:
        logging.error(f"Failed to retrieve chunks: {e}")
        return []

# --- Re-ranking Step (Placeholder) ---
def rerank_chunks(query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Placeholder for a re-ranking function. Currently returns results as is.
    Accepts and returns list of dicts with 'score' and 'chunk'.

    Args:
        query: The original user query.
        results: The list of dictionaries ({'score': score, 'chunk': chunk})
                 retrieved by the initial retrieval step.

    Returns:
        The potentially re-ordered list of result dictionaries.
    """
    logging.info("Skipping re-ranking step (placeholder).")
    # Future implementation could use more sophisticated models like cross-encoders.
    # It would likely re-calculate scores or just re-order the 'results' list.
    return results

# --- Combined Retrieval/Answering Function ---
def answer_query(query: str, db_path: str = VECTOR_DB_PATH, top_n: int = 5, provider: Literal['openai', 'aws'] = DEFAULT_EMBEDDING_PROVIDER):
    """
    Loads the DB, retrieves relevant chunks with scores, re-ranks (placeholder),
    and returns them.
    """
    logging.info(f"Answering query: '{query}'")
    vector_db = load_vector_db(db_path.format(provider=provider))
    if not vector_db:
        logging.error("Failed to load vector database. Cannot answer query.")
        return []

    retrieved = retrieve_chunks(query, vector_db, top_n=top_n, provider=provider)
    if not retrieved:
        logging.warning("No chunks retrieved for the query.")
        return []

    reranked = rerank_chunks(query, retrieved) # Apply re-ranking (currently identity)
    return reranked

# --- 6. Evaluation Step (Simple Example) ---
def evaluate_retrieval(query: str, retrieved_results: List[Dict[str, Any]], expected_ids: List[str] = None):
    """
    Simple evaluation function. Prints retrieved chunks/scores and checks for expected IDs.

    Args:
        query: The query that was run.
        retrieved_results: The list of dicts ({'score': score, 'chunk': chunk})
                           returned by `answer_query`.
        expected_ids: Optional list of chunk_ids expected to be relevant.
    """
    print("\n--- Evaluation ---")
    print(f"Query: '{query}'")
    print(f"Retrieved {len(retrieved_results)} results:")
    for i, result in enumerate(retrieved_results):
        chunk = result['chunk']
        score = result['score']
        print(f"  {i+1}. Score: {score:.4f}, ID: {chunk['chunk_id']}, Filename: {chunk['filename']}")

    # Print the text of the top result after the loop
    if retrieved_results:
        top_chunk = retrieved_results[0]['chunk']
        print(f"\nTop Result Text (ID: {top_chunk['chunk_id']}):")
        print(f"  \"{top_chunk['text']}\"")

    if expected_ids:
        retrieved_ids = {result['chunk']['chunk_id'] for result in retrieved_results}
        hits = [chunk_id for chunk_id in expected_ids if chunk_id in retrieved_ids]
        misses = [chunk_id for chunk_id in expected_ids if chunk_id not in retrieved_ids]

        print("\nExpected Hits:")
        if hits:
            print(f"  Found: {', '.join(hits)}")
            hit_rate = len(hits) / len(expected_ids)
            print(f"  Hit Rate: {hit_rate:.2f}")
        else:
            print("  None of the expected chunks were found.")

        if misses:
            print(f"\nExpected Misses: {', '.join(misses)}")
    print("\n--- End Evaluation ---")


# --- Main Execution Example ---
if __name__ == "__main__":
    # --- Setup: Create dummy data for testing ---
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    if not os.listdir(DATA_DIR): # Only create if data dir is empty
        logging.info("Data directory is empty. Creating sample files...")
        with open(os.path.join(DATA_DIR, "doc1.txt"), "w") as f:
            f.write("The quick brown fox jumps over the lazy dog. This is the first document about animals.\n")
            f.write("Foxes are mammals known for their cunning nature. Dogs are loyal companions.")
        with open(os.path.join(DATA_DIR, "doc2.txt"), "w") as f:
            f.write("The world of finance involves stocks, bonds, and investment strategies.\n")
            f.write("Understanding market trends is crucial for financial success. Banks offer various services.")
        with open(os.path.join(DATA_DIR, "doc3.txt"), "w") as f:
            f.write("Artificial intelligence is transforming industries. Machine learning is a key component.\n")
            f.write("Natural Language Processing enables computers to understand human language. RAG is a technique combining retrieval and generation.")
        logging.info("Sample files created in 'data/' directory.")
    else:
        logging.info("Data directory already contains files. Skipping sample file creation.")

    # --- Select Provider ---
    # Choose 'openai' or 'aws'. Could be set via env var or command line arg.
    # SELECTED_PROVIDER: Literal['openai', 'aws', 'none'] = DEFAULT_EMBEDDING_PROVIDER
    SELECTED_PROVIDER: Literal['openai', 'aws', 'none'] = 'aws'
    logging.info(f"Using embedding provider: {SELECTED_PROVIDER}")

    if SELECTED_PROVIDER == 'none':
        logging.error("No embedding provider available (OpenAI or AWS). Cannot build knowledge base or answer queries.")
        exit(1) # Exit if no provider is configured/available

    # Determine Vector DB path based on provider
    current_vector_db_path = VECTOR_DB_PATH.format(provider=SELECTED_PROVIDER)
    logging.info(f"Vector database path for this provider: {current_vector_db_path}")


    # --- Step 1-4: Build Knowledge Base ---
    # Check if the selected provider is available before building
    provider_is_available = (SELECTED_PROVIDER == 'openai' and openai_available) or \
                            (SELECTED_PROVIDER == 'aws' and aws_available)

    if provider_is_available:
         # Example: Only build if the DB doesn't exist, or add a flag to force rebuild
         if not os.path.exists(current_vector_db_path):
             logging.info(f"Building knowledge base using {SELECTED_PROVIDER} provider...")
             build_knowledge_base(output_path=current_vector_db_path, provider=SELECTED_PROVIDER)
         else:
             logging.info(f"Knowledge base '{current_vector_db_path}' already exists. Skipping build.")
             logging.info(f"Delete '{current_vector_db_path}' to rebuild.")

    else:
        logging.error(f"Selected provider '{SELECTED_PROVIDER}' is not available/configured. Skipping knowledge base build.")
        if not os.path.exists(current_vector_db_path):
            logging.error("Knowledge base cannot be built and does not exist. Exiting.")
            exit(1) # Exit if KB cannot be built and doesn't exist


    # --- Step 5 & 6: Retrieve and Evaluate (only if KB exists) ---
    if os.path.exists(current_vector_db_path):
        logging.info("\n--- Testing Retrieval ---")
        test_query_1 = "Tell me about financial markets"
        results_1 = answer_query(test_query_1, db_path=current_vector_db_path, provider=SELECTED_PROVIDER)
        # print(results_1) # Optionally print raw results
        evaluate_retrieval(test_query_1, results_1) # Simple evaluation

        test_query_2 = "What is RAG in NLP?"
        results_2 = answer_query(test_query_2, db_path=current_vector_db_path, provider=SELECTED_PROVIDER)
        # print(results_2)
        evaluate_retrieval(test_query_2, results_2)


        # Add more test queries relevant to your sample data or use case
        # test_query_3 = "Describe the characteristics of a fox"
        # results_3 = answer_query(test_query_3, db_path=current_vector_db_path, provider=SELECTED_PROVIDER)
        # evaluate_retrieval(test_query_3, results_3)


    else:
        logging.warning(f"Vector database '{current_vector_db_path}' not found. Cannot run retrieval test.")
        logging.warning("Ensure the knowledge base was built successfully (requires appropriate API key/credentials).") 