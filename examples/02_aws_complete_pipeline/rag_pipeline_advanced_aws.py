import os
import json
import numpy as np
import logging
from typing import List, Dict, Any
import fitz # PyMuPDF
import boto3

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Ensure AWS credentials are configured (e.g., via environment variables or IAM role)

# AWS Bedrock Configuration
AWS_REGION_NAME = os.environ.get("AWS_REGION", "eu-west-1") # Or your preferred region
AWS_EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
AWS_PROFILE_NAME = "my-mcp-server-profile" # Set the profile name directly
try:
    # Use default credential chain (env vars, shared cred file, IAM role, etc.)
    session = boto3.Session(profile_name=AWS_PROFILE_NAME, region_name=AWS_REGION_NAME)
    bedrock_runtime = session.client(service_name='bedrock-runtime')
    aws_available = True
    logging.info(f"Successfully initialized AWS Bedrock runtime client in region '{AWS_REGION_NAME}'.")
except Exception as e:
    logging.warning(f"Failed to initialize AWS Bedrock runtime client. AWS embeddings unavailable. Error: {e}")
    bedrock_runtime = None
    aws_available = False


EMBEDDING_PROVIDER_AVAILABLE = aws_available
VECTOR_DB_PATH = "vector_db/vector_db_aws.json" # Fixed path for AWS
DATA_DIR = "data"
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
    aws_model_id: str = AWS_EMBEDDING_MODEL_ID,
) -> List[np.ndarray]:
    """
    Creates embeddings for text chunks using AWS Bedrock.

    Args:
        chunks: A list of chunk dictionaries (from chunk_documents).
        aws_model_id: The AWS Bedrock model ID to use.

    Returns:
        A list of embeddings (as numpy arrays), corresponding to the input chunks.
        Returns empty list if the AWS client is not available or API call fails.
    """
    if not aws_available:
        logging.error("AWS Bedrock client is not available. Cannot create embeddings.")
        return []
    # AWS Titan V2 currently requires sequential processing (batch size 1) via API
    return _create_embeddings_aws(chunks, model_id=aws_model_id)


# --- 4. Storage Step ---
def store_embeddings(chunks: List[Dict[str, Any]], embeddings: List[np.ndarray], output_path: str = VECTOR_DB_PATH):
    """
    Stores chunks and their corresponding embeddings in a JSON file.

    Args:
        chunks: The list of chunk dictionaries.
        embeddings: The list of corresponding embeddings (numpy arrays).
        output_path: The file path to save the data. Defaults to VECTOR_DB_PATH.
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
def build_knowledge_base(data_dir: str = DATA_DIR, output_path: str = VECTOR_DB_PATH):
    """
    Runs the complete ingestion pipeline: Ingest -> Chunk -> Embed (AWS) -> Store.
    """
    logging.info("Starting knowledge base build process using AWS Bedrock...")
    documents = ingest_documents(data_dir)
    if not documents:
        logging.warning("No documents ingested. Stopping knowledge base build.")
        return

    chunks = chunk_documents(documents)
    if not chunks:
        logging.warning("No chunks created. Stopping knowledge base build.")
        return

    embeddings = create_embeddings(chunks) # Directly uses AWS
    if not embeddings or len(embeddings) != len(chunks):
        logging.error("Embedding creation failed or resulted in mismatch. Stopping knowledge base build.")
        return

    store_embeddings(chunks, embeddings, output_path)
    logging.info("Knowledge base build process completed.")

# --- 5. Retrieval Step ---
def load_vector_db(db_path: str = VECTOR_DB_PATH) -> List[Dict[str, Any]]:
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
    aws_model_id: str = AWS_EMBEDDING_MODEL_ID
) -> List[Dict[str, Any]]:
    """
    Retrieves the top_n most relevant chunks for a given query using AWS Bedrock.

    Args:
        query: The user's query string.
        vector_db: The loaded vector database (list of dicts).
        top_n: The number of top chunks to retrieve.
        aws_model_id: The AWS model used for embedding.

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
        # 1. Embed the query using AWS Bedrock
        logging.info(f"Creating query embedding using AWS Bedrock model: {aws_model_id}")
        if not bedrock_runtime:
            logging.error("AWS Bedrock client not available for query embedding.")
            return []
        request_body = json.dumps({"inputText": query})
        response = bedrock_runtime.invoke_model(
            body=request_body, modelId=aws_model_id, accept='application/json', contentType='application/json'
        )
        response_body = json.loads(response.get('body').read())
        query_embedding = np.array(response_body.get('embedding'))

        if query_embedding is None:
             logging.error("Failed to create query embedding with AWS Bedrock.")
             return []
        logging.info("Successfully created embedding for query using AWS.")

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
def answer_query(query: str, db_path: str = VECTOR_DB_PATH, top_n: int = 5):
    """
    Loads the DB, retrieves relevant chunks (using AWS) with scores, re-ranks (placeholder),
    and returns them.
    """
    logging.info(f"Answering query: '{query}' using AWS embeddings")
    vector_db = load_vector_db(db_path)
    if not vector_db:
        logging.error("Failed to load vector database. Cannot answer query.")
        return []

    retrieved = retrieve_chunks(query, vector_db, top_n=top_n) # Directly uses AWS
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

    # --- Check AWS Availability ---
    logging.info(f"Using AWS Bedrock for embeddings.")
    if not EMBEDDING_PROVIDER_AVAILABLE:
        logging.error("AWS Bedrock client not available/configured. Cannot build knowledge base or answer queries.")
        exit(1) # Exit if AWS is not configured/available

    # Vector DB path is now fixed
    current_vector_db_path = VECTOR_DB_PATH
    logging.info(f"Vector database path: {current_vector_db_path}")


    # --- Step 1-4: Build Knowledge Base ---
    # Example: Only build if the DB doesn't exist, or add a flag to force rebuild
    if not os.path.exists(current_vector_db_path):
         logging.info(f"Building knowledge base using AWS Bedrock...")
         build_knowledge_base(output_path=current_vector_db_path) # No provider needed
    else:
         logging.info(f"Knowledge base '{current_vector_db_path}' already exists. Skipping build.")
         logging.info(f"Delete '{current_vector_db_path}' to rebuild.")


    # --- Step 5 & 6: Retrieve and Evaluate (only if KB exists) ---
    if os.path.exists(current_vector_db_path):
        logging.info("\n--- Testing Retrieval ---")
        test_query_1 = "Tell me about financial markets"
        results_1 = answer_query(test_query_1, db_path=current_vector_db_path) # No provider needed
        # print(results_1) # Optionally print raw results
        evaluate_retrieval(test_query_1, results_1) # Simple evaluation

        test_query_2 = "What is RAG in NLP?"
        results_2 = answer_query(test_query_2, db_path=current_vector_db_path) # No provider needed
        # print(results_2)
        evaluate_retrieval(test_query_2, results_2)
        
    else:
        logging.warning(f"Vector database '{current_vector_db_path}' not found. Cannot run retrieval test.")
        logging.warning("Ensure the knowledge base was built successfully (requires AWS credentials).") 