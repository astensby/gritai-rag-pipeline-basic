import os
import json
import numpy as np
from openai import OpenAI
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Ensure OPENAI_API_KEY environment variable is set
# You can set it in your terminal like this: export OPENAI_API_KEY='your_key_here'
# Or use a .env file and a library like python-dotenv
try:
    client = OpenAI() # Reads OPENAI_API_KEY from environment variable
except Exception as e:
    logging.error(f"Failed to initialize OpenAI client. Ensure OPENAI_API_KEY is set: {e}")
    # Provide a dummy client or raise an error if OpenAI is strictly required
    # For now, let's allow proceeding but embedding will fail.
    client = None

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small" # Choose a cost-effective default
VECTOR_DB_PATH = "vector_db/vector_db.json"
DATA_DIR = "data"
MAX_EMBEDDING_BATCH_SIZE = 2000 # Keep slightly below OpenAI's limit of 2048 for safety

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
            # Add more file types (e.g., .pdf, .docx) handling here later
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
def create_embeddings(chunks: List[Dict[str, Any]], model: str = DEFAULT_EMBEDDING_MODEL, batch_size: int = MAX_EMBEDDING_BATCH_SIZE) -> List[np.ndarray]:
    """
    Creates embeddings for text chunks using OpenAI API, handling batching.

    Args:
        chunks: A list of chunk dictionaries (from chunk_documents).
        model: The OpenAI embedding model to use.
        batch_size: The maximum number of chunks to send in one API call.

    Returns:
        A list of embeddings (as numpy arrays), corresponding to the input chunks.
        Returns empty list if OpenAI client is not available or API call fails.
    """
    if not client:
        logging.error("OpenAI client not initialized. Cannot create embeddings.")
        return []
    if not chunks:
        logging.warning("No chunks provided to create_embeddings.")
        return []

    all_embeddings = []
    total_chunks = len(chunks)
    logging.info(f"Creating embeddings for {total_chunks} chunks in batches of {batch_size}...")

    for i in range(0, total_chunks, batch_size):
        batch_chunks = chunks[i:min(i + batch_size, total_chunks)]
        texts_to_embed = [chunk['text'] for chunk in batch_chunks]

        try:
            logging.info(f"Processing batch {i // batch_size + 1}/{(total_chunks + batch_size - 1) // batch_size} ({len(batch_chunks)} chunks)")
            response = client.embeddings.create(input=texts_to_embed, model=model)
            batch_embeddings = [np.array(item.embedding) for item in response.data]
            all_embeddings.extend(batch_embeddings)
            logging.info(f"Successfully created embeddings for batch.")

        except Exception as e:
            logging.error(f"OpenAI API call failed for batch starting at index {i}: {e}")
            # Decide on error handling: stop, continue with gaps, retry?
            # For simplicity, we'll stop and return what we have so far, or an empty list.
            # Depending on the use case, more robust error handling might be needed.
            logging.error("Aborting embedding creation due to API error.")
            return [] # Or potentially return all_embeddings collected so far

    if len(all_embeddings) == total_chunks:
        logging.info(f"Successfully created all {len(all_embeddings)} embeddings using model '{model}'.")
    else:
        logging.warning(f"Mismatch in expected ({total_chunks}) and created ({len(all_embeddings)}) embeddings.")


    return all_embeddings

# --- 4. Storage Step ---
def store_embeddings(chunks: List[Dict[str, Any]], embeddings: List[np.ndarray], output_path: str = VECTOR_DB_PATH):
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
def build_knowledge_base(data_dir: str = DATA_DIR, output_path: str = VECTOR_DB_PATH):
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

    embeddings = create_embeddings(chunks)
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

def retrieve_chunks(query: str, vector_db: List[Dict[str, Any]], top_n: int = 5, model: str = DEFAULT_EMBEDDING_MODEL) -> List[Dict[str, Any]]:
    """
    Retrieves the top_n most relevant chunks for a given query, including scores.

    Args:
        query: The user's query string.
        vector_db: The loaded vector database (list of dicts).
        top_n: The number of top chunks to retrieve.
        model: The OpenAI embedding model to use for the query.

    Returns:
        A list of dictionaries, each containing 'score' and 'chunk' data,
        sorted by similarity score in descending order.
        Returns empty list on failure or if DB is empty.
    """
    if not client:
        logging.error("OpenAI client not initialized. Cannot create query embedding.")
        return []
    if not vector_db:
        logging.warning("Vector database is empty. Cannot retrieve chunks.")
        return []

    try:
        # 1. Embed the query
        response = client.embeddings.create(input=[query], model=model)
        query_embedding = np.array(response.data[0].embedding)
        logging.info(f"Successfully created embedding for query.")

        # 2. Calculate similarities - Brute force approach 
        # The scalable way for this would be to use ANN and a vector database like Faiss
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
def answer_query(query: str, db_path: str = VECTOR_DB_PATH, top_n: int = 5) -> List[Dict[str, Any]]:
    """
    Loads the DB, retrieves relevant chunks with scores, re-ranks (placeholder),
    and returns them.
    """
    logging.info(f"Answering query: '{query}'")
    vector_db = load_vector_db(db_path)
    if not vector_db:
        logging.error("Failed to load vector database. Cannot answer query.")
        return []

    retrieved = retrieve_chunks(query, vector_db, top_n=top_n)
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

    # --- Step 1-4: Build Knowledge Base ---
    # Check if OpenAI API key is likely available before attempting to build
    if client and os.getenv("OPENAI_API_KEY"):
        build_knowledge_base()
    else:
        logging.error("OPENAI_API_KEY not found in environment variables. Skipping knowledge base build.")
        logging.error("Please set the OPENAI_API_KEY environment variable and run the script again.")
        # Optional: Exit here if KB is essential for further steps
        # exit(1)


    # --- Step 5 & 6: Retrieve and Evaluate (only if KB exists) ---
    if os.path.exists(VECTOR_DB_PATH):
        logging.info("\n--- Testing Retrieval ---")
        test_query_1 = "Tell me about space exploration"
        results_1 = answer_query(test_query_1)
        print(results_1)
        evaluate_retrieval(test_query_1, results_1) # Simple evaluation

        test_query_2 = "What is a marine mammal?"
        results_2 = answer_query(test_query_2)
        print(results_2)
        # Example evaluation with expected IDs (adjust IDs based on actual chunking)
        # You would need to run build_knowledge_base once, inspect vector_db.json
        # to find the relevant chunk IDs, and then add them here.
        # evaluate_retrieval(test_query_2, results_2, expected_ids=["doc1.txt_0", "doc1.txt_1"])
        evaluate_retrieval(test_query_2, results_2)

    else:
        logging.warning(f"Vector database '{VECTOR_DB_PATH}' not found. Cannot run retrieval test.")
        logging.warning("Ensure the knowledge base was built successfully (requires OPENAI_API_KEY).") 