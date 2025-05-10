import os
import json
import numpy as np
import logging
from typing import List, Dict, Any
import fitz # PyMuPDF
import boto3
import psycopg2 # <-- Add DB driver
from psycopg2 import sql # For safe query building
from psycopg2.extras import execute_values # For efficient batch inserts

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

# Database Configuration (using environment variables)
DB_NAME = os.environ.get("PGVECTOR_DB_NAME", "vectordb")
DB_USER = os.environ.get("PGVECTOR_DB_USER", "your_username") # Replace with your user if different
DB_PASSWORD = os.environ.get("PGVECTOR_DB_PASSWORD", "your_password") # Set this environment variable!
DB_HOST = os.environ.get("PGVECTOR_DB_HOST", "localhost")
DB_PORT = os.environ.get("PGVECTOR_DB_PORT", "5432")
DB_SCHEMA = "embeddings"
DB_TABLE = "documents"


EMBEDDING_PROVIDER_AVAILABLE = aws_available
DATA_DIR = "data"
# Note: AWS Bedrock Titan V2 currently has a limit of 1 input text per request. Batching needs sequential calls.
MAX_AWS_EMBEDDING_BATCH_SIZE = 1 # For Titan V2, process one by one in a loop


# --- Database Connection ---
def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        logging.info(f"Successfully connected to database '{DB_NAME}' on {DB_HOST}:{DB_PORT}.")
        # Ensure pgvector extension is enabled (optional check)
        # with conn.cursor() as cur:
        #     cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        # conn.commit()
        return conn
    except psycopg2.OperationalError as e:
        logging.error(f"Database connection failed: {e}")
        logging.error("Please ensure the database is running and connection details (env vars) are correct:")
        logging.error(f"  DB_NAME='{DB_NAME}', DB_USER='{DB_USER}', DB_HOST='{DB_HOST}', DB_PORT='{DB_PORT}'")
        logging.error("  Check if PGVECTOR_DB_PASSWORD environment variable is set.")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during database connection: {e}")
        return None

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
def store_embeddings(chunks: List[Dict[str, Any]], embeddings: List[np.ndarray]):
    """
    Stores chunks and their corresponding embeddings in the PostgreSQL database.
    Clears existing entries for the processed filenames before inserting new ones.

    Args:
        chunks: The list of chunk dictionaries (must include 'filename', 'text').
        embeddings: The list of corresponding embeddings (numpy arrays).
    """
    if len(chunks) != len(embeddings):
        logging.error("Mismatch between number of chunks and embeddings. Cannot store.")
        return
    if not chunks:
        logging.warning("No chunks provided to store_embeddings.")
        return

    conn = get_db_connection()
    if not conn:
        logging.error("Failed to get database connection. Cannot store embeddings.")
        return

    # Prepare data for insertion: (filename, content, embedding_list)
    # We use the 'text' field from the chunk as the 'content' column
    data_to_insert = [
        (chunk['filename'], chunk['text'], embedding.tolist())
        for chunk, embedding in zip(chunks, embeddings)
        if chunk.get('filename') and chunk.get('text') and embedding is not None # Basic validation
    ]

    if not data_to_insert:
        logging.warning("No valid data prepared for database insertion.")
        conn.close()
        return

    # Get unique filenames processed in this batch
    processed_filenames = sorted(list(set(item[0] for item in data_to_insert)))

    try:
        with conn.cursor() as cur:
            # Create schema if it doesn't exist
            cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(DB_SCHEMA)))
            logging.info(f"Ensured schema '{DB_SCHEMA}' exists.")

            # Create table if it doesn't exist (idempotent)
            # Use vector(1024) based on your schema and AWS Titan V2 output
            create_table_sql = sql.SQL("""
            CREATE TABLE IF NOT EXISTS {}.{} (
                chunk_id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
                filename TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding vector(1024) NOT NULL
                -- Add indexes separately for clarity/control
            );
            """).format(sql.Identifier(DB_SCHEMA), sql.Identifier(DB_TABLE))
            cur.execute(create_table_sql)
            logging.info(f"Ensured table '{DB_SCHEMA}.{DB_TABLE}' exists.")

            # Add embedding index if it doesn't exist (IVFFlat example)
            # Adjust lists parameter based on expected data size
            # Note: Index creation can take time on large tables
            create_index_sql = sql.SQL("""
            CREATE INDEX IF NOT EXISTS documents_embedding_ivfflat_idx
            ON {}.{} USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
            """).format(sql.Identifier(DB_SCHEMA), sql.Identifier(DB_TABLE))
            cur.execute(create_index_sql)
            logging.info(f"Ensured IVFFlat index exists on embedding column.")

            # Add unique constraint if it doesn't exist (on filename, content hash?)
            # The original unique constraint (filename, chunk_id) is handled by PK.
            # To prevent inserting the *exact* same content for a file again,
            # a constraint on (filename, md5(content)) could be used, but let's
            # clear by filename first for simplicity.
            # Example:
            # CREATE UNIQUE INDEX IF NOT EXISTS unique_file_content_idx
            # ON embeddings.documents (filename, md5(content));


            # Clear existing entries for the files being processed in this run
            # This prevents duplicates if the script runs multiple times.
            delete_sql = sql.SQL("DELETE FROM {}.{} WHERE filename = ANY(%s)").format(
                sql.Identifier(DB_SCHEMA), sql.Identifier(DB_TABLE)
            )
            cur.execute(delete_sql, (processed_filenames,))
            logging.info(f"Cleared existing entries for {len(processed_filenames)} files: {', '.join(processed_filenames)}")


            # Use execute_values for efficient batch insertion
            insert_query = sql.SQL("""
                INSERT INTO {}.{} (filename, content, embedding) VALUES %s
            """).format(sql.Identifier(DB_SCHEMA), sql.Identifier(DB_TABLE))

            execute_values(
                cur,
                insert_query,
                data_to_insert,
                template="(%s, %s, %s::vector)", # Cast embedding list to vector type
                page_size=100 # Adjust batch size as needed
            )

            inserted_count = len(data_to_insert)
            conn.commit()
            logging.info(f"Successfully stored {inserted_count} chunk embeddings in the database for {len(processed_filenames)} files.")

    except psycopg2.Error as e:
        logging.error(f"Database error during embedding storage: {e}")
        logging.error(f"SQL Error Code: {e.pgcode}, Message: {e.pgerror}")
        conn.rollback() # Rollback transaction on error
    except Exception as e:
        logging.error(f"An unexpected error occurred during embedding storage: {e}")
        conn.rollback()
    finally:
        if conn:
            conn.close()
            logging.info("Database connection closed.")


# --- Combined Ingestion Pipeline ---
def build_knowledge_base(data_dir: str = DATA_DIR):
    """
    Runs the complete ingestion pipeline: Ingest -> Chunk -> Embed (AWS) -> Store (DB).
    """
    logging.info("Starting knowledge base build process using AWS Bedrock and PostgreSQL...")
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

    store_embeddings(chunks, embeddings)
    logging.info("Knowledge base build process completed.")

# --- 5. Retrieval Step ---
def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculates cosine similarity between two vectors."""
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0 # Avoid division by zero
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def retrieve_chunks(
    query: str,
    top_n: int = 5,
    aws_model_id: str = AWS_EMBEDDING_MODEL_ID
) -> List[Dict[str, Any]]:
    """
    Retrieves the top_n most relevant chunks for a given query from the PostgreSQL
    database using pgvector cosine distance (<=>) and AWS Bedrock for query embedding.

    Args:
        query: The user's query string.
        top_n: The number of top chunks to retrieve.
        aws_model_id: The AWS model used for embedding.

    Returns:
        A list of dictionaries, each containing 'score' (cosine similarity) and
        'chunk' data ('chunk_id', 'filename', 'text'), sorted by similarity score
        in descending order. Returns empty list on failure.
    """
    if not query:
        logging.warning("Query is empty. Cannot retrieve chunks.")
        return []

    query_embedding_list = None
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
        # Get embedding as a list for psycopg2 parameterization
        query_embedding_list = response_body.get('embedding')

        if query_embedding_list is None:
             logging.error("Failed to create query embedding with AWS Bedrock.")
             return []
        logging.info("Successfully created embedding for query using AWS.")

        # 2. Query the database
        conn = get_db_connection()
        if not conn:
            logging.error("Failed to get database connection for retrieval.")
            return []

        results = []
        try:
            with conn.cursor() as cur:
                # Use <=> for cosine distance (0=identical, 2=opposite) -> smaller is better
                # Order by distance ASC and limit
                # Select chunk_id, filename, content, and distance
                retrieve_sql = sql.SQL("""
                    SELECT chunk_id, filename, content, embedding <=> %s::vector AS distance
                    FROM {}.{}
                    ORDER BY distance ASC
                    LIMIT %s;
                """).format(sql.Identifier(DB_SCHEMA), sql.Identifier(DB_TABLE))

                cur.execute(retrieve_sql, (query_embedding_list, top_n))
                db_results = cur.fetchall()

                # 3. Format results
                # Convert distance back to similarity score (1 - distance for cosine)
                # Note: Max cosine similarity is 1.
                for row in db_results:
                    chunk_id, filename, content, distance = row
                    similarity_score = 1.0 - distance # Cosine distance ranges 0-2
                    results.append({
                        'score': max(0.0, similarity_score), # Ensure score is not negative due to float precision
                        'chunk': {
                            'chunk_id': chunk_id, # Use the DB-generated bigint ID
                            'filename': filename,
                            'text': content,
                            # 'embedding': None # Don't usually need the embedding vector itself here
                        }
                    })
                logging.info(f"Retrieved top {len(results)} chunks from database.")

        except psycopg2.Error as e:
            logging.error(f"Database error during retrieval: {e}")
            logging.error(f"SQL Error Code: {e.pgcode}, Message: {e.pgerror}")
            conn.rollback()
        except Exception as e:
            logging.error(f"An unexpected error occurred during retrieval: {e}")
            conn.rollback()
        finally:
            if conn:
                conn.close()
                logging.info("Database connection closed after retrieval.")

        return results

    except Exception as e:
        # Catch errors from embedding creation or other unexpected issues
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
def answer_query(query: str, top_n: int = 5):
    """
    Retrieves relevant chunks from the DB (using AWS for query embedding),
    re-ranks (placeholder), and returns them.
    """
    logging.info(f"Answering query: '{query}' using AWS embeddings and PostgreSQL")
    retrieved = retrieve_chunks(query, top_n=top_n) # No db/vector_db argument needed
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

    # --- Check Database Connectivity ---
    conn_test = get_db_connection()
    if not conn_test:
        logging.error("Failed initial database connection check. Please verify settings and ensure DB is running.")
        exit(1)
    else:
        conn_test.close()
        logging.info("Initial database connection successful.")


    # --- Step 1-4: Build Knowledge Base ---
    # Decide whether to rebuild. For example, always rebuild for this script,
    # or check if the table is empty, or use a command-line flag.
    # For simplicity, let's always attempt to build/update.
    logging.info(f"Building/updating knowledge base in PostgreSQL...")
    build_knowledge_base() # No output_path needed


    # --- Step 5 & 6: Retrieve and Evaluate ---
    # No need to check for file existence anymore, assume build attempted.
    logging.info("\n--- Testing Retrieval ---")
    test_query_1 = "Tell me about financial markets"
    results_1 = answer_query(test_query_1)
    evaluate_retrieval(test_query_1, results_1)

    test_query_2 = "What is RAG in NLP?"
    results_2 = answer_query(test_query_2)
    evaluate_retrieval(test_query_2, results_2)

    logging.info("\nScript finished.") 