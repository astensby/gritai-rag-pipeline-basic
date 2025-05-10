import os
import json
import numpy as np
import logging
from typing import List, Dict, Any
import boto3
import psycopg2 # <-- Add DB driver
from psycopg2 import sql # For safe query building

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


# --- 5. Retrieval Step ---
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
        'chunk' data ('chunk_id', 'filename', 'content'), sorted by similarity score
        in descending order. Returns empty list on failure.
    """
    if not query:
        logging.warning("Query is empty. Cannot retrieve chunks.")
        return []
    if not aws_available or not bedrock_runtime:
        logging.error("AWS Bedrock client not available for query embedding.")
        return []

    query_embedding_list = None
    try:
        # 1. Embed the query using AWS Bedrock
        logging.info(f"Creating query embedding using AWS Bedrock model: {aws_model_id}")
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
                    # pgvector cosine distance is 1 - cosine_similarity
                    # Cosine similarity = 1 - distance
                    similarity_score = 1.0 - distance
                    results.append({
                        'score': max(0.0, similarity_score), # Ensure score is not negative
                        'chunk': {
                            'chunk_id': chunk_id, # Use the DB-generated bigint ID
                            'filename': filename,
                            'content': content,
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
    return results

# --- Combined Retrieval/Answering Function ---
def answer_query(query: str, top_n: int = 5):
    """
    Retrieves relevant chunks from the DB (using AWS for query embedding),
    re-ranks (placeholder), and returns them.
    """
    logging.info(f"Answering query: '{query}' using AWS embeddings and PostgreSQL")
    retrieved = retrieve_chunks(query, top_n=top_n)
    if not retrieved:
        logging.warning("No chunks retrieved for the query.")
        return []

    reranked = rerank_chunks(query, retrieved) # Apply re-ranking (currently identity)
    return reranked

# --- Evaluation Step (Simple Example) ---
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
    if not retrieved_results:
        print("No results retrieved.")
        print("--- End Evaluation ---")
        return

    print(f"Retrieved {len(retrieved_results)} results:")
    for i, result in enumerate(retrieved_results):
        chunk = result['chunk']
        score = result['score']
        print(f"  {i+1}. Score: {score:.4f}, ID: {chunk['chunk_id']}, Filename: {chunk['filename']}")

    # Print the content of the top result after the loop
    top_chunk = retrieved_results[0]['chunk']
    print(f"\nTop Result Text (ID: {top_chunk['chunk_id']}):")
    print(f"  \"{top_chunk['content']}\"")

    if expected_ids:
        retrieved_ids = {result['chunk']['chunk_id'] for result in retrieved_results}
        hits = [chunk_id for chunk_id in expected_ids if chunk_id in retrieved_ids]
        misses = [chunk_id for chunk_id in expected_ids if chunk_id not in retrieved_ids]

        print("\nExpected Hits:")
        if hits:
            print(f"  Found: {', '.join(map(str, hits))}") # Ensure IDs are strings for join
            hit_rate = len(hits) / len(expected_ids)
            print(f"  Hit Rate: {hit_rate:.2f}")
        else:
            print("  None of the expected chunks were found.")

        if misses:
            print(f"\nExpected Misses: {', '.join(map(str, misses))}") # Ensure IDs are strings for join
    print("\n--- End Evaluation ---")


# --- Main Execution Example ---
if __name__ == "__main__":

    # --- Check AWS Availability ---
    logging.info(f"Checking AWS Bedrock availability...")
    if not aws_available:
        logging.error("AWS Bedrock client not available/configured. Cannot answer queries.")
        exit(1) # Exit if AWS is not configured/available
    else:
        logging.info("AWS Bedrock client available.")

    # --- Check Database Connectivity ---
    logging.info("Checking database connectivity...")
    conn_test = get_db_connection()
    if not conn_test:
        logging.error("Failed initial database connection check. Please verify settings and ensure DB is running.")
        exit(1)
    else:
        conn_test.close()
        logging.info("Initial database connection successful.")

    # --- Query the Existing Knowledge Base ---
    logging.info("--- Testing Retrieval Against Existing VectorDB ---")
    test_query_1 = "Tell me about NVIDIA revenues"
    results_1 = answer_query(test_query_1)
    evaluate_retrieval(test_query_1, results_1)

    test_query_2 = "What are the biggest competitors of NVIDIA?"
    results_2 = answer_query(test_query_2)
    evaluate_retrieval(test_query_2, results_2)


    logging.info("\nRetrieval script finished.") 