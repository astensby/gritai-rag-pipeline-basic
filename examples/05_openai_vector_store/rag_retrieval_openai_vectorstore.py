import os
import json
import time
from openai import OpenAI
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
try:
    client = OpenAI()  # Reads OPENAI_API_KEY from environment variable
except Exception as e:
    logging.error(f"Failed to initialize OpenAI client. Ensure OPENAI_API_KEY is set: {e}")
    client = None

CONFIG_PATH = "examples/05_openai_vector_store/vectorstore_config.json"

# --- Load Configuration ---
def load_config(config_path: str = CONFIG_PATH) -> Dict[str, Any]:
    """
    Loads the vector store and assistant configuration from JSON file.

    Args:
        config_path: Path to the configuration file.

    Returns:
        Configuration dictionary, or None if file doesn't exist or fails to load.
    """
    if not os.path.exists(config_path):
        logging.error(f"Configuration file not found at '{config_path}'")
        logging.error("Please run rag_pipeline_openai_vectorstore.py first to build the knowledge base.")
        return None

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logging.info(f"Successfully loaded configuration from '{config_path}'")
        logging.info(f"Vector Store ID: {config.get('vector_store_id')}")
        logging.info(f"Assistant ID: {config.get('assistant_id')}")
        return config
    except Exception as e:
        logging.error(f"Failed to load configuration from '{config_path}': {e}")
        return None

# --- Query Function ---
def query_knowledge_base(query: str, assistant_id: str, max_tokens: int = 1000) -> Dict[str, Any]:
    """
    Queries the knowledge base using the OpenAI Assistant with file_search.

    Args:
        query: The user's query string.
        assistant_id: The ID of the assistant to use.
        max_tokens: Maximum tokens for the response.

    Returns:
        Dictionary containing the response and metadata, or None on failure.
    """
    if not client:
        logging.error("OpenAI client not initialized. Cannot query knowledge base.")
        return None

    try:
        logging.info(f"Querying knowledge base with: '{query}'")

        # Create a thread for the conversation
        thread = client.beta.threads.create()
        logging.info(f"Created thread: {thread.id}")

        # Add the user's message to the thread
        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=query
        )

        # Run the assistant
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant_id
        )

        # Poll for completion
        logging.info("Waiting for assistant response...")
        while run.status in ["queued", "in_progress"]:
            time.sleep(1)
            run = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )

        logging.info(f"Run completed with status: {run.status}")

        if run.status == "completed":
            # Retrieve the assistant's messages
            messages = client.beta.threads.messages.list(
                thread_id=thread.id,
                order="desc",
                limit=1
            )

            if messages.data:
                assistant_message = messages.data[0]

                # Extract text content and annotations
                response_text = ""
                citations = []

                for content_block in assistant_message.content:
                    if content_block.type == "text":
                        response_text = content_block.text.value

                        # Extract file citations from annotations
                        if hasattr(content_block.text, 'annotations'):
                            for annotation in content_block.text.annotations:
                                if hasattr(annotation, 'file_citation'):
                                    citations.append({
                                        'file_id': annotation.file_citation.file_id,
                                        'quote': annotation.file_citation.quote if hasattr(annotation.file_citation, 'quote') else None
                                    })

                result = {
                    'query': query,
                    'response': response_text,
                    'citations': citations,
                    'thread_id': thread.id,
                    'run_id': run.id,
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                }

                logging.info("Successfully retrieved response from assistant.")
                return result
            else:
                logging.error("No messages found in the thread.")
                return None
        else:
            logging.error(f"Run failed with status: {run.status}")
            if hasattr(run, 'last_error'):
                logging.error(f"Error details: {run.last_error}")
            return None

    except Exception as e:
        logging.error(f"Failed to query knowledge base: {e}")
        return None

# --- Evaluation Function ---
def evaluate_response(result: Dict[str, Any]):
    """
    Prints the query result in a formatted way for evaluation.

    Args:
        result: The result dictionary from query_knowledge_base.
    """
    if not result:
        print("\n--- Evaluation ---")
        print("No result to evaluate.")
        print("--- End Evaluation ---\n")
        return

    print("\n" + "="*80)
    print("QUERY RESULTS")
    print("="*80)

    print(f"\nQuery: {result['query']}")
    print(f"Timestamp: {result['timestamp']}")

    print(f"\n{'-'*80}")
    print("RESPONSE:")
    print(f"{'-'*80}")
    print(result['response'])

    if result.get('citations'):
        print(f"\n{'-'*80}")
        print("SOURCES (File Citations):")
        print(f"{'-'*80}")
        for i, citation in enumerate(result['citations'], 1):
            print(f"{i}. File ID: {citation['file_id']}")
            if citation.get('quote'):
                print(f"   Quote: {citation['quote']}")

    print(f"\n{'-'*80}")
    print("METADATA:")
    print(f"{'-'*80}")
    print(f"Thread ID: {result['thread_id']}")
    print(f"Run ID: {result['run_id']}")

    print("\n" + "="*80 + "\n")

# --- Interactive Query Mode ---
def interactive_mode(assistant_id: str):
    """
    Runs an interactive query session where users can ask multiple questions.

    Args:
        assistant_id: The ID of the assistant to use.
    """
    print("\n" + "="*80)
    print("INTERACTIVE QUERY MODE")
    print("="*80)
    print("Type your questions below. Type 'exit' or 'quit' to end the session.")
    print("="*80 + "\n")

    while True:
        try:
            query = input("\nYour question: ").strip()

            if query.lower() in ['exit', 'quit', 'q']:
                print("\nExiting interactive mode. Goodbye!")
                break

            if not query:
                print("Please enter a question.")
                continue

            result = query_knowledge_base(query, assistant_id)
            evaluate_response(result)

        except KeyboardInterrupt:
            print("\n\nExiting interactive mode. Goodbye!")
            break
        except Exception as e:
            logging.error(f"Error in interactive mode: {e}")

# --- Batch Query Function ---
def batch_query(queries: List[str], assistant_id: str) -> List[Dict[str, Any]]:
    """
    Runs multiple queries in batch mode.

    Args:
        queries: List of query strings.
        assistant_id: The ID of the assistant to use.

    Returns:
        List of result dictionaries.
    """
    results = []
    for i, query in enumerate(queries, 1):
        logging.info(f"\nProcessing query {i}/{len(queries)}")
        result = query_knowledge_base(query, assistant_id)
        if result:
            results.append(result)
            evaluate_response(result)

    return results

# --- Main Execution ---
if __name__ == "__main__":
    import sys

    # Load configuration
    config = load_config()
    if not config:
        logging.error("Cannot proceed without configuration. Exiting.")
        exit(1)

    assistant_id = config.get('assistant_id')
    if not assistant_id:
        logging.error("Assistant ID not found in configuration. Exiting.")
        exit(1)

    # Check for command line arguments
    if len(sys.argv) > 1:
        # Check if interactive mode is requested
        if sys.argv[1] == "--interactive" or sys.argv[1] == "-i":
            interactive_mode(assistant_id)
        else:
            # Treat arguments as queries
            queries = sys.argv[1:]
            logging.info(f"Running batch query mode with {len(queries)} queries.")
            batch_query(queries, assistant_id)
    else:
        # Default: Run some test queries
        logging.info("\n--- Running Test Queries ---")

        test_queries = [
            "What are the main topics covered in these documents?",
            "Tell me about space exploration.",
            "What information is available about animals or marine mammals?",
        ]

        batch_query(test_queries, assistant_id)

        print("\n" + "="*80)
        print("TIP: Run with --interactive flag for interactive query mode:")
        print(f"  python {sys.argv[0]} --interactive")
        print("\nOr provide queries as arguments:")
        print(f"  python {sys.argv[0]} \"Your question here\"")
        print("="*80 + "\n")
