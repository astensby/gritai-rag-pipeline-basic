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

DATA_DIR = "data"
CONFIG_PATH = "examples/05_openai_vector_store/vectorstore_config.json"
VECTOR_STORE_NAME = "GritAI RAG Knowledge Base"

# --- 1. Ingestion Step ---
def ingest_documents(data_dir: str) -> List[str]:
    """
    Collects file paths from the specified directory for upload to OpenAI.

    Args:
        data_dir: The path to the directory containing files.

    Returns:
        A list of file paths. Returns empty list if dir not found.
    """
    file_paths = []
    if not os.path.isdir(data_dir):
        logging.warning(f"Data directory '{data_dir}' not found.")
        return file_paths

    try:
        for filename in os.listdir(data_dir):
            # OpenAI Vector Stores support various file types
            if filename.endswith((".txt", ".pdf", ".md", ".doc", ".docx")):
                file_path = os.path.join(data_dir, filename)
                file_paths.append(file_path)
                logging.info(f"Found file for ingestion: '{filename}'")
    except Exception as e:
        logging.error(f"Error listing directory '{data_dir}': {e}")

    return file_paths

# --- 2. Create Vector Store ---
def create_vector_store(name: str = VECTOR_STORE_NAME) -> str:
    """
    Creates an OpenAI Vector Store or retrieves existing one.

    Args:
        name: The name for the vector store.

    Returns:
        The vector store ID, or None on failure.
    """
    if not client:
        logging.error("OpenAI client not initialized. Cannot create vector store.")
        return None

    try:
        # Create a new vector store
        vector_store = client.vector_stores.create(
            name=name
        )
        logging.info(f"Successfully created vector store '{name}' with ID: {vector_store.id}")
        return vector_store.id
    except Exception as e:
        logging.error(f"Failed to create vector store: {e}")
        return None

# --- 3. Upload Files to Vector Store ---
def upload_files_to_vector_store(file_paths: List[str], vector_store_id: str) -> List[str]:
    """
    Uploads files to the OpenAI Vector Store using batch upload.

    Args:
        file_paths: List of file paths to upload.
        vector_store_id: The ID of the vector store.

    Returns:
        List of file IDs that were successfully uploaded.
    """
    if not client:
        logging.error("OpenAI client not initialized. Cannot upload files.")
        return []

    if not file_paths:
        logging.warning("No files provided for upload.")
        return []

    file_ids = []
    file_streams = []

    try:
        # Open all files for batch upload
        for file_path in file_paths:
            try:
                file_stream = open(file_path, "rb")
                file_streams.append(file_stream)
            except Exception as e:
                logging.error(f"Failed to open file '{file_path}': {e}")

        if not file_streams:
            logging.error("No files could be opened for upload.")
            return []

        # Batch upload files to vector store
        logging.info(f"Uploading {len(file_streams)} files to vector store...")
        file_batch = client.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vector_store_id,
            files=file_streams
        )

        # Check status
        logging.info(f"Upload batch status: {file_batch.status}")
        logging.info(f"File counts - Total: {file_batch.file_counts.total}, "
                    f"Completed: {file_batch.file_counts.completed}, "
                    f"Failed: {file_batch.file_counts.failed}")

        # Get the file IDs from the batch
        if file_batch.file_counts.completed > 0:
            # List files in the vector store to get their IDs
            vector_store_files = client.vector_stores.files.list(
                vector_store_id=vector_store_id
            )
            file_ids = [f.id for f in vector_store_files.data]
            logging.info(f"Successfully uploaded {len(file_ids)} files to vector store.")

        return file_ids

    except Exception as e:
        logging.error(f"Failed to upload files to vector store: {e}")
        return []
    finally:
        # Close all file streams
        for stream in file_streams:
            try:
                stream.close()
            except:
                pass

# --- 4. Create Assistant with File Search ---
def create_assistant(vector_store_id: str, name: str = "RAG Assistant") -> str:
    """
    Creates an OpenAI Assistant configured with file_search tool.

    Args:
        vector_store_id: The ID of the vector store to attach.
        name: The name for the assistant.

    Returns:
        The assistant ID, or None on failure.
    """
    if not client:
        logging.error("OpenAI client not initialized. Cannot create assistant.")
        return None

    try:
        assistant = client.beta.assistants.create(
            name=name,
            instructions=(
                "You are a helpful assistant that answers questions based on the "
                "provided knowledge base. Use the file_search tool to find relevant "
                "information from the documents. Always cite your sources by mentioning "
                "the document filenames when providing answers."
            ),
            model="gpt-4o-mini",  # Use a capable model
            tools=[{"type": "file_search"}],
            tool_resources={
                "file_search": {
                    "vector_store_ids": [vector_store_id]
                }
            }
        )
        logging.info(f"Successfully created assistant '{name}' with ID: {assistant.id}")
        return assistant.id
    except Exception as e:
        logging.error(f"Failed to create assistant: {e}")
        return None

# --- 5. Save Configuration ---
def save_config(vector_store_id: str, assistant_id: str, file_ids: List[str], output_path: str = CONFIG_PATH):
    """
    Saves the vector store and assistant configuration to a JSON file.

    Args:
        vector_store_id: The vector store ID.
        assistant_id: The assistant ID.
        file_ids: List of uploaded file IDs.
        output_path: Path to save the configuration.
    """
    config = {
        "vector_store_id": vector_store_id,
        "assistant_id": assistant_id,
        "file_ids": file_ids,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4)
        logging.info(f"Successfully saved configuration to '{output_path}'")
    except Exception as e:
        logging.error(f"Failed to save configuration to '{output_path}': {e}")

# --- 6. Load Configuration ---
def load_config(config_path: str = CONFIG_PATH) -> Dict[str, Any]:
    """
    Loads the vector store and assistant configuration from JSON file.

    Args:
        config_path: Path to the configuration file.

    Returns:
        Configuration dictionary, or None if file doesn't exist or fails to load.
    """
    if not os.path.exists(config_path):
        logging.warning(f"Configuration file not found at '{config_path}'")
        return None

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logging.info(f"Successfully loaded configuration from '{config_path}'")
        return config
    except Exception as e:
        logging.error(f"Failed to load configuration from '{config_path}': {e}")
        return None

# --- Combined Pipeline ---
def build_knowledge_base(data_dir: str = DATA_DIR, force_rebuild: bool = False):
    """
    Runs the complete pipeline: Ingest -> Upload to Vector Store -> Create Assistant -> Save Config.

    Args:
        data_dir: Directory containing the source documents.
        force_rebuild: If True, creates new vector store even if config exists.
    """
    logging.info("Starting OpenAI Vector Store knowledge base build process...")

    # Check if configuration already exists
    if not force_rebuild:
        existing_config = load_config()
        if existing_config:
            logging.warning("Configuration file already exists. Use force_rebuild=True to rebuild.")
            logging.info(f"Existing Vector Store ID: {existing_config.get('vector_store_id')}")
            logging.info(f"Existing Assistant ID: {existing_config.get('assistant_id')}")
            return

    # Step 1: Ingest documents
    file_paths = ingest_documents(data_dir)
    if not file_paths:
        logging.warning("No files found for ingestion. Stopping knowledge base build.")
        return

    # Step 2: Create vector store
    vector_store_id = create_vector_store()
    if not vector_store_id:
        logging.error("Failed to create vector store. Stopping knowledge base build.")
        return

    # Step 3: Upload files to vector store
    file_ids = upload_files_to_vector_store(file_paths, vector_store_id)
    if not file_ids:
        logging.error("Failed to upload files to vector store. Stopping knowledge base build.")
        return

    # Step 4: Create assistant with file_search
    assistant_id = create_assistant(vector_store_id)
    if not assistant_id:
        logging.error("Failed to create assistant. Stopping knowledge base build.")
        return

    # Step 5: Save configuration
    save_config(vector_store_id, assistant_id, file_ids)

    logging.info("Knowledge base build process completed successfully!")
    logging.info(f"Vector Store ID: {vector_store_id}")
    logging.info(f"Assistant ID: {assistant_id}")
    logging.info(f"Uploaded {len(file_ids)} files")

# --- Cleanup Function ---
def cleanup_resources(config_path: str = CONFIG_PATH):
    """
    Deletes the vector store and assistant to clean up resources.
    WARNING: This will permanently delete your knowledge base!

    Args:
        config_path: Path to the configuration file.
    """
    if not client:
        logging.error("OpenAI client not initialized.")
        return

    config = load_config(config_path)
    if not config:
        logging.warning("No configuration found. Nothing to clean up.")
        return

    try:
        # Delete assistant
        if config.get('assistant_id'):
            client.beta.assistants.delete(config['assistant_id'])
            logging.info(f"Deleted assistant: {config['assistant_id']}")

        # Delete vector store (this also deletes associated files)
        if config.get('vector_store_id'):
            client.vector_stores.delete(config['vector_store_id'])
            logging.info(f"Deleted vector store: {config['vector_store_id']}")

        # Remove config file
        os.remove(config_path)
        logging.info(f"Removed configuration file: {config_path}")

        logging.info("Cleanup completed successfully!")
    except Exception as e:
        logging.error(f"Error during cleanup: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    import sys

    # Check for command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--cleanup":
        logging.warning("Running cleanup - this will DELETE your vector store and assistant!")
        response = input("Are you sure you want to continue? (yes/no): ")
        if response.lower() == "yes":
            cleanup_resources()
        else:
            logging.info("Cleanup cancelled.")
    elif len(sys.argv) > 1 and sys.argv[1] == "--rebuild":
        logging.info("Forcing rebuild of knowledge base...")
        build_knowledge_base(force_rebuild=True)
    else:
        # Normal build process
        if client and os.getenv("OPENAI_API_KEY"):
            build_knowledge_base()
        else:
            logging.error("OPENAI_API_KEY not found in environment variables.")
            logging.error("Please set the OPENAI_API_KEY environment variable and run again.")
