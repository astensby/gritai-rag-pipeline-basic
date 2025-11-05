# OpenAI Vector Store RAG Pipeline

This example demonstrates using OpenAI's managed Vector Store API with the Assistants API for RAG (Retrieval Augmented Generation) applications.

## Overview

Unlike other examples in this repository that handle chunking and embedding manually, this pipeline uses OpenAI's fully managed solution where:

- **OpenAI handles chunking**: Your documents are automatically split into optimal chunks
- **OpenAI handles embedding**: Embeddings are generated and stored automatically
- **OpenAI handles search**: Vector similarity search is performed by OpenAI's infrastructure
- **Automatic citations**: Responses include source citations from your documents

## Architecture

This pipeline uses two main components:

1. **Vector Stores**: OpenAI's managed vector database for storing and searching document embeddings
2. **Assistants API**: AI assistants configured with the `file_search` tool to query the vector store

## Features

- **Fully Managed**: No need to handle chunking, embedding, or vector search
- **Multiple File Formats**: Supports `.txt`, `.pdf`, `.md`, `.doc`, `.docx`
- **Batch Upload**: Efficient batch uploading of documents
- **Source Citations**: Automatic citation of source documents in responses
- **Interactive Mode**: Query your knowledge base interactively
- **Configuration Persistence**: Stores vector store and assistant IDs for reuse

## Prerequisites

- Python 3.8+
- OpenAI API key with access to:
  - Vector Stores API
  - Assistants API
  - GPT-4o-mini (or compatible model)

## Setup

1. **Set your OpenAI API key:**
   ```bash
   export OPENAI_API_KEY="your_openai_api_key_here"
   ```

2. **Prepare your documents:**
   Place your documents in the `data/` directory (two levels up from this example)

## Usage

### Building the Knowledge Base

Run the build script to upload your documents and create a vector store:

```bash
python examples/05_openai_vector_store/rag_pipeline_openai_vectorstore.py
```

This will:
1. Scan the `data/` directory for supported files
2. Create a new OpenAI Vector Store
3. Upload all files in batch
4. Create an Assistant configured with file_search
5. Save configuration to `vectorstore_config.json`

**Options:**

```bash
# Force rebuild (overwrites existing configuration)
python examples/05_openai_vector_store/rag_pipeline_openai_vectorstore.py --rebuild

# Clean up resources (deletes vector store and assistant)
python examples/05_openai_vector_store/rag_pipeline_openai_vectorstore.py --cleanup
```

### Querying the Knowledge Base

After building, query your knowledge base using the retrieval script:

```bash
# Run with default test queries
python examples/05_openai_vector_store/rag_retrieval_openai_vectorstore.py

# Interactive mode - great for exploration
python examples/05_openai_vector_store/rag_retrieval_openai_vectorstore.py --interactive

# Single query
python examples/05_openai_vector_store/rag_retrieval_openai_vectorstore.py "What is machine learning?"

# Multiple queries
python examples/05_openai_vector_store/rag_retrieval_openai_vectorstore.py "Query 1" "Query 2"
```

## Configuration File

The `vectorstore_config.json` file stores:

```json
{
    "vector_store_id": "vs_xxx",
    "assistant_id": "asst_xxx",
    "file_ids": ["file_xxx", "file_yyy"],
    "created_at": "2025-11-05 12:00:00"
}
```

This configuration allows the retrieval script to access your pre-built knowledge base without rebuilding.

## How It Works

### Build Process

1. **File Discovery**: Scans `data/` directory for supported file types
2. **Vector Store Creation**: Creates a new vector store via OpenAI API
3. **Batch Upload**: Uploads all files in a single batch operation
4. **Processing**: OpenAI automatically chunks and embeds the documents
5. **Assistant Creation**: Creates an assistant linked to the vector store
6. **Configuration Save**: Persists IDs for later retrieval

### Query Process

1. **Configuration Load**: Loads vector store and assistant IDs
2. **Thread Creation**: Creates a new conversation thread
3. **Message Submission**: Adds user query to the thread
4. **Assistant Run**: Assistant searches vector store and generates response
5. **Citation Extraction**: Extracts file citations from the response
6. **Result Display**: Shows response with source citations

## Comparison with Other Examples

| Feature | Standard Pipelines | OpenAI Vector Store |
|---------|-------------------|---------------------|
| Chunking Control | ✓ Full control | ✗ OpenAI managed |
| Embedding Provider | Multiple options | OpenAI only |
| Storage | Local/PostgreSQL | OpenAI managed |
| Search Method | Cosine similarity | OpenAI ANN |
| Cost | Pay per API call | Storage + usage fees |
| Complexity | Higher | Lower |
| Citations | Manual | Automatic |
| Scalability | DIY | Managed |

## Cost Considerations

OpenAI charges for:
- **Vector Store Storage**: Per GB per day
- **File Processing**: One-time fee per file
- **Assistant API Usage**: Per request
- **Model Usage**: GPT-4o-mini tokens

Monitor your usage in the [OpenAI Dashboard](https://platform.openai.com/usage).

## Best Practices

1. **Cleanup Unused Resources**: Use `--cleanup` to delete old vector stores
2. **Batch Operations**: Upload multiple files at once for efficiency
3. **Monitor Costs**: Check OpenAI dashboard regularly
4. **File Formats**: Use supported formats for best results
5. **Reuse Configuration**: Keep `vectorstore_config.json` to avoid rebuilds

## Troubleshooting

### "Configuration file not found"
Run the build script first:
```bash
python examples/05_openai_vector_store/rag_pipeline_openai_vectorstore.py
```

### "OpenAI client not initialized"
Ensure `OPENAI_API_KEY` is set:
```bash
export OPENAI_API_KEY="your_key"
```

### "File upload failed"
Check:
- File format is supported (`.txt`, `.pdf`, `.md`, `.doc`, `.docx`)
- File is not corrupted
- File size is within OpenAI's limits

### Long processing time
Large document sets take time to process. The script polls automatically until completion.

## Further Reading

- [OpenAI Vector Stores Documentation](https://platform.openai.com/docs/assistants/tools/file-search)
- [OpenAI Assistants API Documentation](https://platform.openai.com/docs/assistants/overview)
- [File Search Tool Guide](https://platform.openai.com/docs/assistants/tools/file-search)

## License

Part of the GritAI RAG & MCP Training Courses educational materials.
