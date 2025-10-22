# Fraggle

The simplest RAG API for Python. Build a question and answer interface to your own content in minutes.

```bash
pip install fraggle
fraggle index
fraggle serve
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Fraggle?"}'
```

Fraggle is the successor to my badly named [microllama](https://github.com/tomdyson/microllama) project, modernised with:
- Provider-agnostic LLM support via [any-llm](https://github.com/mozilla-ai/any-llm)
- Modern LangChain (0.3+)
- Simple deployment and usage

## Installation

```bash
pip install fraggle
```

## Quick Start

1. **Prepare your content** in a `source.json` file:

```json
[
  {
    "content": "Fraggle is a RAG API for building Q&A interfaces.",
    "title": "What is Fraggle",
    "url": "https://example.com/docs"
  },
  {
    "content": "You can use OpenAI or Anthropic models with Fraggle.",
    "title": "Supported Models"
  }
]
```

2. **Set your API key**:

```bash
export OPENAI_API_KEY="your-key-here"
# or
export ANTHROPIC_API_KEY="your-key-here"
```

3. **Create an index of embeddings**:

```bash
fraggle index
```

4. **Start the API server**:

```bash
fraggle serve
```

5. **Query your content**:

```bash
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Fraggle?"}'
```

## Configuration

Configure Fraggle with environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SOURCE_JSON_PATH` | `source.json` | Path to your content JSON file |
| `INDEX_PATH` | `faiss_index` | Path to store/load the FAISS index |
| `LLM_PROVIDER` | `openai` | LLM provider: `openai` or `anthropic` |
| `LLM_MODEL` | `gpt-4o-mini` | Model ID (e.g., `gpt-4o`, `claude-3-5-sonnet-20241022`) |
| `EMBEDDINGS_PROVIDER` | `openai` | Embeddings provider |
| `EMBEDDINGS_MODEL` | `text-embedding-3-small` | Embeddings model |
| `CHUNK_SIZE` | `1000` | Text chunk size for indexing |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `K_CONTEXT_DOCS` | `4` | Number of documents to retrieve |
| `UVICORN_HOST` | `0.0.0.0` | Host to bind the server to |
| `UVICORN_PORT` | `8000` | Port to bind the server to |

### Using Anthropic Claude

```bash
export ANTHROPIC_API_KEY="your-key-here"
export LLM_PROVIDER="anthropic"
export LLM_MODEL="claude-3-5-sonnet-20241022"
fraggle serve
```

## CLI Commands

### `fraggle serve`

Start the API server. Automatically serves the frontend at `/` if the `frontend/` directory exists.

### `fraggle index`

Create a FAISS index from your source documents.

Options:
- `--source`: Path to source JSON file (default: source.json)
- `--output`: Path to save the index (default: faiss_index)

### `fraggle make-front-end`

Generate a simple HTML frontend for your Q&A interface.

Options:
- `--output`: Directory for frontend files (default: frontend)

### `fraggle make-dockerfile`

Generate a Dockerfile for containerized deployment.

## API Endpoints

### POST `/api/ask`

Non-streaming question answering.

Request:
```json
{
  "question": "What is Fraggle?"
}
```

Response:
```json
{
  "answer": "Fraggle is a RAG API for building Q&A interfaces to your content."
}
```

### POST `/api/stream`

Streaming question answering (Server-Sent Events).

Request:
```json
{
  "question": "What is Fraggle?"
}
```

Response: SSE stream of text chunks.

## Deployment

### Docker

```bash
fraggle make-dockerfile
docker build -t fraggle .
docker run -p 8000:8000 -e OPENAI_API_KEY=your-key fraggle
```

### Pre-building the Index

For faster startup, create the index at build time by uncommenting the `RUN fraggle index` line in the Dockerfile.

## Development

```bash
# Clone the repo
git clone https://github.com/tomdyson/fraggle.git
cd fraggle

# Install with uv
uv sync

# Run in development mode
uv run fraggle serve
```

## Comparison with microllama

Fraggle improves on microllama by:
- Provider-agnostic: Use any LLM via any-llm
- Modern dependencies: LangChain 0.3+
- Better naming: Focus on RAG, not LLMs
- Same simple API and deployment story

## License

MIT

## Publishing a New Version

Fraggle uses GitHub Actions to automatically publish to PyPI when you push a version tag:

```bash
# 1. Update the version in pyproject.toml
# 2. Commit and tag the release
git add pyproject.toml
git commit -m "Bump version to 0.1.x"
git tag v0.1.x
git push && git push --tags
```

The GitHub Actions workflow will automatically build and publish to PyPI.

**First-time setup:** Configure PyPI Trusted Publishing at https://pypi.org/manage/account/publishing/ with:
- PyPI Project Name: `fraggle`
- Owner: `tomdyson`
- Repository: `fraggle`
- Workflow: `publish.yml`
- Environment: `pypi`
