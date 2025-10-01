"""Fraggle: A simple RAG API for building Q&A interfaces to your content."""

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from sse_starlette.sse import EventSourceResponse
import typer
import uvicorn
from any_llm import acompletion as llm_completion

__version__ = "0.1.0"

# Configuration from environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
SOURCE_JSON_PATH = os.environ.get("SOURCE_JSON_PATH", "source.json")
INDEX_PATH = os.environ.get("INDEX_PATH", "faiss_index")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "100"))
K_CONTEXT_DOCS = int(os.environ.get("K_CONTEXT_DOCS", "4"))
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
EMBEDDINGS_PROVIDER = os.environ.get("EMBEDDINGS_PROVIDER", "openai")
EMBEDDINGS_MODEL = os.environ.get("EMBEDDINGS_MODEL", "text-embedding-3-small")
UVICORN_HOST = os.environ.get("UVICORN_HOST", "0.0.0.0")
UVICORN_PORT = int(os.environ.get("UVICORN_PORT", "8000"))

app = FastAPI(title="Fraggle API")


def load_documents(source_path: str) -> List[Document]:
    """Load documents from a JSON file."""
    with open(source_path, "r") as f:
        data = json.load(f)

    documents = []
    for item in data:
        content = item.get("content", "")
        metadata = {k: v for k, v in item.items() if k != "content"}
        documents.append(Document(page_content=content, metadata=metadata))

    return documents


def create_index(
    source_path: str = SOURCE_JSON_PATH,
    index_path: str = INDEX_PATH,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> FAISS:
    """Create a FAISS index from source documents."""
    documents = load_documents(source_path)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    texts = text_splitter.split_documents(documents)

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local(index_path)

    return vectorstore


def load_index(index_path: str = INDEX_PATH) -> FAISS:
    """Load an existing FAISS index."""
    embeddings = get_embeddings()
    return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)


def get_embeddings():
    """Get embeddings instance based on configuration."""
    if EMBEDDINGS_PROVIDER == "openai":
        return OpenAIEmbeddings(model=EMBEDDINGS_MODEL)
    else:
        # Default to OpenAI for now
        return OpenAIEmbeddings(model=EMBEDDINGS_MODEL)


def get_llm_config():
    """Get LLM configuration for any-llm."""
    if LLM_PROVIDER == "anthropic":
        return {"provider": "anthropic", "api_key": ANTHROPIC_API_KEY}
    else:
        # Default to OpenAI
        return {"provider": "openai", "api_key": OPENAI_API_KEY}


async def ask_question(question: str, vectorstore: FAISS) -> str:
    """Ask a question and get an answer based on the indexed documents."""
    llm_config = get_llm_config()
    retriever = vectorstore.as_retriever(search_kwargs={"k": K_CONTEXT_DOCS})

    # Get relevant documents
    docs = retriever.invoke(question)

    # Build context from documents
    context = "\n\n".join([doc.page_content for doc in docs])

    # Create prompt
    prompt = f"""Answer the question based on the context below. If you cannot answer the question based on the context, say "I don't have enough information to answer that question."

Context:
{context}

Question: {question}

Answer:"""

    # Get completion
    messages = [{"role": "user", "content": prompt}]
    response = await llm_completion(
        model=LLM_MODEL,
        provider=llm_config["provider"],
        api_key=llm_config["api_key"],
        messages=messages
    )

    return response.choices[0].message.content


async def ask_question_stream(question: str, vectorstore: FAISS):
    """Ask a question and stream the answer."""
    llm_config = get_llm_config()
    retriever = vectorstore.as_retriever(search_kwargs={"k": K_CONTEXT_DOCS})

    # Get relevant documents
    docs = retriever.invoke(question)

    # Build context from documents
    context = "\n\n".join([doc.page_content for doc in docs])

    # Create prompt
    prompt = f"""Answer the question based on the context below. If you cannot answer the question based on the context, say "I don't have enough information to answer that question."

Context:
{context}

Question: {question}

Answer:"""

    # Stream completion
    messages = [{"role": "user", "content": prompt}]
    stream = await llm_completion(
        model=LLM_MODEL,
        provider=llm_config["provider"],
        api_key=llm_config["api_key"],
        messages=messages,
        stream=True
    )

    async for chunk in stream:
        if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


# Initialize vectorstore
vectorstore = None


@app.on_event("startup")
async def startup_event():
    """Load the index on startup."""
    global vectorstore
    try:
        vectorstore = load_index()
    except Exception as e:
        print(f"Warning: Could not load index: {e}")
        print("Run 'fraggle index' to create an index first.")


# Root endpoint will be overridden by static files if mounted


@app.post("/api/ask")
async def api_ask(request: Dict[str, Any]):
    """Non-streaming question answering endpoint."""
    question = request.get("question", "")
    if not question:
        return {"error": "No question provided"}

    if vectorstore is None:
        return {"error": "Index not loaded. Run 'fraggle index' first."}

    answer = await ask_question(question, vectorstore)
    return {"answer": answer}


@app.post("/api/stream")
async def api_stream(request: Dict[str, Any]):
    """Streaming question answering endpoint."""
    question = request.get("question", "")
    if not question:
        return {"error": "No question provided"}

    if vectorstore is None:
        return {"error": "Index not loaded. Run 'fraggle index' first."}

    async def event_generator():
        async for chunk in ask_question_stream(question, vectorstore):
            yield {"data": chunk}

    return EventSourceResponse(event_generator())


# CLI
cli = typer.Typer(help="Fraggle: A simple RAG API")


@cli.command()
def serve():
    """Start the Fraggle API server."""
    try:
        # Try to mount frontend directory if it exists
        if Path("frontend").exists():
            app.mount("/", StaticFiles(directory="frontend", html=True), name="static")
        elif Path("index.html").exists():
            app.mount("/", StaticFiles(directory=".", html=True), name="static")
    except RuntimeError:
        typer.echo("Static directory not found, front end will not be available")
    uvicorn.run(app, host=UVICORN_HOST, port=UVICORN_PORT)


@cli.command()
def index(
    source: str = SOURCE_JSON_PATH,
    output: str = INDEX_PATH,
):
    """Create a FAISS index from source documents."""
    typer.echo(f"Creating index from {source}...")
    create_index(source, output)
    typer.echo(f"Index created at {output}")


@cli.command()
def make_front_end(output: str = "frontend"):
    """Generate a simple HTML frontend."""
    frontend_dir = Path(output)
    frontend_dir.mkdir(exist_ok=True)

    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Fraggle Q&A</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            line-height: 1.6;
        }
        h1 { color: #333; }
        #question {
            width: 100%;
            padding: 12px;
            font-size: 16px;
            border: 2px solid #ddd;
            border-radius: 4px;
            box-sizing: border-box;
        }
        button {
            background: #007bff;
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 4px;
            font-size: 16px;
            cursor: pointer;
            margin-top: 10px;
        }
        button:hover { background: #0056b3; }
        #answer {
            margin-top: 20px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 4px;
            min-height: 100px;
        }
        .loading { opacity: 0.6; }
    </style>
</head>
<body>
    <h1>Ask a Question</h1>
    <input type="text" id="question" placeholder="Enter your question...">
    <button onclick="askQuestion()">Ask</button>
    <div id="answer"></div>

    <script>
        async function askQuestion() {
            const question = document.getElementById('question').value;
            const answerDiv = document.getElementById('answer');

            if (!question) return;

            answerDiv.innerHTML = 'Thinking...';
            answerDiv.classList.add('loading');

            try {
                const response = await fetch('/api/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question })
                });

                const data = await response.json();
                answerDiv.innerHTML = data.answer || data.error || 'No answer received';
            } catch (error) {
                answerDiv.innerHTML = 'Error: ' + error.message;
            } finally {
                answerDiv.classList.remove('loading');
            }
        }

        document.getElementById('question').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') askQuestion();
        });
    </script>
</body>
</html>"""

    (frontend_dir / "index.html").write_text(html_content)
    typer.echo(f"Frontend created at {frontend_dir}/index.html")
    typer.echo("Mount it in your app with: app.mount('/frontend', StaticFiles(directory='frontend'), name='frontend')")


@cli.command()
def make_dockerfile():
    """Generate a Dockerfile for deployment."""
    dockerfile_content = """FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Copy dependency files
COPY pyproject.toml .

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy application code
COPY . .

# Create index at build time (optional - comment out to create at runtime)
# RUN uv run fraggle index

EXPOSE 8000

CMD ["uv", "run", "fraggle", "serve"]
"""

    Path("Dockerfile").write_text(dockerfile_content)
    typer.echo("Dockerfile created")


def cli_wrapper():
    """Wrapper for the CLI."""
    cli()
