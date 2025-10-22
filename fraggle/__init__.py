"""Fraggle: A simple RAG API for building Q&A interfaces to your content."""

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from sse_starlette.sse import EventSourceResponse
import typer
import uvicorn
from any_llm import acompletion as llm_completion

__version__ = "0.1.5"

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
    """Ask a question and stream the answer with sources."""
    llm_config = get_llm_config()
    retriever = vectorstore.as_retriever(search_kwargs={"k": K_CONTEXT_DOCS})

    # Get relevant documents
    docs = retriever.invoke(question)

    # Extract sources from document metadata
    import json
    sources = []
    for doc in docs:
        title = doc.metadata.get("title", "Untitled")
        url = doc.metadata.get("url", "")
        sources.append([title, url])

    # Send sources first
    yield f"SOURCES::{json.dumps(sources)}"

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

    # Signal stream complete
    yield "stream-complete"


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


@app.get("/api/stream")
async def api_stream(q: str = ""):
    """Streaming question answering endpoint."""
    if not q:
        return {"error": "No question provided"}

    if vectorstore is None:
        return {"error": "Index not loaded. Run 'fraggle index' first."}

    async def event_generator():
        async for chunk in ask_question_stream(q, vectorstore):
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
<html lang="en">

<head>
    <title>Fraggle</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta charset="utf-8" />
    <link rel="stylesheet" href="https://unpkg.com/tachyons/css/tachyons.min.css">
    <script src="https://cdn.jsdelivr.net/npm/vue/dist/vue.min.js"></script>
</head>

<body class="w-100 avenir black-80 bg-yellow">
    <div id="app" class="mw6 center">
        <form @submit="formSubmit" class="pa2 black-80">
            <div class="measure">
                <h1 class="f-subheadline-ns f1 lh-solid mb4 near-black">Fraggle</h1>
                <label for="question" class="f4 db mb3">Your question:</label>
                <input id="question" class="input-reset f4 ba b--black-20 pa2 mb1 db w-100 br2" type="text"
                    v-model.lazy="question">
                <input id="submit" class="dim mt3 pointer ph3 pv2 input-reset ba b--black br2 bg-transparent f4 mb2"
                    type="submit" :value="button_text">
                <p v-if="sources.length" class="mt2 georgia i mid-gray f5 mb0">Sources:
                <ul>
                    <li v-for="source in sources" class="georgia i mid-gray f5 lh-copy">
                        <a v-if="source[1]" :href="source[1]" class="mid-gray dim">{{ source[0] }}</a>
                        <span v-else>{{ source[0] }}</span>
                    </li>
                </ul>
                <p class="mt1 georgia f4 lh-copy" v-text="answer"></p>
                </p>
            </div>
        </form>
    </div>

    <script>
        var app = new Vue({
            el: '#app',
            data: {
                question: '',
                answer: '',
                sources: [],
                prompt_messages: [],
                button_text: 'Tell me',
                sseClient: null,
            },
            mounted() {
                this.connectToSSE();
            },
            beforeDestroy() {
                if (this.sseClient) {
                    this.sseClient.close();
                }
            },
            methods: {
                formSubmit(e) {
                    e.preventDefault();
                    app.button_text = "Checking sources...";
                    app.sources = [];
                    app.prompt_messages = [];
                    app.answer = "";
                    streaming_api_url = "/api/stream?q=" + app.question;
                    this.connectToSSE(streaming_api_url);
                },
                connectToSSE(streamURL) {
                    this.sseClient = new EventSource(streamURL);
                    console.log('SSE connection opened to ' + streamURL);

                    this.sseClient.addEventListener('message', (event) => {
                        // if the event starts with "SOURCES::" then it's a list of sources
                        if (event.data.startsWith('SOURCES::')) {
                            this.sources = JSON.parse(event.data.split('::')[1]);
                            console.log('Sources updated');
                            app.button_text = "Working out an answer...";
                            return;
                        }
                        // if the event starts with "PROMPT::" then the prompt messages are included
                        if (event.data.startsWith('PROMPT::')) {
                            this.prompt_messages = JSON.parse(event.data.split('::')[1]);
                            console.log('Prompt messages updated');
                            return;
                        }
                        // if the event is "stream-complete", disconnect
                        if (event.data === 'stream-complete') {
                            console.log('Stream complete');
                            this.sseClient.close();
                            app.button_text = "Tell me";
                            return;
                        }
                        this.answer += event.data;
                    });

                    this.sseClient.addEventListener('error', (event) => {
                        if (event.target.readyState === EventSource.CLOSED) {
                            console.log('SSE connection closed');
                        } else if (event.target.readyState === EventSource.CONNECTING) {
                            console.log('SSE connection reconnecting');
                        }
                    });
                },
            },
        });
    </script>
</body>

</html>"""

    (frontend_dir / "index.html").write_text(html_content)
    typer.echo(f"Frontend created at {frontend_dir}/index.html")
    typer.echo("Run 'fraggle serve' to start the server with the frontend")


@cli.command()
def rock():
    """Dance your cares away!"""
    typer.echo("Dance your cares away, worry's for another day")


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
