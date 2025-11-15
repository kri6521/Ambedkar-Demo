import os
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_chroma import Chroma

try:
    from langchain_ollama import OllamaLLM
    OLLAMA_AVAILABLE = True
except ImportError:
    try:
        from langchain_community.llms import Ollama as OllamaLLM
        OLLAMA_AVAILABLE = True
    except ImportError:
        OLLAMA_AVAILABLE = False
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Configuration
SPEECH_PATH = Path("speech.txt")
PERSIST_DIR = "chroma_db"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "mistral"  # name that appears in `ollama ls` (e.g., 'mistral')


def build_vectorstore(persist_directory: str = PERSIST_DIR) -> Chroma:
    """Build or load a Chroma vectorstore persisted to disk."""
    # Create embeddings object (this will download the sentence-transformers model locally)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # If a persisted chroma DB exists, load it
    if os.path.isdir(persist_directory) and any(Path(persist_directory).iterdir()):
        print(f"Loading existing Chroma DB from '{persist_directory}'...")
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        return db

    # Otherwise, load the speech file and create the DB
    if not SPEECH_PATH.exists():
        raise FileNotFoundError(f"speech.txt not found at: {SPEECH_PATH.resolve()}")

    print("Loading speech.txt and splitting into chunks...")
    loader = TextLoader(str(SPEECH_PATH), encoding="utf-8")
    docs = loader.load()

    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )

    texts = splitter.split_documents(docs)
    print(f"Created {len(texts)} chunks.")

    print("Creating embeddings and building Chroma DB (this may take a moment)...")
    db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)
    # Note: persist() is deprecated in newer ChromaDB versions - persistence is automatic
    print(f"Chroma DB persisted to '{persist_directory}'.")
    return db


def build_qa_chain(db: Chroma):
    """Create a retrieval QA chain using Ollama as the LLM (LCEL pattern)."""
    if not OLLAMA_AVAILABLE:
        raise ImportError(
            "Ollama is not available. Please install it with: pip install langchain-ollama"
        )
    
    # Create Ollama LLM wrapper
    try:
        llm = OllamaLLM(model=OLLAMA_MODEL, temperature=0.0)
    except Exception as e:
        raise ConnectionError(
            f"Failed to connect to Ollama. Make sure Ollama is running on localhost:11434.\n"
            f"Start Ollama service and ensure the model '{OLLAMA_MODEL}' is available.\n"
            f"Original error: {str(e)}"
        ) from e

    # Create retriever from Chroma
    retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 4})

    # Create prompt template
    prompt = ChatPromptTemplate.from_template(
        """Answer the following question based only on the provided context:

Context: {context}

Question: {input}

Answer:"""
    )

    # Create the chain using LCEL (LangChain Expression Language)
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    qa = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return qa


def print_relevant_chunks(db: Chroma, query: str, k: int = 3):
    """Utility to fetch and print the top-k relevant chunks and their similarity scores."""
    docs_and_scores = db.similarity_search_with_score(query, k=k)
    print("\nRelevant chunks from the speech (top {}):\n".format(k))
    for i, (doc, score) in enumerate(docs_and_scores, start=1):
        print(f"[{i}] score={score:.4f} | chunk: {doc.page_content.strip()}\n")


def main():
    print("Ambedkar RAG Demo — starting up")
    db = build_vectorstore()
    qa_chain = build_qa_chain(db)

    print("\nReady. Ask questions about the speech. Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            query = input("Question> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        print_relevant_chunks(db, query, k=3)

        try:
            answer = qa_chain.invoke(query)
            print("Answer:\n")
            print(answer)
            print("\n" + ("-" * 60) + "\n")
        except ConnectionError as e:
            print(f"\nError: {e}\n")
            print("Please make sure Ollama is running. You can start it by running 'ollama serve' in a separate terminal.\n")
            print("Also verify that the model is available with: ollama ls\n")
        except Exception as e:
            print(f"\nError generating answer: {e}\n")
            print("\n" + ("-" * 60) + "\n")


if __name__ == "__main__":
    main()

