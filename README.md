# Ambedkar-Demo

# Ambedkar RAG Demo


A simple command-line Q&A system that ingests a short speech by Dr. B.R. Ambedkar and answers questions based solely on that content. Built with LangChain, ChromaDB, HuggingFace embeddings, and Ollama (Mistral 7B).


## What you get


- `main.py` — Python script that builds a local RAG pipeline and exposes a simple CLI for asking questions.
- `speech.txt` — Provided speech excerpt used as the knowledge source.
- `requirements.txt` — Python dependencies.


## Prerequisites


1. **Python 3.8+**
2. **Conda** (recommended) — to create the virtual environment.
3. **Ollama** installed and running locally with the Mistral model pulled.


### Install Ollama and pull Mistral


```bash
# Install Ollama (macOS / Linux)
curl -fsSL https://ollama.ai/install.sh | sh
# Windows: follow instructions at https://ollama.ai/docs


# Pull the mistral model (run after installing ollama)
ollama pull mistral
