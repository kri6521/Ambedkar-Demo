# Ambedkar-Demo

A simple command-line Question-Answering (Q&A) system built using a Retrieval-Augmented Generation (RAG) pipeline.  
The system ingests a short excerpt from Dr. B.R. Ambedkar’s “Annihilation of Caste,” stores vector embeddings locally using ChromaDB, and answers user questions **based solely on that text** using **Ollama Mistral LLM**.

---

##  Features

- Loads and splits `speech.txt` into chunks  
- Creates embeddings using **HuggingFace all-MiniLM-L6-v2**  
- Stores vectors locally in **ChromaDB**  
- Retrieves relevant chunks using similarity search  
- Generates answers using **Ollama Mistral** via LangChain  
- Fully local — **no API keys or internet required**  
- Simple command-line interface

---

##  Prerequisites

Before running this project, ensure you have:

1. **Python 3.8+**
2. **Conda** (recommended)
3. **Ollama installed and running locally**
4. **Mistral model pulled for Ollama**

---

##  Install Ollama + Mistral

### Windows  
Follow instructions at:  
https://ollama.ai/docs

### macOS / Linux

```bash
curl -fsSL https://ollama.ai/install.sh | sh

---

##  Setup Instructions

```bash
conda create -n ambedkar-rag python=3.10 -y
conda activate ambedkar-rag
