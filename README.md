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

## Setup Instructions
1. Create a conda environment

```bash
conda create -n ambedkar-rag python=3.10 -y
conda activate ambedkar-rag

---

2. Install Python dependencies
pip install -r requirements.txt

---

📂 Project Structure
.
├── main.py
├── speech.txt
├── requirements.txt
└── chroma_db/         (auto-created after first run)

---

## Running the App
1. Start the Ollama service
ollama serve

2. Run the RAG application
python main.py


Ask questions such as:

Question> What does the speech say about social reform?


Exit with:

exit

📘 How the Pipeline Works
1. Load the text

speech.txt is loaded using TextLoader.

2. Split into chunks

CharacterTextSplitter creates overlapping chunks for better retrieval.

3. Create embeddings

Using:

sentence-transformers/all-MiniLM-L6-v2

4. Store in ChromaDB

Vector store is saved locally in:

./chroma_db

5. Retrieve context

Retriever uses:

MMR retrieval

k = 4 relevant chunks

6. Generate the answer

Uses:

OllamaLLM(model="mistral")


The LangChain Expression Language (LCEL) connects:

retriever

prompt

LLM

output parser
