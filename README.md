# AmbedkarGPT-Intern-Task

[cite_start]This project is a simple command-line Q&A system created for the Kalpit Pvt Ltd AI Intern Hiring Assignment[cite: 1, 4].

[cite_start]The system uses a Retrieval-Augmented Generation (RAG) pipeline orchestrated by the **LangChain** framework[cite: 15]. It ingests a short speech by Dr. B.R. [cite_start]Ambedkar [cite: 5][cite_start], stores it in a **ChromaDB** vector store [cite: 16][cite_start], and uses a locally-run **Ollama** model (Mistral 7B) to answer questions based *only* on the provided text[cite: 6, 18].

## 🛠️ Tech Stack

* [cite_start]**Language:** Python 3.8+ [cite: 14]
* [cite_start]**Core Framework:** LangChain [cite: 15]
* [cite_start]**LLM:** Ollama (with Mistral 7B) [cite: 18]
* [cite_start]**Embeddings:** HuggingFace `sentence-transformers/all-MiniLM-L6-v2` [cite: 17]
* [cite_start]**Vector Database:** ChromaDB [cite: 16]

## 🚀 Setup and Installation

Follow these steps to set up and run the project locally.

### 1. Clone the Repository

```bash
git clone [https://github.com/](https://github.com/)[YourUsername]/AmbedkarGPT-Intern-Task.git
cd AmbedkarGPT-Intern-Task
```

### 2. Install Ollama and Pull Mistral

[cite_start]You must install Ollama and download the Mistral 7B model before running the Python script[cite: 45].

**a. Install Ollama:**
(Run the command provided in the assignment brief) [cite_start][cite: 57]
```bash
curl -fsSL [https://ollama.ai/install.sh](https://ollama.ai/install.sh) | sh
```

**b. Pull the Mistral Model:**
(This will download the Mistral 7B model) [cite_start][cite: 58]
```bash
ollama pull mistral
```
**Important:** Ensure the Ollama application is running in the background before proceeding.

### 3. Create a Virtual Environment

[cite_start]It is highly recommended to use a virtual environment[cite: 44].

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate
```

### 4. Install Dependencies

[cite_start]Install all the required Python packages from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

## 🏃‍♀️ How to Run

Once you have completed the setup:

1.  Make sure your Ollama application is running.
2.  Run the `main.py` script from your terminal:

    ```bash
    python main.py
    ```

3.  The script will initialize the RAG pipeline. Once you see the `Ask a question:` prompt, you can ask questions based on the `speech.txt` content.

### Example Usage

```
$ python main.py
Initializing Q&A system...
Initialization complete. Ask a question based on the text.
Type 'quit' or 'exit' to stop.

Ask a question: What is the real remedy?

Answer:
The real remedy is to destroy the belief in the sanctity of the shastras.

Ask a question: What is the problem of caste?

Answer:
The problem of caste is not a problem of social reform. It is a problem of overthrowing the authority of the shastras.

Ask a question: quit
Exiting...
```

## 📂 File Structure

```
.
├── .gitignore         # Ignores Python cache and venv
[cite_start]├── main.py            # The main, executable Python script [cite: 20]
[cite_start]├── README.md          # This file 
[cite_start]├── requirements.txt   # Project dependencies 
[cite_start]└── speech.txt         # The source text file [cite: 23, 26]
```