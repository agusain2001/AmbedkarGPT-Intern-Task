# ✨ AmbedkarGPT – AI Intern Assignment Project

Welcome to **AmbedkarGPT**, a simple but powerful command-line Q&A system built as part of the **Kalpit Pvt Ltd AI Intern Hiring Task**. This project transforms a short speech by **Dr. B. R. Ambedkar** into an interactive question-answer tool using a compact RAG (Retrieval-Augmented Generation) pipeline.

You ask a question. The system retrieves the most relevant lines from the speech. A local AI model (Mistral 7B on Ollama) answers strictly from that text. No extra knowledge. No hallucination. Just clean retrieval.

---

## ⚙️ Tech Stack

* 🐍 **Python 3.8+**
* 🔗 **LangChain** for RAG orchestration
* 🤖 **Ollama** running **Mistral 7B** locally
* 🧠 **Sentence Transformers** (all-MiniLM-L6-v2) for embeddings
* 🗃️ **ChromaDB** for the vector store

---

## 🚀 Getting Started

Follow the steps below to set up and run the project on your machine.

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/agusain2001/AmbedkarGPT-Intern-Task.git
cd AmbedkarGPT-Intern-Task
```

### 2️⃣ Install Ollama + Pull the Mistral Model

Make sure Ollama is installed and running.

📥 **Install Ollama:**

```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

📦 **Download Mistral:**

```bash
ollama pull mistral
```

➡️ Keep the Ollama app running before starting the script.

---

### 3️⃣ Create and Activate a Virtual Environment

```bash
python3 -m venv venv

# macOS/Linux
source venv/bin/activate

# Windows
./venv/Scripts/activate
```

### 4️⃣ Install Project Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Program

Once everything is installed:

```bash
python main.py
```

You’ll see an interactive prompt:

```
Initializing Q&A system...
Initialization complete. Ask a question based on the text.
Type 'quit' or 'exit' to stop.
```

Now you can start asking questions based strictly on **speech.txt**.

### 💬 Example

```
Ask a question: What is the real remedy?

Answer:
The real remedy is to destroy the belief in the sanctity of the shastras.
```

```
Ask a question: What is the problem of caste?

Answer:
The problem of caste is not a matter of social reform. It is about overturning the authority of the shastras.
```

---

## 📂 Project Structure

```
.
├── .gitignore        # Ignores environment and cache files
├── main.py           # Main program (interactive Q&A)
├── README.md         # Documentation
├── requirements.txt  # Python dependencies
└── speech.txt        # Source text used in retrieval
```

---

## 🙌 Final Notes

This project is built to show your understanding of:

* How RAG works
* How embeddings and vector search improve retrieval
* How LLMs can be restricted to a specific knowledge base
* How local models (Ollama) can power lightweight AI apps

If you'd like enhancements like:
✨ better CLI design
✨ colorized terminal output
✨ improved chunking & retrieval logic
✨ LangChain Expression Language (LCEL
