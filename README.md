# 🎓 AmbedkarGPT – AI Intern Assignment Project

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-Framework-green.svg)](https://langchain.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Welcome to **AmbedkarGPT**, a RAG (Retrieval-Augmented Generation) powered command-line Q&A system built as part of the **Kalpit Pvt Ltd AI Intern Hiring Assignment**. This project transforms a speech excerpt by **Dr. B. R. Ambedkar** from "Annihilation of Caste" into an interactive question-answering tool.

## 🎯 What Does It Do?

You ask a question → The system retrieves the most relevant text segments → A local AI model (Mistral 7B) generates an answer **strictly from the retrieved context** → No hallucinations, no external knowledge, just accurate retrieval-based responses.

---

## ⚙️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| 🐍 Language | **Python 3.8+** | Core programming language |
| 🔗 Framework | **LangChain** | RAG pipeline orchestration |
| 🤖 LLM | **Ollama (Mistral 7B)** | Local language model for answer generation |
| 🧠 Embeddings | **sentence-transformers/all-MiniLM-L6-v2** | Text-to-vector conversion |
| 🗃️ Vector DB | **ChromaDB** | Local vector storage and similarity search |

**Key Features:**
- ✅ 100% Local - No API keys, no cloud dependencies, no costs
- ✅ Privacy-First - All data stays on your machine
- ✅ Zero Hallucination - Answers only from provided text
- ✅ Fast Retrieval - Vector similarity search in milliseconds

---

## 📋 Prerequisites

Before you begin, ensure you have:

- **Python 3.8 or higher** installed
- **5 GB free disk space** (for Mistral model)
- **4 GB RAM minimum** (8 GB recommended)
- **Internet connection** (for initial setup only)

---

## 🚀 Installation & Setup

Follow these steps carefully to set up the project on your machine.

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/agusain2001/AmbedkarGPT-Intern-Task.git
cd AmbedkarGPT-Intern-Task
```

### 2️⃣ Install Ollama & Pull Mistral Model

**Ollama** is required to run the Mistral 7B model locally.

#### 📥 Install Ollama (Linux/macOS):
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

#### 📥 Install Ollama (Windows):
Download from [ollama.ai](https://ollama.ai) and run the installer.

#### 📦 Download the Mistral Model:
```bash
ollama pull mistral
```
*This downloads ~4 GB. First-time setup only.*

#### ✅ Verify Installation:
```bash
ollama run mistral "Hello"
```
You should see Mistral respond. Press `Ctrl+D` to exit.

**Important:** Keep Ollama running in the background before proceeding!

---

### 3️⃣ Create a Virtual Environment

Creating a virtual environment isolates project dependencies.

```bash
# Create virtual environment
python3 -m venv venv

# Activate it (macOS/Linux)
source venv/bin/activate

# Activate it (Windows)
venv\Scripts\activate
```

You should see `(venv)` in your terminal prompt.

---

### 4️⃣ Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**First-time note:** The sentence-transformers model (~80 MB) will download automatically on first run.

---

## ▶️ Running the Program

Once everything is installed and Ollama is running:

```bash
python main.py
```

### 🎬 Expected Output:

```
======================================================================
               🎓 AmbedkarGPT - Q&A System 🎓
======================================================================

A RAG-powered system to explore Dr. B.R. Ambedkar's speech on
the 'Annihilation of Caste' through interactive Q&A.

🔍 Checking prerequisites...
   ✓ All prerequisites met

🚀 Initializing RAG pipeline...

📄 Loading speech.txt...
   ✓ Loaded 1 document(s)

✂️  Splitting text into chunks...
   ✓ Created 4 text chunks

🧠 Creating embeddings with HuggingFace model...
   ✓ Loaded embedding model: sentence-transformers/all-MiniLM-L6-v2

🗃️  Storing embeddings in ChromaDB vector store...
   ✓ Vector store created successfully

🔍 Setting up retriever...
   ✓ Retriever configured (k=3 similarity search)

🤖 Connecting to Ollama with Mistral 7B...
   ✓ Connected to Mistral model

⛓️  Building RetrievalQA chain...
   ✓ QA chain created successfully

======================================================================
                    ✅ SYSTEM READY!
======================================================================

📖 You can now ask questions based on the speech text.
💡 Examples:
   - What is the real remedy?
   - What is the problem with caste?
   - What does Ambedkar say about the shastras?

⌨️  Type 'quit', 'exit', or 'q' to stop.

❓ Ask a question: 
```

---

## 💬 Example Usage

### Example 1: Understanding the Core Message
```
❓ Ask a question: What is the real remedy?

💬 Answer:
   The real remedy is to destroy the belief in the sanctity of the shastras.
```

### Example 2: Exploring the Problem
```
❓ Ask a question: What is the problem of caste according to Ambedkar?

💬 Answer:
   The problem of caste is not a problem of social reform. It is a problem of 
   overthrowing the authority of the shastras. Social reform alone cannot 
   eliminate caste as long as people believe in the sanctity of the scriptures.
```

### Example 3: Understanding the Analogy
```
❓ Ask a question: What analogy does he use for social reform?

💬 Answer:
   Dr. Ambedkar compares social reform to a gardener who constantly prunes 
   leaves and branches of a tree without ever attacking the roots. This 
   illustrates that superficial changes won't solve the core problem.
```

---

## 📂 Project Structure

```
AmbedkarGPT-Intern-Task/
│
├── main.py              # Core application - RAG pipeline implementation
├── speech.txt           # Source text - Dr. Ambedkar's speech excerpt
├── requirements.txt     # Python dependencies with version constraints
├── README.md            # This file - comprehensive documentation
├── .gitignore          # Git ignore rules (excludes cache, venv, etc.)
│
└── chroma/             # (Auto-generated) ChromaDB vector store data
```

---

## 🔧 Troubleshooting

### Issue: "Error connecting to Ollama"
**Solution:**
1. Check if Ollama is running: `ollama list`
2. Restart Ollama service
3. Verify Mistral is installed: `ollama pull mistral`

### Issue: "speech.txt not found"
**Solution:**
Ensure you're running `python main.py` from the project root directory where `speech.txt` is located.

### Issue: "ImportError" or missing modules
**Solution:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Issue: Slow first-time startup
**Expected behavior:** The first run downloads the sentence-transformers model (~80 MB). Subsequent runs are fast.

### Issue: Out of memory
**Solution:** 
- Close other applications
- Mistral 7B requires ~4 GB RAM
- Consider using a smaller model or upgrading RAM

---

## 🧪 How It Works (Technical Deep-Dive)

### The RAG Pipeline Architecture:

```
User Question
     ↓
[1] Text Embedding (Query → Vector)
     ↓
[2] Similarity Search (Find relevant chunks)
     ↓
[3] Context Retrieval (Top 3 chunks)
     ↓
[4] LLM Generation (Mistral processes context + query)
     ↓
Answer (Based only on retrieved context)
```

### Step-by-Step Breakdown:

1. **Document Loading**: `speech.txt` is loaded and preprocessed
2. **Text Chunking**: Split into 250-character chunks with 50-char overlap for context continuity
3. **Embedding Generation**: Each chunk is converted to a 384-dimensional vector using sentence-transformers
4. **Vector Storage**: Embeddings stored in ChromaDB for fast similarity search
5. **Query Processing**: User question → embedded → similarity search → top 3 chunks retrieved
6. **Answer Generation**: Mistral 7B receives chunks + question → generates contextual answer

### Why This Approach?

- **No Hallucination**: LLM only sees retrieved text, can't make things up
- **Explainable**: You can trace answers back to specific text segments
- **Efficient**: Only relevant context sent to LLM, reducing tokens and latency
- **Scalable**: Works with documents of any size (just add more chunks)

---

## 🎓 Learning Outcomes

This project demonstrates understanding of:

✅ **RAG Architecture** - Retrieval-Augmented Generation fundamentals  
✅ **Vector Embeddings** - Converting text to semantic vectors  
✅ **Similarity Search** - Finding relevant information via cosine similarity  
✅ **LLM Integration** - Combining retrieval with language models  
✅ **Local AI Stack** - Building AI apps without cloud dependencies  
✅ **Python Best Practices** - Clean code, error handling, documentation  

---

## 🚀 Possible Enhancements

Want to take this further? Here are some ideas:

- 🎨 **Better UI**: Add colorized terminal output with `rich` or `colorama`
- 📊 **Source Citations**: Show which text chunks were used for each answer
- 🔍 **Advanced Retrieval**: Implement hybrid search (keyword + semantic)
- 💾 **Persistent Storage**: Keep vector store across sessions
- 📈 **Performance Metrics**: Track retrieval accuracy and response time
- 🌐 **Web Interface**: Build a Streamlit or Flask frontend
- 📚 **Multi-Document**: Extend to handle multiple speeches/books
- 🤖 **Model Swapping**: Easy switching between different LLMs

---

## 📝 Assignment Compliance Checklist

This project meets all requirements:

- ✅ Python 3.8+ with clean, commented code
- ✅ LangChain framework for RAG orchestration
- ✅ ChromaDB as local vector store
- ✅ HuggingFace embeddings (sentence-transformers/all-MiniLM-L6-v2)
- ✅ Ollama with Mistral 7B (free, local, no API keys)
- ✅ All 5 pipeline steps implemented correctly
- ✅ Public GitHub repository with proper structure
- ✅ requirements.txt with all dependencies
- ✅ Comprehensive README.md documentation
- ✅ speech.txt included in repository

---

---

## 🙏 Acknowledgments

- **Dr. B. R. Ambedkar** - For his profound writings on social justice
- **LangChain Community** - For excellent RAG framework and documentation
- **Ollama Team** - For making local LLMs accessible
- **HuggingFace** - For open-source embedding models
- **Kalpit Pvt Ltd** - For this learning opportunity

---

## 📞 Support

If you encounter any issues:

1. Check the [Troubleshooting](#-troubleshooting) section above
2. Ensure all prerequisites are installed correctly
3. Verify Ollama is running: `ollama list`
4. Check Python version: `python --version` (needs 3.8+)


---

**Built with ❤️ using 100% local, open-source AI tools**
