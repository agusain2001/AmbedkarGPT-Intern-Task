"""
AmbedkarGPT - Command-line Q&A System
======================================
A RAG (Retrieval-Augmented Generation) pipeline that answers questions
based on Dr. B.R. Ambedkar's speech excerpt from "Annihilation of Caste".

Author: AI Intern Candidate
Assignment: Kalpit Pvt Ltd - Phase 1 Core Skills Evaluation
"""

import sys
import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA


def check_prerequisites():
    """
    Verify that all prerequisites are met before initialization.
    Returns: tuple (bool, str) - (success status, error message if any)
    """
    # Check if speech.txt exists
    if not os.path.exists('./speech.txt'):
        return False, "speech.txt not found in current directory"
    
    # Check if file is readable and not empty
    try:
        with open('./speech.txt', 'r', encoding='utf-8') as f:
            content = f.read()
            if not content.strip():
                return False, "speech.txt is empty"
    except Exception as e:
        return False, f"Cannot read speech.txt: {str(e)}"
    
    return True, ""


def create_rag_pipeline():
    """
    Initializes and returns the complete RAG pipeline.
    
    This function implements all 5 required steps:
    1. Load the text file (speech.txt)
    2. Split text into manageable chunks
    3. Create embeddings and store in ChromaDB vector store
    4. Set up retriever for relevant chunk retrieval
    5. Initialize Ollama LLM and create RetrievalQA chain
    
    Returns:
        RetrievalQA chain object or None if initialization fails
    """
    try:
        # Step 1: Load the provided text file
        print("📄 Loading speech.txt...")
        loader = TextLoader('./speech.txt', encoding='utf-8')
        documents = loader.load()
        print(f"   ✓ Loaded {len(documents)} document(s)")

        # Step 2: Split the text into manageable chunks
        print("\n✂️  Splitting text into chunks...")
        # Chunk size of 250 characters is optimal for this short speech
        # Overlap of 50 ensures context isn't lost between chunks
        text_splitter = CharacterTextSplitter(
            chunk_size=250,      # Small chunks for precise retrieval
            chunk_overlap=50,    # Overlap to maintain context continuity
            separator="\n"       # Split on newlines for natural boundaries
        )
        chunks = text_splitter.split_documents(documents)
        print(f"   ✓ Created {len(chunks)} text chunks")

        # Step 3: Create Embeddings and store in a local vector store
        print("\n🧠 Creating embeddings with HuggingFace model...")
        print("   (This may take a moment on first run as the model downloads)")
        # Using sentence-transformers/all-MiniLM-L6-v2 as specified
        # This model runs 100% locally with no API keys or accounts needed
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},  # Explicitly use CPU for compatibility
            encode_kwargs={'normalize_embeddings': True}  # Normalize for better similarity
        )
        print(f"   ✓ Loaded embedding model: {model_name}")
        
        # Store embeddings in ChromaDB (local, persistent-capable vector database)
        print("\n🗃️  Storing embeddings in ChromaDB vector store...")
        vectorstore = Chroma.from_documents(
            documents=chunks, 
            embedding=embeddings,
            collection_name="ambedkar_speech"  # Named collection for organization
        )
        print("   ✓ Vector store created successfully")

        # Step 4: Create retriever to fetch relevant chunks based on user queries
        print("\n🔍 Setting up retriever...")
        # Retriever uses similarity search to find most relevant chunks
        # k=3 means we retrieve top 3 most similar chunks for context
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}  # Retrieve top 3 chunks for balanced context
        )
        print("   ✓ Retriever configured (k=3 similarity search)")

        # Step 5: Initialize the LLM (Ollama with Mistral 7B model)
        print("\n🤖 Connecting to Ollama with Mistral 7B...")
        # Mistral 7B via Ollama - 100% free, local, no API keys needed
        # Requires: ollama pull mistral (must be run beforehand)
        llm = Ollama(
            model="mistral",
            temperature=0.1  # Low temperature (0.0-1.0) for factual, consistent responses
        )
        print("   ✓ Connected to Mistral model")

        # Create the RetrievalQA chain - the core of our RAG system
        print("\n⛓️  Building RetrievalQA chain...")
        # This chain orchestrates: Query → Retrieve Context → Generate Answer
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # "stuff" = pass all retrieved docs to LLM at once
            retriever=retriever,
            return_source_documents=False,  # Set True to see which chunks were used
            verbose=False  # Set True for debugging
        )
        print("   ✓ QA chain created successfully")
        
        return qa_chain

    except FileNotFoundError:
        print("\n❌ Error: speech.txt file not found!")
        print("   Please ensure speech.txt is in the same directory as main.py")
        return None
    
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("   Please install dependencies: pip install -r requirements.txt")
        return None
    
    except Exception as e:
        error_msg = str(e).lower()
        
        # Specific error handling for common issues
        if "ollama" in error_msg or "connection" in error_msg:
            print(f"\n❌ Error connecting to Ollama: {e}")
            print("\n🔧 Troubleshooting steps:")
            print("   1. Install Ollama: curl -fsSL https://ollama.ai/install.sh | sh")
            print("   2. Pull Mistral model: ollama pull mistral")
            print("   3. Ensure Ollama is running in the background")
            print("   4. Test with: ollama run mistral 'Hello'")
        else:
            print(f"\n❌ Unexpected error during initialization: {e}")
            print("\n🔧 Troubleshooting steps:")
            print("   1. Verify all dependencies: pip install -r requirements.txt")
            print("   2. Check Python version: python --version (needs 3.8+)")
            print("   3. Ensure speech.txt exists and is readable")
            print("   4. Restart Ollama service")
        
        return None


def main():
    """
    Main function to run the command-line Q&A system.
    
    Provides an interactive loop where users can:
    - Ask questions about Dr. Ambedkar's speech
    - Receive answers generated from retrieved context only
    - Exit gracefully with 'quit' or 'exit' commands
    """
    # Print welcome banner
    print("\n" + "=" * 70)
    print(" " * 15 + "🎓 AmbedkarGPT - Q&A System 🎓")
    print("=" * 70)
    print("\nA RAG-powered system to explore Dr. B.R. Ambedkar's speech on")
    print("the 'Annihilation of Caste' through interactive Q&A.\n")
    
    # Check prerequisites before initialization
    print("🔍 Checking prerequisites...")
    prereq_ok, error_msg = check_prerequisites()
    if not prereq_ok:
        print(f"❌ Prerequisite check failed: {error_msg}")
        print("   Please fix the issue and try again.")
        sys.exit(1)
    print("   ✓ All prerequisites met\n")
    
    # Initialize the RAG pipeline
    print("🚀 Initializing RAG pipeline...\n")
    qa_chain = create_rag_pipeline()

    if qa_chain is None:
        print("\n💥 Failed to initialize the system.")
        print("   Please check the errors above and try again.\n")
        sys.exit(1)

    # Success message
    print("\n" + "=" * 70)
    print(" " * 20 + "✅ SYSTEM READY!")
    print("=" * 70)
    print("\n📖 You can now ask questions based on the speech text.")
    print("💡 Examples:")
    print("   - What is the real remedy?")
    print("   - What is the problem with caste?")
    print("   - What does Ambedkar say about the shastras?")
    print("\n⌨️  Type 'quit', 'exit', or 'q' to stop.\n")

    # Main interaction loop
    question_count = 0
    while True:
        try:
            # Get user input
            query = input("❓ Ask a question: ").strip()

            # Check for exit commands
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n" + "=" * 70)
                print(f"   Thank you for using AmbedkarGPT!")
                print(f"   Questions answered: {question_count}")
                print("=" * 70)
                print("   Goodbye! 👋\n")
                break
            
            # Skip empty queries
            if not query:
                print("⚠️  Please enter a question.\n")
                continue

            # Process the query through the RAG pipeline
            print("\n⏳ Processing your question...", end="", flush=True)
            
            # The RAG pipeline workflow:
            # 1. Convert user query to embedding vector
            # 2. Search vector store for most similar document chunks
            # 3. Retrieve top k=3 relevant chunks
            # 4. Pass chunks + query to Mistral LLM
            # 5. LLM generates answer based ONLY on provided context
            response = qa_chain.invoke({"query": query})
            
            print("\r" + " " * 50 + "\r", end="")  # Clear processing message
            
            # Display the answer
            print("💬 Answer:")
            print("   " + response['result'].strip().replace('\n', '\n   '))
            print()
            
            question_count += 1

        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print("\n\n⚠️  Interrupted by user.")
            print(f"   Questions answered: {question_count}")
            print("   Exiting... 👋\n")
            break
            
        except Exception as e:
            print(f"\n\n❌ Error processing your question: {e}")
            print("   Please try again or type 'quit' to exit.\n")


if __name__ == "__main__":
    main()
