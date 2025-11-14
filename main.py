import sys
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

def create_rag_pipeline():
    """
    Initializes and returns the complete RAG pipeline.
    """
    try:
        # Step 1: Load the provided text file [cite: 8]
        loader = TextLoader('./speech.txt')
        documents = loader.load()

        # Step 2: Split the text into manageable chunks [cite: 9]
        # Given the text is short, a smaller chunk size is effective.
        text_splitter = CharacterTextSplitter(chunk_size=250, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)

        # Step 3: Create Embeddings and store in a local vector store [cite: 10]
        
        # Use the specified HuggingFace model (runs locally) [cite: 17]
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        
        # Use ChromaDB as the local vector store [cite: 16]
        # This creates an in-memory vector store from the document chunks
        vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)

        # Step 4: Initialize the LLM (Ollama with Mistral) [cite: 18]
        # This assumes 'ollama pull mistral' has been run [cite: 58]
        llm = Ollama(model="mistral")

        # Step 5: Create the RetrievalQA chain [cite: 12, 15]
        # This chain combines the retriever (from the vector store) and the LLM
        retriever = vectorstore.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # "stuff" chain type feeds all retrieved context to the LLM
            retriever=retriever
        )
        
        return qa_chain

    except Exception as e:
        print(f"Error initializing RAG pipeline: {e}")
        print("\nPlease ensure Ollama is running and you have run 'ollama pull mistral'.")
        return None

def main():
    """
    Main function to run the command-line Q&A system.
    """
    print("Initializing Q&A system...")
    qa_chain = create_rag_pipeline()

    if qa_chain is None:
        sys.exit(1) # Exit if initialization failed

    print("Initialization complete. Ask a question based on the text.")
    print("Type 'quit' or 'exit' to stop.")

    while True:
        try:
            # Wait for user input
            query = input("\nAsk a question: ")

            if query.lower() in ['quit', 'exit']:
                print("Exiting...")
                break
            
            if not query:
                continue

            # Step 4 & 5 (in action): Retrieve context and generate answer [cite: 11, 12]
            response = qa_chain.invoke(query)
            
            print("\nAnswer:")
            print(response['result'].strip())

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()