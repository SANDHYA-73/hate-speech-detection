"""
Script to force rebuild the vector index with the correct embedding model.
"""
import os
import sys
import shutil
import time
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Get API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY is not set in environment")
    exit(1)

# Add app directory to path so we can import from it
sys.path.insert(0, os.path.abspath('.'))

def test_embedding_model(model_name):
    """Test if an embedding model works."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=GEMINI_API_KEY,
            model=model_name
        )
        
        # Try embedding a test string
        result = embeddings.embed_query("Test embedding")
        return True
    except Exception as e:
        print(f"Error with model {model_name}: {e}")
        return False

def find_working_embedding_model():
    """Find a working embedding model format."""
    # List of model formats to try
    model_formats = [
        "models/embedding-gecko-001",
        "embedding-gecko-001",
        "models/embedding-001",
        "embedding-001",
        "text-embedding-004"
    ]
    
    for model_format in model_formats:
        print(f"Testing model format: {model_format}")
        if test_embedding_model(model_format):
            print(f"✅ Found working model format: {model_format}")
            return model_format
    
    print("❌ Could not find a working embedding model format.")
    return None

def build_index_manually():
    """Build the FAISS index manually using policy documents."""
    try:
        # Get configurations
        from app.config import POLICY_DOCS_DIR, VECTOR_STORE_DIR
        
        print(f"Policy docs directory: {POLICY_DOCS_DIR}")
        print(f"Vector store directory: {VECTOR_STORE_DIR}")
        
        # Find a working embedding model
        model_format = find_working_embedding_model()
        if not model_format:
            print("Cannot proceed without a working embedding model.")
            return False
        
        # Create the embedding model
        embedding_model = GoogleGenerativeAIEmbeddings(
            google_api_key=GEMINI_API_KEY,
            model=model_format
        )
        
        # Create text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Get all policy documents
        policy_docs = []
        metadata_list = []
        
        # Read all policy files
        for filename in os.listdir(POLICY_DOCS_DIR):
            if filename.endswith('.txt'):
                file_path = os.path.join(POLICY_DOCS_DIR, filename)
                try:
                    print(f"Reading file: {filename}")
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Split document into chunks
                    chunks = text_splitter.split_text(content)
                    print(f"  - Created {len(chunks)} chunks")
                    
                    # Add chunks with metadata
                    for chunk in chunks:
                        policy_docs.append(chunk)
                        metadata_list.append({
                            'source': filename,
                            'chunk': chunk
                        })
                        
                except Exception as e:
                    print(f"Error reading file {filename}: {e}")
        
        if not policy_docs:
            print("No policy documents found to index")
            return False
            
        print(f"Creating vector store with {len(policy_docs)} chunks...")
        
        # Create the vector store
        vector_store = FAISS.from_texts(
            texts=policy_docs,
            embedding=embedding_model,
            metadatas=metadata_list
        )
        
        # Save the index
        os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
        vector_store.save_local(os.path.join(VECTOR_STORE_DIR, "faiss_index"))
        
        print(f"Index built successfully with {len(policy_docs)} chunks")
        
        # Test the retrieval
        print("\nTesting retrieval with the new index...")
        query = "hate speech against religious groups"
        results = vector_store.similarity_search(query, k=1)
        
        if results:
            print("Retrieval successful! Sample result:")
            print(f"Source: {results[0].metadata['source']}")
            print(f"Content: {results[0].page_content[:100]}...")
            return True
        else:
            print("Retrieval test failed. No results found.")
            return False
            
    except Exception as e:
        print(f"Error building index manually: {e}")
        return False

def force_rebuild_index():
    """Force rebuild the vector index with the correct embedding model."""
    try:
        # Get vector store directory from config
        from app.config import VECTOR_STORE_DIR
        
        print(f"Vector store directory: {VECTOR_STORE_DIR}")
        
        # Delete existing index if it exists
        if os.path.exists(VECTOR_STORE_DIR):
            print(f"Removing existing vector store from {VECTOR_STORE_DIR}...")
            try:
                # Try to delete just the faiss_index directory
                faiss_index_path = os.path.join(VECTOR_STORE_DIR, "faiss_index")
                if os.path.exists(faiss_index_path):
                    shutil.rmtree(faiss_index_path)
                else:
                    # If that doesn't exist, delete the entire vector store directory
                    shutil.rmtree(VECTOR_STORE_DIR)
                print("Successfully removed existing vector store.")
            except Exception as e:
                print(f"Error removing vector store: {e}")
                print("Will attempt to rebuild anyway.")
        
        # Create required directories
        os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
        
        # First try with the retriever agent
        print("First trying to rebuild with HybridRetrieverAgent...")
        success = False
        
        try:
            # Import the retriever agent
            from app.agents.retriever_agent import HybridRetrieverAgent
            
            # Force rebuild the index
            print("Creating retriever agent with force_rebuild=True...")
            retriever = HybridRetrieverAgent(rebuild_index=True)
            
            # Check if the index was created successfully
            if retriever.vector_store:
                print("Successfully rebuilt vector index with HybridRetrieverAgent!")
                success = True
        except Exception as e:
            print(f"Error rebuilding with HybridRetrieverAgent: {e}")
            print("Will try manual rebuild instead.")
        
        # If that failed, try building the index manually
        if not success:
            print("\nAttempting manual rebuild...")
            time.sleep(1)  # Pause briefly for readability
            
            if build_index_manually():
                print("\nSuccessfully rebuilt vector index manually!")
                
                # Update the model format in retriever_agent.py
                model_format = find_working_embedding_model()
                if model_format:
                    retriever_path = "app/agents/retriever_agent.py"
                    if os.path.exists(retriever_path):
                        with open(retriever_path, "r", encoding="utf-8") as file:
                            content = file.read()
                        
                        # Replace the model format
                        import re
                        updated_content = re.sub(
                            r'model="[^"]+"', 
                            f'model="{model_format}"', 
                            content
                        )
                        
                        with open(retriever_path, "w", encoding="utf-8") as file:
                            file.write(updated_content)
                        
                        print(f"Updated retriever_agent.py with working model format: {model_format}")
                
                return True
            else:
                print("\nFailed to rebuild vector index.")
                return False
        
        return success
        
    except Exception as e:
        print(f"Error rebuilding index: {e}")
        return False

if __name__ == "__main__":
    print("Forcing rebuild of vector index with the correct embedding model...")
    success = force_rebuild_index()
    
    if success:
        print("\nSuccessfully rebuilt the vector index!")
    else:
        print("\nFailed to rebuild the vector index.")
    
    print("\nDone! Please restart your application:")
    print("streamlit run app/main.py")