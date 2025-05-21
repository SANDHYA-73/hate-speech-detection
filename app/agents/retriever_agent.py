"""
HybridRetrieverAgent: Responsible for retrieving relevant policy documents
using embeddings and FAISS.
"""
import os
import logging
from typing import List, Dict, Any, Optional
import numpy as np
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS as LangchainFAISS
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

from ..config import POLICY_DOCS_DIR, VECTOR_STORE_DIR, GEMINI_API_KEY, TOP_K_RETRIEVAL, SIMILARITY_THRESHOLD

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridRetrieverAgent:
    """Agent to retrieve relevant policy documents using hybrid retrieval methods."""
    
    def __init__(self, rebuild_index: bool = False):
        """
        Initialize the HybridRetrieverAgent.
        
        Args:
            rebuild_index: Whether to force rebuilding the vector index
        """
        # Use the fully qualified model name with projects prefix - this is required format
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            google_api_key=GEMINI_API_KEY,
            model="models/embedding-001"  # Full model path required
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        self.vector_store = None
        self.index_path = os.path.join(VECTOR_STORE_DIR, "faiss_index")
        
        # Load or build the index
        if os.path.exists(self.index_path) and not rebuild_index:
            self._load_index()
        else:
            self._build_index()
    
    def _load_index(self):
        """Load the FAISS index from disk."""
        try:
            logger.info(f"Loading existing FAISS index from {self.index_path}")
            self.vector_store = LangchainFAISS.load_local(
                self.index_path, 
                self.embedding_model,
                allow_dangerous_deserialization=True
            )
            logger.info("Index loaded successfully")
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            logger.info("Rebuilding index...")
            self._build_index()
    
    def _build_index(self):
        """Build the FAISS index from policy documents."""
        try:
            logger.info("Building new FAISS index from policy documents")
            
            # Get all policy documents
            policy_docs = []
            metadata_list = []
            
            # Read all policy files
            for filename in os.listdir(POLICY_DOCS_DIR):
                if filename.endswith('.txt'):
                    file_path = os.path.join(POLICY_DOCS_DIR, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Split document into chunks
                        chunks = self.text_splitter.split_text(content)
                        
                        # Add chunks with metadata
                        for chunk in chunks:
                            policy_docs.append(chunk)
                            metadata_list.append({
                                'source': filename,
                                'chunk': chunk
                            })
                            
                    except Exception as e:
                        logger.error(f"Error reading file {filename}: {e}")
            
            if not policy_docs:
                raise ValueError("No policy documents found to index")
                
            # Create the vector store
            self.vector_store = LangchainFAISS.from_texts(
                texts=policy_docs,
                embedding=self.embedding_model,
                metadatas=metadata_list
            )
            
            # Save the index
            os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
            self.vector_store.save_local(self.index_path)
            
            logger.info(f"Index built successfully with {len(policy_docs)} chunks")
            
        except Exception as e:
            logger.error(f"Error building index: {e}")
            raise
    
    def retrieve(self, query: str, classification: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant policy documents based on query and classification.
        
        Args:
            query: The user's input text
            classification: The classification from the detection agent
            
        Returns:
            List of dictionaries with relevant policy documents
        """
        try:
            if not self.vector_store:
                raise ValueError("Vector store not initialized")
                
            # Combine the query with the classification for better retrieval
            augmented_query = f"{classification} speech regarding: {query}"
            
            # For ambiguous classifications, create multiple queries to improve retrieval
            if classification.lower() == "ambiguous":
                potential_queries = [
                    f"Hate speech regarding: {query}",
                    f"Offensive content like: {query}",
                    f"Toxic statements similar to: {query}",
                    f"Content policies about statements like: {query}"
                ]
                
                all_results = []
                
                # Try each query to maximize retrieval
                for potential_query in potential_queries:
                    try:
                        docs_with_scores = self.vector_store.similarity_search_with_score(
                            potential_query, 
                            k=1  # Get just one best match per query
                        )
                        
                        for doc, score in docs_with_scores:
                            # Convert the score to a similarity score (FAISS returns distance)
                            similarity = float(1.0 - min(1.0, score / 2.0))
                            
                            # Use a lower threshold for ambiguous content
                            if similarity >= (SIMILARITY_THRESHOLD * 0.8):  # 20% lower threshold
                                result = {
                                    "content": doc.page_content,
                                    "source": doc.metadata["source"],
                                    "similarity": similarity,
                                }
                                
                                # Only add if not already in results
                                if not any(r["source"] == result["source"] for r in all_results):
                                    all_results.append(result)
                    except Exception as e:
                        logger.warning(f"Error in one of the alternative queries: {e}")
                        continue
                
                # If we found results with the alternative queries, return them
                if all_results:
                    # Sort by similarity and take top results
                    all_results.sort(key=lambda x: x["similarity"], reverse=True)
                    top_results = all_results[:TOP_K_RETRIEVAL]
                    
                    logger.info(f"Retrieved {len(top_results)} relevant policy documents using alternative queries")
                    return top_results
            
            # Standard retrieval for non-ambiguous classifications or if alternative queries failed
            docs_with_scores = self.vector_store.similarity_search_with_score(
                augmented_query, 
                k=TOP_K_RETRIEVAL
            )
            
            # Filter by similarity threshold and format results
            results = []
            for doc, score in docs_with_scores:
                # Convert the score to a similarity score (FAISS returns distance)
                similarity = float(1.0 - min(1.0, score / 2.0))
                
                # Lower the threshold slightly for ambiguous content
                effective_threshold = SIMILARITY_THRESHOLD * 0.9 if classification.lower() == "ambiguous" else SIMILARITY_THRESHOLD
                
                if similarity >= effective_threshold:
                    results.append({
                        "content": doc.page_content,
                        "source": doc.metadata["source"],
                        "similarity": similarity,
                    })
            
            # If no results and the classification is potentially problematic, just return the top 2 policies
            if not results and classification.lower() in ["hate", "toxic", "offensive", "ambiguous"]:
                logger.info("No policies matched threshold, returning top policies as fallback")
                docs_with_scores = self.vector_store.similarity_search_with_score(
                    "hate speech policies", 
                    k=2
                )
                
                for doc, score in docs_with_scores:
                    results.append({
                        "content": doc.page_content,
                        "source": doc.metadata["source"],
                        "similarity": 0.65,  # Default similarity
                    })
            
            logger.info(f"Retrieved {len(results)} relevant policy documents")
            return results
            
        except Exception as e:
            logger.error(f"Error in document retrieval: {e}")
            return []