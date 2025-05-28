"""
HybridRetrieverAgent: Responsible for retrieving relevant policy documents
using TRUE hybrid retrieval combining dense (embeddings) and sparse (BM25) methods.
"""
import os
import logging
from typing import List, Dict, Any, Optional
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS as LangchainFAISS
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

from ..config import POLICY_DOCS_DIR, VECTOR_STORE_DIR, GEMINI_API_KEY, TOP_K_RETRIEVAL, SIMILARITY_THRESHOLD
from ..utils.fallback_retrieval import FallbackRetriever

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridRetrieverAgent:
    """Agent to retrieve relevant policy documents using TRUE hybrid retrieval (Dense + Sparse BM25)."""
    
    def __init__(self, rebuild_index: bool = False):
        """
        Initialize the HybridRetrieverAgent with both dense and sparse capabilities.
        
        Args:
            rebuild_index: Whether to force rebuilding the vector index
        """
        # Dense retrieval setup (Vector embeddings)
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            google_api_key=GEMINI_API_KEY,
            model="models/embedding-001"
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Dense retrieval components
        self.vector_store = None
        self.index_path = os.path.join(VECTOR_STORE_DIR, "faiss_index")
        
        # Sparse retrieval setup (BM25)
        self.bm25_index = None
        self.tokenized_docs = []
        self.doc_metadata = []
        self.policy_chunks = []
        
        # Fallback retriever
        self.fallback_retriever = FallbackRetriever(POLICY_DOCS_DIR)
        
        # Load or build the indices
        if os.path.exists(self.index_path) and not rebuild_index:
            self._load_index()
        else:
            self._build_index()
    
    def _tokenize_document(self, text: str) -> List[str]:
        """
        Tokenize document for BM25 indexing.
        
        Args:
            text: Document text to tokenize
            
        Returns:
            List of cleaned tokens
        """
        # Convert to lowercase and split by spaces
        tokens = text.lower().split()
        
        # Clean tokens: remove punctuation and filter short tokens
        cleaned_tokens = []
        for token in tokens:
            # Remove punctuation and keep only alphanumeric characters
            cleaned_token = ''.join(char for char in token if char.isalnum())
            # Keep tokens with at least 2 characters
            if len(cleaned_token) >= 2:
                cleaned_tokens.append(cleaned_token)
        
        return cleaned_tokens
    
    def _build_bm25_index(self, policy_docs: List[str], metadata_list: List[Dict[str, Any]]):
        """
        Build BM25 sparse index from policy documents.
        
        Args:
            policy_docs: List of policy document chunks
            metadata_list: Corresponding metadata for each chunk
        """
        try:
            logger.info("Building BM25 sparse index...")
            
            # Clear existing sparse index data
            self.tokenized_docs = []
            self.doc_metadata = []
            self.policy_chunks = []
            
            # Tokenize each document chunk for BM25
            for i, chunk in enumerate(policy_docs):
                tokens = self._tokenize_document(chunk)
                
                if tokens:  # Only add non-empty tokenized docs
                    self.tokenized_docs.append(tokens)
                    self.policy_chunks.append(chunk)
                    self.doc_metadata.append(metadata_list[i])
            
            if self.tokenized_docs:
                # Create BM25 index
                self.bm25_index = BM25Okapi(self.tokenized_docs)
                logger.info(f"BM25 sparse index built successfully with {len(self.tokenized_docs)} documents")
            else:
                logger.warning("No documents available for BM25 indexing")
                
        except Exception as e:
            logger.error(f"Error building BM25 index: {e}")
            self.bm25_index = None
    
    def _load_index(self):
        """Load the FAISS index from disk and rebuild BM25 index."""
        try:
            logger.info(f"Loading existing FAISS index from {self.index_path}")
            self.vector_store = LangchainFAISS.load_local(
                self.index_path, 
                self.embedding_model,
                allow_dangerous_deserialization=True
            )
            logger.info("Dense index loaded successfully")
            
            # Rebuild BM25 index since it's not persisted
            self._rebuild_bm25_from_existing_data()
            
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            logger.info("Rebuilding both indices...")
            self._build_index()
    
    def _rebuild_bm25_from_existing_data(self):
        """Rebuild BM25 index from existing policy documents."""
        try:
            logger.info("Rebuilding BM25 index from policy documents...")
            
            # Read policy documents again for BM25
            policy_docs = []
            metadata_list = []
            
            for filename in os.listdir(POLICY_DOCS_DIR):
                if filename.endswith('.txt'):
                    file_path = os.path.join(POLICY_DOCS_DIR, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Split document into chunks (same as vector store)
                        chunks = self.text_splitter.split_text(content)
                        
                        for chunk in chunks:
                            policy_docs.append(chunk)
                            metadata_list.append({
                                'source': filename,
                                'chunk': chunk
                            })
                            
                    except Exception as e:
                        logger.error(f"Error reading file {filename} for BM25: {e}")
            
            # Build BM25 index
            self._build_bm25_index(policy_docs, metadata_list)
            
        except Exception as e:
            logger.error(f"Error rebuilding BM25 index: {e}")
    
    def _build_index(self):
        """Build both dense (FAISS) and sparse (BM25) indices from policy documents."""
        try:
            logger.info("Building hybrid indices (Dense FAISS + Sparse BM25) from policy documents")
            
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
            
            # Build dense vector store (FAISS)
            logger.info("Building dense vector store with FAISS...")
            self.vector_store = LangchainFAISS.from_texts(
                texts=policy_docs,
                embedding=self.embedding_model,
                metadatas=metadata_list
            )
            
            # Save the dense index
            os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
            self.vector_store.save_local(self.index_path)
            logger.info(f"Dense index built and saved with {len(policy_docs)} chunks")
            
            # Build sparse BM25 index
            self._build_bm25_index(policy_docs, metadata_list)
            
            logger.info("Hybrid indices (Dense + Sparse) built successfully!")
            
        except Exception as e:
            logger.error(f"Error building hybrid indices: {e}")
            raise
    
    def _dense_retrieval(self, query: str, k: int = TOP_K_RETRIEVAL) -> List[Dict[str, Any]]:
        """
        Perform dense retrieval using vector embeddings.
        
        Args:
            query: Search query
            k: Number of results to retrieve
            
        Returns:
            List of retrieved documents with scores
        """
        try:
            if not self.vector_store:
                logger.warning("Vector store not available for dense retrieval")
                return []
            
            docs_with_scores = self.vector_store.similarity_search_with_score(query, k=k)
            
            results = []
            for doc, score in docs_with_scores:
                # Convert FAISS distance to similarity score (0-1)
                similarity = float(max(0, 1.0 - min(1.0, score / 2.0)))
                
                results.append({
                    "content": doc.page_content,
                    "source": doc.metadata["source"],
                    "similarity": similarity,
                    "retrieval_method": "dense"
                })
            
            logger.info(f"Dense retrieval found {len(results)} documents")
            return results
            
        except Exception as e:
            logger.error(f"Error in dense retrieval: {e}")
            return []
    
    def _sparse_retrieval(self, query: str, k: int = TOP_K_RETRIEVAL) -> List[Dict[str, Any]]:
        """
        Perform sparse retrieval using BM25.
        
        Args:
            query: Search query
            k: Number of results to retrieve
            
        Returns:
            List of retrieved documents with BM25 scores
        """
        try:
            if not self.bm25_index or not self.tokenized_docs:
                logger.warning("BM25 index not available for sparse retrieval")
                return []
            
            # Tokenize query
            query_tokens = self._tokenize_document(query)
            
            if not query_tokens:
                logger.warning("Query tokenization resulted in empty tokens")
                return []
            
            # Get BM25 scores for all documents
            bm25_scores = self.bm25_index.get_scores(query_tokens)
            
            # Get top-k documents based on BM25 scores
            if len(bm25_scores) == 0:
                return []
            
            # Sort by score and get top k
            scored_docs = [(i, score) for i, score in enumerate(bm25_scores) if score > 0]
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            top_docs = scored_docs[:k]
            
            results = []
            max_score = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
            
            for doc_idx, score in top_docs:
                if doc_idx < len(self.doc_metadata):
                    # Normalize BM25 score to 0-1 range
                    normalized_score = min(1.0, score / max_score)
                    
                    results.append({
                        "content": self.doc_metadata[doc_idx]["chunk"],
                        "source": self.doc_metadata[doc_idx]["source"],
                        "similarity": float(normalized_score),
                        "retrieval_method": "sparse_bm25"
                    })
            
            logger.info(f"BM25 sparse retrieval found {len(results)} documents")
            return results
            
        except Exception as e:
            logger.error(f"Error in BM25 sparse retrieval: {e}")
            return []
    
    def _combine_results(self, dense_results: List[Dict[str, Any]], 
                        sparse_results: List[Dict[str, Any]], 
                        alpha: float = 0.7) -> List[Dict[str, Any]]:
        """
        Combine dense and sparse retrieval results using weighted score fusion.
        
        Args:
            dense_results: Results from dense retrieval
            sparse_results: Results from sparse (BM25) retrieval
            alpha: Weight for dense scores (1-alpha for sparse scores)
            
        Returns:
            Combined and ranked results
        """
        # Create a dictionary to combine scores for documents
        combined_scores = {}
        all_docs = {}
        
        # Process dense results
        for result in dense_results:
            # Create unique key for document (source + content hash)
            doc_key = f"{result['source']}_{hash(result['content'][:100])}"
            combined_scores[doc_key] = alpha * result['similarity']
            all_docs[doc_key] = result.copy()
            all_docs[doc_key]['combined_score'] = combined_scores[doc_key]
            all_docs[doc_key]['methods'] = ['dense']
        
        # Process sparse results
        for result in sparse_results:
            doc_key = f"{result['source']}_{hash(result['content'][:100])}"
            sparse_contribution = (1 - alpha) * result['similarity']
            
            if doc_key in combined_scores:
                # Document found in both - combine scores
                combined_scores[doc_key] += sparse_contribution
                all_docs[doc_key]['combined_score'] = combined_scores[doc_key]
                all_docs[doc_key]['methods'].append('sparse_bm25')
            else:
                # Document only in sparse results
                combined_scores[doc_key] = sparse_contribution
                all_docs[doc_key] = result.copy()
                all_docs[doc_key]['combined_score'] = combined_scores[doc_key]
                all_docs[doc_key]['methods'] = ['sparse_bm25']
        
        # Sort by combined score and return top results
        sorted_docs = sorted(all_docs.values(), key=lambda x: x['combined_score'], reverse=True)
        
        # Prepare final results
        final_results = []
        for doc in sorted_docs[:TOP_K_RETRIEVAL]:
            doc['similarity'] = doc['combined_score']
            doc['retrieval_method'] = '+'.join(doc['methods'])
            # Remove temporary fields
            doc.pop('combined_score', None)
            doc.pop('methods', None)
            final_results.append(doc)
        
        return final_results
    
    def retrieve(self, query: str, classification: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant policy documents using TRUE Hybrid RAG (Dense + Sparse BM25).
        
        Args:
            query: The user's input text
            classification: The classification from the detection agent
            
        Returns:
            List of dictionaries with relevant policy documents
        """
        try:
            logger.info(f"Starting TRUE Hybrid RAG retrieval (Dense + BM25) for: {classification}")
            
            # Enhance query with classification context
            enhanced_query = f"{classification} speech regarding: {query}"
            
            # Perform BOTH dense and sparse retrieval in parallel
            dense_results = self._dense_retrieval(enhanced_query, k=TOP_K_RETRIEVAL)
            sparse_results = self._sparse_retrieval(enhanced_query, k=TOP_K_RETRIEVAL)
            
            logger.info(f"Dense retrieval: {len(dense_results)} results, Sparse BM25: {len(sparse_results)} results")
            
            # Combine results using score fusion
            if dense_results or sparse_results:
                combined_results = self._combine_results(dense_results, sparse_results)
                
                # Filter by similarity threshold
                filtered_results = [
                    result for result in combined_results 
                    if result['similarity'] >= SIMILARITY_THRESHOLD
                ]
                
                if filtered_results:
                    logger.info(f"Hybrid RAG successful: {len(filtered_results)} final documents")
                    return filtered_results
            
            # Fallback: Try with original query (without classification prefix)
            logger.info("Trying hybrid retrieval with original query...")
            dense_fallback = self._dense_retrieval(query, k=TOP_K_RETRIEVAL)
            sparse_fallback = self._sparse_retrieval(query, k=TOP_K_RETRIEVAL)
            
            if dense_fallback or sparse_fallback:
                fallback_combined = self._combine_results(dense_fallback, sparse_fallback)
                
                # Use lower threshold for fallback
                lower_threshold = SIMILARITY_THRESHOLD * 0.8
                fallback_filtered = [
                    result for result in fallback_combined 
                    if result['similarity'] >= lower_threshold
                ]
                
                if fallback_filtered:
                    logger.info(f"Fallback hybrid retrieval successful: {len(fallback_filtered)} documents")
                    return fallback_filtered
            
            # Final fallback to keyword-based retrieval
            logger.info("Using keyword-based fallback retrieval")
            return self.fallback_retriever.retrieve(query, classification)
            
        except Exception as e:
            logger.error(f"Error in hybrid RAG retrieval: {e}")
            # Use fallback retriever as last resort
            return self.fallback_retriever.retrieve(query, classification)