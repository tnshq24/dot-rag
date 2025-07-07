# Azure RAG Pipeline Implementation
# This pipeline demonstrates how to build a Retrieval-Augmented Generation system
# using Azure AI Search, OpenAI, and Azure Blob Storage

import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import asyncio
from dotenv import load_dotenv

# Azure SDK imports
from azure.storage.blob import BlobServiceClient, BlobClient
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import VectorizedQuery
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    VectorSearchProfile,
    VectorSearchAlgorithmConfiguration,
    HnswAlgorithmConfiguration,
    VectorSearchAlgorithmKind,
    SearchableField,
    SimpleField,
    ComplexField
)
from azure.core.credentials import AzureKeyCredential

# PDF processing imports
import PyPDF2
from io import BytesIO
import openai
from openai import OpenAI, AzureOpenAI

# Load environment variables from ENV.txt file
load_dotenv("ENV.txt")

# Configure logging to track pipeline operations
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AzureRAGPipeline:
    """
    Azure RAG Pipeline Class
    
    This class implements a complete RAG system using:
    - Azure AI Search for vector database functionality
    - OpenAI for LLM and embedding models
    - Azure Blob Storage for PDF document storage
    """
    
    def __init__(self):
        """
        Initialize the RAG pipeline with Azure services
        Load all configuration from environment variables
        """
        
        # Load Azure AI Search configuration
        # These credentials allow us to connect to Azure AI Search service
        self.search_service_name = os.getenv("AZURE_SEARCH_SERVICE_NAME")
        self.search_admin_key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
        self.search_endpoint = f"https://{self.search_service_name}.search.windows.net"
        
        # Load Azure Blob Storage configuration
        # These credentials allow us to store and retrieve PDF files
        self.blob_connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        self.blob_container_name = os.getenv("AZURE_BLOB_CONTAINER_NAME")
        
        # Load OpenAI configuration (supports both OpenAI and Azure OpenAI)
        # These credentials allow us to use OpenAI's models for embeddings and chat
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
        self.openai_chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-3.5-turbo")
        
        # Azure OpenAI specific configuration
        self.azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        self.azure_openai_embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        self.azure_openai_chat_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
        self.use_azure_openai = os.getenv("USE_AZURE_OPENAI", "false").lower() == "true"
        
        # Set the search index name where we'll store our document vectors
        # Use a different name to avoid conflicts with existing indexes
        self.index_name = os.getenv("AZURE_SEARCH_INDEX_NAME", "rag-documents-index")
        
        # Initialize Azure clients
        self._initialize_clients()
        
        # Validate that all required environment variables are set
        self._validate_configuration()
    
    def _initialize_clients(self):
        """
        Initialize all Azure service clients
        These clients provide the interface to interact with Azure services
        """
        
        try:
            # Validate that required credentials are strings and not None
            if not self.search_admin_key or not isinstance(self.search_admin_key, str):
                raise ValueError(f"AZURE_SEARCH_ADMIN_KEY must be a non-empty string. Got: {type(self.search_admin_key)}")
            
            if not self.search_service_name or not isinstance(self.search_service_name, str):
                raise ValueError(f"AZURE_SEARCH_SERVICE_NAME must be a non-empty string. Got: {type(self.search_service_name)}")
            
            if not self.blob_connection_string or not isinstance(self.blob_connection_string, str):
                raise ValueError(f"AZURE_STORAGE_CONNECTION_STRING must be a non-empty string. Got: {type(self.blob_connection_string)}")
            
            # Initialize Azure AI Search clients
            # SearchIndexClient manages the search index structure
            self.search_index_client = SearchIndexClient(
                endpoint=self.search_endpoint,
                credential=AzureKeyCredential(self.search_admin_key)
            )
            
            # SearchClient performs search operations on the index
            self.search_client = SearchClient(
                endpoint=self.search_endpoint,
                index_name=self.index_name,
                credential=AzureKeyCredential(self.search_admin_key)
            )
            
            # Initialize Azure Blob Storage client
            # This client manages file storage and retrieval
            self.blob_service_client = BlobServiceClient.from_connection_string(
                self.blob_connection_string
            )
            
            # Initialize OpenAI client (supports both OpenAI and Azure OpenAI)
            if self.use_azure_openai:
                # Validate Azure OpenAI credentials
                if not self.azure_openai_api_key or not isinstance(self.azure_openai_api_key, str):
                    raise ValueError(f"AZURE_OPENAI_API_KEY must be a non-empty string. Got: {type(self.azure_openai_api_key)}")
                
                if not self.azure_openai_endpoint or not isinstance(self.azure_openai_endpoint, str):
                    raise ValueError(f"AZURE_OPENAI_ENDPOINT must be a non-empty string. Got: {type(self.azure_openai_endpoint)}")
                
                # Initialize Azure OpenAI client using the correct Azure format
                self.openai_client = AzureOpenAI(
                    api_key=self.azure_openai_api_key,
                    azure_endpoint=self.azure_openai_endpoint,
                    api_version=self.azure_openai_api_version
                )
                # Use the same client for both embeddings and chat
                self.openai_chat_client = self.openai_client
            else:
                # Validate OpenAI credentials
                if not self.openai_api_key or not isinstance(self.openai_api_key, str):
                    raise ValueError(f"OPENAI_API_KEY must be a non-empty string. Got: {type(self.openai_api_key)}")
                
                # Initialize standard OpenAI client
                self.openai_client = OpenAI(api_key=self.openai_api_key)
                self.openai_chat_client = self.openai_client
                
        except Exception as e:
            logger.error(f"Error initializing clients: {str(e)}")
            raise
    
    def _validate_configuration(self):
        """
        Validate that all required environment variables are set
        This prevents runtime errors due to missing configuration
        """
        required_vars = [
            "AZURE_SEARCH_SERVICE_NAME",
            "AZURE_SEARCH_ADMIN_KEY", 
            "AZURE_STORAGE_CONNECTION_STRING",
            "AZURE_BLOB_CONTAINER_NAME"
        ]
        
        # Check OpenAI configuration based on which service is being used
        if self.use_azure_openai:
            required_vars.extend([
                "AZURE_OPENAI_ENDPOINT",
                "AZURE_OPENAI_API_KEY",
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
            ])
        else:
            required_vars.append("OPENAI_API_KEY")
        
        # Debug: Print all environment variables to see what's loaded
        logger.info("Environment variables loaded:")
        for var in required_vars:
            value = os.getenv(var)
            if value:
                # Mask sensitive values for security
                masked_value = value[:4] + "..." + value[-4:] if len(value) > 8 else "***"
                logger.info(f"  {var}: {masked_value}")
            else:
                logger.warning(f"  {var}: NOT SET")
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")
        
        logger.info("Configuration validation completed successfully")
    
    async def create_search_index(self):
        """
        Create the Azure AI Search index with vector search capabilities
        This index will store document chunks and their vector embeddings
        """
        
        # First, check if the index already exists and delete it to avoid conflicts
        try:
            existing_index = self.search_index_client.get_index(self.index_name)
            logger.info(f"Found existing index '{self.index_name}', deleting it...")
            self.search_index_client.delete_index(self.index_name)
            logger.info(f"Deleted existing index '{self.index_name}'")
        except Exception:
            # Index doesn't exist, which is fine
            logger.info(f"Index '{self.index_name}' doesn't exist, will create new one")
        
        # Define the search index fields
        # These fields define the structure of documents in our search index
        fields = [
            # Unique identifier for each document chunk
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            
            # Original filename of the PDF
            SearchableField(name="filename", type=SearchFieldDataType.String),
            
            # Text content of the document chunk
            SearchableField(name="content", type=SearchFieldDataType.String),
            
            # Page number where this chunk appears
            SimpleField(name="page_number", type=SearchFieldDataType.Int32),
            
            # Timestamp when the document was processed
            SimpleField(name="created_at", type=SearchFieldDataType.DateTimeOffset),
            
            # Vector embedding of the content (1536 dimensions for Ada model)
            SearchField(
                name="contentVector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=1536,  # Ada embedding model dimension
                vector_search_profile_name="my-vector-profile"
            )
        ]
        
        # Configure vector search settings
        # This enables similarity search using vector embeddings
        vector_search = VectorSearch(
            profiles=[
                VectorSearchProfile(
                    name="my-vector-profile",
                    algorithm_configuration_name="my-hnsw-config"
                )
            ],
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="my-hnsw-config",
                    kind=VectorSearchAlgorithmKind.HNSW,
                    parameters={
                        "m": 4,  # Number of bi-directional links for every new element
                        "efConstruction": 400,  # Size of the dynamic candidate list
                        "efSearch": 500,  # Size of the dynamic candidate list used during search
                        "metric": "cosine"  # Distance metric for similarity calculation
                    }
                )
            ]
        )
        
        # Create the search index with minimal configuration to avoid semantic search issues
        index = SearchIndex(
            name=self.index_name,
            fields=fields,
            vector_search=vector_search
        )
        
        try:
            # Create the index in Azure AI Search
            result = self.search_index_client.create_index(index)
            logger.info(f"Search index '{self.index_name}' created successfully")
            return result
        except Exception as e:
            logger.error(f"Error creating search index: {str(e)}")
            raise
    
    def upload_pdf_to_blob(self, file_path: str, blob_name: str) -> str:
        """
        Upload a PDF file to Azure Blob Storage
        
        Args:
            file_path: Local path to the PDF file
            blob_name: Name to give the blob in storage
            
        Returns:
            URL of the uploaded blob
        """
        try:
            # Get a reference to the blob client
            blob_client = self.blob_service_client.get_blob_client(
                container=self.blob_container_name,
                blob=blob_name
            )
            
            # Upload the file to blob storage
            with open(file_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
            
            logger.info(f"PDF uploaded to blob storage: {blob_name}")
            return blob_client.url
            
        except Exception as e:
            logger.error(f"Error uploading PDF to blob storage: {str(e)}")
            raise
    
    def extract_text_from_pdf_blob(self, blob_name: str) -> List[Dict[str, Any]]:
        """
        Extract text content from a PDF stored in Azure Blob Storage
        
        Args:
            blob_name: Name of the PDF blob in storage
            
        Returns:
            List of dictionaries containing page text and metadata
        """
        try:
            # Get the blob client for the PDF file
            blob_client = self.blob_service_client.get_blob_client(
                container=self.blob_container_name,
                blob=blob_name
            )
            
            # Download the PDF content from blob storage
            pdf_content = blob_client.download_blob().readall()
            
            # Create a BytesIO object to read the PDF content
            pdf_stream = BytesIO(pdf_content)
            
            # Extract text from each page using PyPDF2
            pdf_reader = PyPDF2.PdfReader(pdf_stream)
            pages_content = []
            
            for page_num, page in enumerate(pdf_reader.pages):
                # Extract text from the current page
                text = page.extract_text()
                
                # Only include pages with meaningful content
                if text.strip():
                    pages_content.append({
                        "page_number": page_num + 1,
                        "content": text.strip(),
                        "filename": blob_name
                    })
            
            logger.info(f"Extracted text from {len(pages_content)} pages")
            return pages_content
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split text into overlapping chunks for better retrieval
        
        Args:
            text: Input text to chunk
            chunk_size: Maximum size of each chunk
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        # Clean the text first
        text = text.strip()
        if not text:
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Calculate the end position for this chunk
            end = start + chunk_size
            
            # If this isn't the last chunk, try to break at a word boundary
            if end < len(text):
                # Find the last space before the end position
                while end > start and text[end] != ' ':
                    end -= 1
                
                # If no space found, use the original end position
                if end == start:
                    end = start + chunk_size
            
            # Extract the chunk
            chunk = text[start:end].strip()
            
            # Only add non-empty chunks with reasonable length
            if chunk and len(chunk) > 10:  # Skip very short chunks
                chunks.append(chunk)
            
            # Move start position for next chunk (with overlap)
            start = end - overlap
            
            # Ensure we don't go backwards and avoid infinite loops
            if start <= 0 or start >= len(text):
                break
        
        return chunks
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using OpenAI's Ada model
        Supports both standard OpenAI and Azure OpenAI
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            if self.use_azure_openai:
                # For Azure OpenAI, we need to use the deployment name as the model
                response = self.openai_client.embeddings.create(
                    model=self.azure_openai_embedding_deployment,
                    input=texts
                )
            else:
                # For standard OpenAI, use the model name
                response = self.openai_client.embeddings.create(
                    model=self.openai_embedding_model,
                    input=texts
                )
            
            # Extract embedding vectors from the response
            embeddings = [embedding.embedding for embedding in response.data]
            
            logger.info(f"Generated embeddings for {len(texts)} texts")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    async def index_document(self, blob_name: str):
        """
        Process a PDF document and index it in Azure AI Search
        
        Args:
            blob_name: Name of the PDF blob to process
        """
        try:
            # Step 1: Extract text from PDF
            logger.info(f"Processing document: {blob_name}")
            pages_content = self.extract_text_from_pdf_blob(blob_name)
            
            if not pages_content:
                logger.warning(f"No content extracted from {blob_name}")
                return
            
            # Step 2: Chunk the text content
            all_chunks = []
            for page_data in pages_content:
                # Skip pages with very little content
                if len(page_data["content"].strip()) < 50:
                    continue
                    
                # Split page content into smaller chunks
                chunks = self.chunk_text(page_data["content"])
                
                # Create chunk documents with metadata
                for i, chunk in enumerate(chunks):
                    # Clean and validate chunk content
                    chunk = chunk.strip()
                    if len(chunk) < 20:  # Skip very short chunks
                        continue
                    
                    # Create a safe, unique ID
                    safe_filename = blob_name.replace('.', '_').replace(' ', '_')
                    chunk_id = f"{safe_filename}_p{page_data['page_number']}_c{i}"
                    
                    chunk_doc = {
                        "content": chunk[:4000],  # Limit content length
                        "filename": blob_name[:100],  # Limit filename length
                        "page_number": page_data["page_number"],
                        "chunk_id": chunk_id[:100]  # Limit ID length
                    }
                    all_chunks.append(chunk_doc)
            
            if not all_chunks:
                logger.warning(f"No valid chunks created from {blob_name}")
                return
            
            logger.info(f"Created {len(all_chunks)} chunks from {blob_name}")
            
            # Step 3: Process chunks in smaller batches to avoid overwhelming the API
            batch_size = 10  # Process 10 chunks at a time
            total_indexed = 0
            
            for i in range(0, len(all_chunks), batch_size):
                batch_chunks = all_chunks[i:i + batch_size]
                
                # Step 4: Generate embeddings for this batch
                chunk_texts = [chunk["content"] for chunk in batch_chunks]
                embeddings = await self.generate_embeddings(chunk_texts)
                
                # Step 5: Prepare documents for indexing
                documents = []
                for chunk, embedding in zip(batch_chunks, embeddings):
                    doc = {
                        "id": chunk["chunk_id"],
                        "filename": chunk["filename"],
                        "content": chunk["content"],
                        "page_number": chunk["page_number"],
                        "created_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "contentVector": embedding
                    }
                    documents.append(doc)
                
                # Step 6: Upload this batch to Azure AI Search
                try:
                    result = self.search_client.upload_documents(documents)
                    successful_uploads = sum(1 for r in result if r.succeeded)
                    total_indexed += successful_uploads
                    
                    # Log any failures
                    for r in result:
                        if not r.succeeded:
                            logger.warning(f"Failed to index document {r.key}: {r.error_message}")
                    
                    logger.info(f"Batch {i//batch_size + 1}: Indexed {successful_uploads}/{len(documents)} chunks")
                    
                except Exception as e:
                    logger.error(f"Error indexing batch {i//batch_size + 1}: {str(e)}")
                    # Continue with next batch instead of failing completely
                    continue
            
            logger.info(f"Successfully indexed {total_indexed} chunks from {blob_name}")
            return {"indexed_chunks": total_indexed, "total_chunks": len(all_chunks)}
            
        except Exception as e:
            logger.error(f"Error indexing document: {str(e)}")
            raise
    
    async def search_similar_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for documents similar to the query using vector similarity
        
        Args:
            query: Search query text
            top_k: Number of similar documents to return
            
        Returns:
            List of similar documents with scores
        """
        try:
            # Step 1: Generate embedding for the search query
            query_embedding = await self.generate_embeddings([query])
            query_vector = query_embedding[0]
            
            # Step 2: Create a vectorized query for Azure AI Search
            vector_query = VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=top_k,
                fields="contentVector"
            )
            
            # Step 3: Execute the search
            search_results = self.search_client.search(
                search_text=None,  # We're doing pure vector search
                vector_queries=[vector_query],
                top=top_k
            )
            
            # Step 4: Process and return results
            results = []
            for result in search_results:
                doc = {
                    "content": result["content"],
                    "filepath": result.get("filepath", "N/A"),
                    "page_number": result.get("page_number", "N/A"),
                    "score": result["@search.score"]
                }
                results.append(doc)
            
            logger.info(f"Found {len(results)} similar documents")
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            raise
    
    async def generate_answer(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """
        Generate an answer using OpenAI's chat model and retrieved context
        Supports both standard OpenAI and Azure OpenAI
        
        Args:
            query: User's question
            context_docs: Retrieved documents for context
            
        Returns:
            Generated answer
        """
        try:
            # Step 1: Prepare the context from retrieved documents
            context = "\n\n".join([
                f"Document: {doc['filepath']} (Page {doc['page_number']})\n{doc['content']}"
                for doc in context_docs
            ])
            
            # Step 2: Create the prompt for the chat model
            system_prompt = """You are a helpful assistant that answers questions based on the provided context documents. 
            Use only the information from the context to answer questions. If the answer cannot be found in the context, 
            say so clearly. Always cite which document and page number you're referencing in your answer."""
            
            user_prompt = f"""Context Documents:
            {context}
            
            Question: {query}
            
            Please provide a detailed answer based on the context documents above."""
            
            # Step 3: Call OpenAI's chat completion API
            if self.use_azure_openai:
                # For Azure OpenAI, use the deployment name as the model
                response = self.openai_chat_client.chat.completions.create(
                    model=self.azure_openai_chat_deployment,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=1000,
                    temperature=0.1  # Low temperature for more focused answers
                )
            else:
                # For standard OpenAI, use the model name
                response = self.openai_chat_client.chat.completions.create(
                    model=self.openai_chat_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=1000,
                    temperature=0.1  # Low temperature for more focused answers
                )
            
            # Step 4: Extract and return the generated answer
            answer = response.choices[0].message.content
            
            logger.info("Generated answer using RAG pipeline")
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            raise
    
    async def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Complete RAG query: search for relevant documents and generate answer
        
        Args:
            question: User's question
            top_k: Number of documents to retrieve for context
            
        Returns:
            Dictionary containing the answer and source documents
        """
        try:
            # Step 1: Search for relevant documents
            logger.info(f"Processing query: {question}")
            relevant_docs = await self.search_similar_documents(question, top_k)
            
            # Step 2: Generate answer using retrieved documents
            answer = await self.generate_answer(question, relevant_docs)
            
            # Step 3: Return complete response
            response = {
                "question": question,
                "answer": answer,
                "source_documents": relevant_docs,
                "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
            }
            
            logger.info("RAG query completed successfully")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise

# Example usage and testing functions
async def main():
    """
    Example usage of the Azure RAG Pipeline
    This demonstrates how to use the pipeline for document indexing and querying
    """
    
    # Initialize the RAG pipeline
    rag_pipeline = AzureRAGPipeline()
    
    # Step 1: Create the search index (optional, skip if already created)
    # print("Creating search index...")
    # await rag_pipeline.create_search_index()
    
    # Step 2: Query the system
    test_question = "find me the village name of district bastar and sub district kondagaon"
    print(f"\n❓ Test Question: {test_question}")
    print("🔍 Querying the system...")
    response = await rag_pipeline.query(test_question, top_k=5)
    print(f"\n🤖 Answer:\n   {response['answer']}")

if __name__ == "__main__":
    # Run the example
    asyncio.run(main())