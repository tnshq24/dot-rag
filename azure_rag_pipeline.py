# Azure RAG Pipeline Implementation
# This pipeline demonstrates how to build a Retrieval-Augmented Generation system
# using Azure AI Search, OpenAI, and Azure Blob Storage

import warnings

warnings.filterwarnings(action="ignore", message=".*Failed to resolve*")
warnings.filterwarnings(action="ignore", message=".*Failed to establish*")

import os
import json
import logging
import re
from typing import List, Dict, Any
from datetime import datetime
import asyncio
import string
from dotenv import load_dotenv

# Azure SDK imports
from azure.storage.blob import BlobServiceClient
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import VectorizedQuery
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
    VectorSearchAlgorithmKind,
    SearchableField,
    SimpleField,
)

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest

from azure.core.credentials import AzureKeyCredential

# PDF processing imports
import PyPDF2
from io import BytesIO
from openai import OpenAI, AzureOpenAI
from azure_cosmos import AzureCosmos
import uuid
from prompts import _query_rephrase_prompt, is_language_query

# # Load environment variables from ENV.txt file
# load_dotenv("ENV.txt")

# Load environment variables from .env file
load_dotenv()

# Configure logging to track pipeline operations
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="logs.log",
    filemode="a",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
logger.info("Logger Initialised..")


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
        self.openai_embedding_model = os.getenv(
            "OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002"
        )
        self.openai_chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-3.5-turbo")

        # Azure OpenAI specific configuration
        self.azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")

        self.azure_document_intelligence_endpoint = os.getenv(
            "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"
        )
        self.azure_document_intelligence_api_key = os.getenv(
            "AZURE_DOCUMENT_INTELLIGENCE_API_KEY"
        )

        self.azure_openai_api_version = os.getenv(
            "AZURE_OPENAI_API_VERSION", "2024-02-15-preview"
        )
        self.azure_openai_embedding_deployment = os.getenv(
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
        )
        self.azure_openai_chat_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
        self.use_azure_openai = os.getenv("USE_AZURE_OPENAI", "false").lower() == "true"

        # Set the search index name where we'll store our document vectors
        self.index_name = os.getenv("AZURE_SEARCH_INDEX_NAME", "documents-index")

        # Initialize Azure clients
        self._initialize_clients()

        # Validate that all required environment variables are set
        self._validate_configuration()

    def _initialize_clients(self):
        """
        Initialize all Azure service clients
        These clients provide the interface to interact with Azure services
        """

        # Initialize Azure AI Search clients
        # SearchIndexClient manages the search index structure
        self.search_index_client = SearchIndexClient(
            endpoint=self.search_endpoint,
            credential=AzureKeyCredential(self.search_admin_key),
        )

        # # Initialize Azure AI Search clients
        # # SearchIndexerClient manages the search index structure
        # self.search_indexer_client = SearchIndexerClient(
        #     endpoint=self.search_endpoint,
        #     credential=AzureKeyCredential(self.search_admin_key)
        # )

        # SearchClient performs search operations on the index
        self.search_client = SearchClient(
            endpoint=self.search_endpoint,
            index_name=self.index_name,
            credential=AzureKeyCredential(self.search_admin_key),
        )

        # Initialize Document Intelligence client
        self.document_intelligence_client = DocumentIntelligenceClient(
            endpoint=self.azure_document_intelligence_endpoint,
            credential=AzureKeyCredential(self.azure_document_intelligence_api_key),
        )

        # Initialize Azure Blob Storage client
        # This client manages file storage and retrieval
        self.blob_service_client = BlobServiceClient.from_connection_string(
            self.blob_connection_string
        )

        # Initialize OpenAI client (supports both OpenAI and Azure OpenAI)
        if self.use_azure_openai:
            # Initialize Azure OpenAI client using the correct Azure format
            self.openai_client = AzureOpenAI(
                api_key=self.azure_openai_api_key,
                azure_endpoint=self.azure_openai_endpoint,
                api_version=self.azure_openai_api_version,
            )
            # Use the same client for both embeddings and chat
            self.openai_chat_client = self.openai_client
        else:
            # Initialize standard OpenAI client
            self.openai_client = OpenAI(api_key=self.openai_api_key)
            self.openai_chat_client = self.openai_client

        # Initialize Azure Cosmos DB client
        self.azure_cosmos = AzureCosmos()

    def _validate_configuration(self):
        """
        Validate that all required environment variables are set
        This prevents runtime errors due to missing configuration
        """
        required_vars = [
            "AZURE_SEARCH_SERVICE_NAME",
            "AZURE_SEARCH_ADMIN_KEY",
            "AZURE_STORAGE_CONNECTION_STRING",
            "AZURE_BLOB_CONTAINER_NAME",
        ]

        # Check OpenAI configuration based on which service is being used
        if self.use_azure_openai:
            required_vars.extend(
                [
                    "AZURE_OPENAI_ENDPOINT",
                    "AZURE_OPENAI_API_KEY",
                    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
                ]
            )
        else:
            required_vars.append("OPENAI_API_KEY")

        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")

        logger.info("Configuration validation completed successfully")

    async def create_search_index(self):
        """
        Create the Azure AI Search index with vector search capabilities
        This index will store document chunks and their vector embeddings
        """

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
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=1536,  # Ada embedding model dimension
                vector_search_profile_name="my-vector-profile",
            ),
        ]

        # Configure vector search settings
        # This enables similarity search using vector embeddings
        vector_search = VectorSearch(
            profiles=[
                VectorSearchProfile(
                    name="my-vector-profile",
                    algorithm_configuration_name="my-hnsw-config",
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
                        "metric": "cosine",  # Distance metric for similarity calculation
                    },
                )
            ],
        )

        # Create the search index
        index = SearchIndex(
            name=self.index_name, fields=fields, vector_search=vector_search
        )

        try:
            # Create or update the index in Azure AI Search
            result = self.search_index_client.create_or_update_index(index)
            logger.info(f"Search index '{self.index_name}' created successfully")
            return result
        except Exception as e:
            logger.error(f"Error creating search index: {str(e)}")
            raise

    def upload_pdf_to_blob_with_metadata(
        self, file_path: str, blob_name: str, metadata: dict
    ) -> str:
        """
        Upload a PDF file to Azure Blob Storage with metadata and custom path structure

        Args:
            file_path: Local path to the PDF file
            blob_name: Original filename
            metadata: Dictionary containing filename, project_code, and label_tag

        Returns:
            URL of the uploaded blob
        """
        try:
            from azure.storage.blob import BlobServiceClient
            from azure.identity import ClientSecretCredential

            # Extract metadata
            filename = metadata.get("filename", "")
            project_code = metadata.get("project_code", "")
            label_tag = metadata.get("label_tag", "")

            # safe_filename = os.path.basename(filename)
            # # print(f"safe_filename:{safe_filename}")
            # print(f"filename: {filename}")
            # print(f"blobname:{blob_name}")
            # Use original filename if metadata filename is empty
            if not filename:
                filename = blob_name

            # Create blob name with custom path structure
            blob_name_with_path = (
                f"Categories/{project_code}/{project_code}_{filename}.pdf"
            )

            # Authenticate using Service Principal
            credential = ClientSecretCredential(
                tenant_id=os.getenv("TENANT_ID"),
                client_id=os.getenv("CLIENT_ID"),
                client_secret=os.getenv("CLIENT_SECRET"),
            )

            account_url = "https://fabricbckp.blob.core.windows.net"
            blob_service = BlobServiceClient(
                account_url=account_url, credential=credential
            )

            # Get container client
            container = blob_service.get_container_client("dot-docs")

            # Read file data
            with open(file_path, "rb") as file_data:
                file_content = file_data.read()

            # Upload blob with metadata
            blob_metadata = {
                "filename": filename,
                "project_code": project_code,
                "label_tag": label_tag,
                "upload_timestamp": datetime.now().isoformat(),
            }

            # Upload the file to blob storage with metadata
            container.upload_blob(
                name=blob_name_with_path,
                data=file_content,
                overwrite=True,
                metadata=blob_metadata,
            )

            logger.info(
                f"PDF uploaded to blob storage with metadata: {blob_name_with_path}"
            )
            return f"{account_url}/dot-docs/{blob_name_with_path}"

        except Exception as e:
            logger.error(f"Error uploading PDF to blob storage with metadata: {str(e)}")
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
            # print(f"blob name in extraction:{blob_name}")

            # Get the blob client for the PDF file
            # print("CALLING Blob client")
            blob_client = self.blob_service_client.get_blob_client(
                container=self.blob_container_name, blob=blob_name
            )

            # Download the PDF content from blob storage
            # print("Downloading")
            pdf_content = blob_client.download_blob().readall()

            # Create a BytesIO object to read the PDF content
            pdf_stream = BytesIO(pdf_content)
            # print("Streaming")
            # Extract text from each page using PyPDF2
            pdf_reader = PyPDF2.PdfReader(pdf_stream)
            pages_content = []
            # print(pages_content)
            for page_num, page in enumerate(pdf_reader.pages):
                # Extract text from the current page
                text = page.extract_text()

                # Only include pages with meaningful content
                if text.strip():
                    pages_content.append(
                        {
                            "page_number": page_num + 1,
                            "content": text.strip(),
                            "filename": blob_name,
                        }
                    )

            if len(pages_content) == 0:
                pages_content = self.extract_text_from_pdf_blob_v2(blob_name=blob_name)
            # # print(f"pages content added filename :{pages_content["filename"]}")
            logger.info(f"Extracted text from {len(pages_content)} pages")
            return pages_content

        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise

    def extract_text_from_pdf_blob_v2(self, blob_name: str) -> List[Dict[str, Any]]:
        """
        Extract text content from a PDF stored in Azure Blob Storage

        Args:
            blob_name: Name of the PDF blob in storage

        Returns:
            List of dictionaries containing page text and metadata
        """
        try:
            # print(f"blob name in extraction:{blob_name}")
            # Get the blob client for the PDF file
            blob_client = self.blob_service_client.get_blob_client(
                container=self.blob_container_name, blob=blob_name
            )

            pdf_reader = self.document_intelligence_client.begin_analyze_document(
                "prebuilt-read", AnalyzeDocumentRequest(url_source=blob_client.url)
            )

            pdf_reader = pdf_reader.result()
            pages_content = []
            for page in pdf_reader.pages:
                # Combine all text from the page
                page_text = ""

                # Extract text from lines (preserves reading order)
                if hasattr(page, "lines") and page.lines:
                    for line in page.lines:
                        page_text += line.content + "\n"

                # If no lines, try paragraphs
                elif hasattr(pdf_reader, "paragraphs"):
                    page_paragraphs = [
                        p
                        for p in pdf_reader.paragraphs
                        if hasattr(p, "bounding_regions")
                        and any(
                            br.page_number == page.page_number
                            for br in p.bounding_regions
                        )
                    ]
                    for paragraph in page_paragraphs:
                        page_text += paragraph.content + "\n\n"

                # Only include pages with meaningful content
                if page_text.strip():
                    pages_content.append(
                        {
                            "page_number": page.page_number,
                            "content": page_text.strip(),
                            "filename": blob_name,
                            # "extraction_method": "Document Intelligence"
                        }
                    )
                    # # print(f"Pages content name {pages_content["filename"]}")
            return pages_content

        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise

    def chunk_text(
        self, text: str, chunk_size: int = 1000, overlap: int = 200
    ) -> List[str]:
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
                while end > start and text[end] != " ":
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
                    model=self.azure_openai_embedding_deployment, input=texts
                )
            else:
                # For standard OpenAI, use the model name
                response = self.openai_client.embeddings.create(
                    model=self.openai_embedding_model, input=texts
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
                    safe_filename = (
                        blob_name.replace(".", "_").replace(" ", "_").replace("/", "=")
                    )
                    chunk_id = f"{safe_filename}_p{page_data['page_number']}_c{i}"

                    chunk_doc = {
                        "content": chunk[:4000],  # Limit content length
                        "filename": blob_name[:100],  # Limit filename length
                        "page_number": page_data["page_number"],
                        "chunk_id": chunk_id[:100],  # Limit ID length
                    }
                    all_chunks.append(chunk_doc)

            if not all_chunks:
                logger.warning(f"No valid chunks created from {blob_name}")
                return

            logger.info(f"Created {len(all_chunks)} chunks from {blob_name}")
            for c in all_chunks[:3]:
                logger.info(f"Chunk preview: {c['content'][:200]}...")

            # Step 3: Process chunks in smaller batches to avoid overwhelming the API
            batch_size = 10  # Process 10 chunks at a time
            total_indexed = 0

            for i in range(0, len(all_chunks), batch_size):
                batch_chunks = all_chunks[i : i + batch_size]

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
                        "content_vector": embedding,
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
                            logger.warning(
                                f"Failed to index document {r.key}: {r.error_message}"
                            )

                    logger.info(
                        f"Batch {i//batch_size + 1}: Indexed {successful_uploads}/{len(documents)} chunks"
                    )

                except Exception as e:
                    logger.error(f"Error indexing batch {i//batch_size + 1}: {str(e)}")
                    # Continue with next batch instead of failing completely
                    continue

            logger.info(f"Successfully indexed {total_indexed} chunks from {blob_name}")
            return {"indexed_chunks": total_indexed, "total_chunks": len(all_chunks)}

        except Exception as e:
            logger.error(f"Error indexing document: {str(e)}")
            raise

    def get_blob_url(self, blob_name: str) -> str:
        """
        Get the URL for a blob in Azure Blob Storage

        Args:
            blob_name: Name of the blob file

        Returns:
            URL to access the blob
        """
        try:
            # Get the blob client for the PDF file
            blob_client = self.blob_service_client.get_blob_client(
                container=self.blob_container_name, blob=blob_name
            )
            return blob_client.url
        except Exception as e:
            logger.error(f"Error getting blob URL for {blob_name}: {str(e)}")
            return None

    def get_blob_view_url(self, blob_name: str) -> str:
        """
        Get a URL for viewing a blob in the browser using Flask route

        Args:
            blob_name: Name of the blob file

        Returns:
            URL to view the blob in browser
        """
        try:
            # Return the Flask route URL for viewing PDFs
            return f"/view_pdf/{blob_name}"
        except Exception as e:
            logger.error(f"Error getting blob view URL for {blob_name}: {str(e)}")
            return None

    async def search_similar_documents(
        self, query: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for documents similar to the query using vector similarity

        Args:
            query: Search query text
            top_k: Number of similar documents to return

        Returns:
            List of similar documents with scores and blob URLs
        """
        try:
            # Step 1: Generate embedding for the search query
            # print(f"query : {query}")
            query_embedding = await self.generate_embeddings([query])
            query_vector = query_embedding[0]

            # Step 2: Create a vectorized query for Azure AI Search
            vector_query = VectorizedQuery(
                vector=query_vector, k_nearest_neighbors=top_k, fields="content_vector"
            )
            # print(
                # f"Query vector: {query_embedding[0][:5]}... total dims: {len(query_embedding[0])}"
            # )

            # Step 3: Execute the search
            # translator = str.maketrans(
            #     string.punctuation, " " * len(string.punctuation)
            # )

            # keyword_text = query.translate(translator).strip()
            # # print(f"hybrid search: {keyword_text}")

            search_results = self.search_client.search(
                search_text=None,  # We're doing pure vector search
                vector_queries=[vector_query],
                top=top_k,
            )
            search_results = list(search_results)
            # print(f"Found {len(search_results)} search results")
            # for result in search_results:
                # print(
                #     f"{result['filename']} | Page {result['page_number']} | Score: {result['@search.score']}"
                # )

            # Step 4: Process and return results with blob URLs
            results = []
            for result in search_results:
                download_url = self.get_blob_url(result["filename"])
                view_url = self.get_blob_view_url(result["filename"])
                doc = {
                    "content": result["content"],
                    "filename": result["filename"],
                    "page_number": result["page_number"],
                    "score": result["@search.score"],
                    "download_url": download_url,
                    "view_url": view_url,
                }
                results.append(doc)

            logger.info(f"Found {len(results)} similar documents")
            return results

        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            raise

    # async def generate_answer(
    #     self,
    #     query: str,
    #     context_docs: List[Dict[str, Any]],
    # ) -> str:
    #     """
    #     Generate an answer using OpenAI's chat model and retrieved context
    #     Supports both standard OpenAI and Azure OpenAI

    #     Args:
    #         query: User's question
    #         context_docs: Retrieved documents for context

    #     Returns:
    #         Generated answer
    #     """
    #     try:
    #         # Step 1: Prepare the context from retrieved documents
    #         context = "\n\n".join(
    #             [
    #                 f"Document: {doc['filename']} (Page {doc['page_number']})\n{doc['content']}"
    #                 for doc in context_docs
    #             ]
    #         )

    #         # Step 2: Create the prompt for the chat model
    #         system_prompt = """You are a helpful assistant that answers questions based on the provided context documents.
    #         Use only the information from the context to answer questions. If the answer cannot be found in the context,
    #         say so clearly. Always cite which document and page number you're referencing in your answer."""

    #         user_prompt = f"""Context Documents:
    #         {context}

    #         Question: {query}

    #         Please provide a detailed answer based on the context documents above."""

    #         # Step 3: Call OpenAI's chat completion API
    #         if self.use_azure_openai:
    #             # For Azure OpenAI, use the deployment name as the model
    #             response = self.openai_chat_client.chat.completions.create(
    #                 model=self.azure_openai_chat_deployment,
    #                 messages=[
    #                     {"role": "system", "content": system_prompt},
    #                     {"role": "user", "content": user_prompt},
    #                 ],
    #                 max_tokens=1000,
    #                 temperature=0.1,  # Low temperature for more focused answers
    #             )
    #         else:
    #             # For standard OpenAI, use the model name
    #             response = self.openai_chat_client.chat.completions.create(
    #                 model=self.openai_chat_model,
    #                 messages=[
    #                     {"role": "system", "content": system_prompt},
    #                     {"role": "user", "content": user_prompt},
    #                 ],
    #                 max_tokens=1000,
    #                 temperature=0.1,  # Low temperature for more focused answers
    #             )

    #         # Step 4: Extract and return the generated answer
    #         answer = response.choices[0].message.content

    #         logger.info("Generated answer using RAG pipeline")
    #         return answer

    #     except Exception as e:
    #         logger.error(f"Error generating answer: {str(e)}")
    #         raise

    async def query(
        self,
        question: str,
        user_id: str,
        conversation_id: str,
        session_id: str,
        file_name=None,
        project_code=None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Complete RAG query: search for relevant documents and generate answer

        Args:
            question: User's question
            user_id: Unique identifier for the user
            conversation_id: Unique identifier for the conversation
            session_id: Unique identifier for the session
            file_name: Name of the file being queried. If None, queries all files.
            project_code: Code of the project being queried. If None, queries all projects.
            top_k: Number of documents to retrieve for context

        Returns:
            Dictionary containing the answer and source documents
        """
        try:

            cosmos_data = {
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "conversation_id": conversation_id,
                "session_id": session_id,
                "project_code": project_code,
                "file_name": file_name,
                "question": question,
                "rephrased_question": "",
            }
            logger.info(f"Processing query: {question}")
            question_column_name = "question"
            # Step 1: Fetch Past Questions from Cosmos DB
            if not file_name or not project_code:
                cosmos_query = f"""SELECT TOP 5 c.{question_column_name}, c.rephrased_question, c.answer FROM chat_sessions c WHERE c.user_id = '{user_id}' AND c.session_id = '{session_id}' ORDER BY c._ts DESC"""
            else:
                cosmos_query = f"""SELECT TOP 5 c.{question_column_name}, c.rephrased_question, c.answer FROM chat_sessions c WHERE c.user_id = '{user_id}' AND c.session_id = '{session_id}' AND c.project_code = '{project_code}' AND c.file_name = '{file_name}' ORDER BY c._ts DESC"""
            # # print(cosmos_query)

            fetched_df = self.azure_cosmos.read_table(
                query=cosmos_query,
                container_name="chat_sessions",
            )
            # # print(fetched_df)
            if fetched_df is None or fetched_df is False:
                previous_convo_string = None
                logger.info(
                    "No previous questions found in Cosmos DB or error occurred"
                )

                # setting rephrased query to original query
                cosmos_data["rephrased_question"] = question
                question_column_name = "rephrased_question"
            else:
                previous_convo_string = ""
                for idx, row in fetched_df.iloc[::-1].reset_index(drop=True).iterrows():
                    ques = row[question_column_name]
                    if row["rephrased_question"]:
                        ques = row["rephrased_question"]
                    previous_convo_string += (
                        f'Question {idx+1}: {ques}\nAnswer {idx+1}: {row["answer"]}\n\n'
                    )
                previous_convo_string = previous_convo_string.strip()
                # print("PREVIOUS CHAT:", previous_convo_string)
                rephrase_messages = _query_rephrase_prompt(
                    query=question, previous_conversation=previous_convo_string
                )
                # print(f"rephrase question:{rephrase_messages}")
                if self.use_azure_openai:
                    # For Azure OpenAI, use the deployment name as the model
                    response = self.openai_chat_client.chat.completions.create(
                        model=self.azure_openai_chat_deployment,
                        messages=rephrase_messages,
                        max_tokens=1000,
                        temperature=0.1,  # Low temperature for more focused answers
                        response_format={
                            "type": "json_object",
                        },
                    )
                else:
                    # For standard OpenAI, use the model name
                    response = self.openai_chat_client.chat.completions.create(
                        model=self.openai_chat_model,
                        messages=rephrase_messages,
                        max_tokens=1000,
                        temperature=0.1,  # Low temperature for more focused answers
                        response_format={
                            "type": "json_object",
                        },
                    )

                response = response.choices[0].message.content
                rephrased_query = json.loads(response)["rephrased_query"]
                if "not a follow-up question" not in rephrased_query.lower():
                    cosmos_data["rephrased_question"] = rephrased_query
                    question_column_name = "rephrased_question"
                else:
                    cosmos_data["rephrased_question"] = question
                    question_column_name = "rephrased_question"

                # del response
            is_language = await is_language_query(cosmos_data[question_column_name])
            # print(f"Is language query: {is_language}")
            if is_language:
                logger.info(
                    "Blocked a language/literature query from proceeding to retrieval."
                )
                return {
                    "question": question,
                    "rephrased_question": cosmos_data["rephrased_question"],
                    "answer": "Sorry, I cannot help with word meanings or literature-related questions. Please ask something related to the uploaded documents.",
                    "source_documents": [],
                    "is_relevant": False,
                    "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
            # Step 2: Search for relevant documents
            relevant_docs = await self.search_similar_documents(
                cosmos_data[question_column_name], top_k
            )
            # # print("*" * 50)
            # # print(len(relevant_docs))
            # # print(relevant_docs)
            # # print("*" * 50)

            # Step 3: Check if the query is relevant to the documents
            is_relevant = self._is_query_relevant(
                cosmos_data[question_column_name], relevant_docs
            )

            # Step 4: Generate answer using retrieved documents
            answer = await self.generate_answer(
                cosmos_data[question_column_name],
                relevant_docs,
                is_relevant,
                previous_convo_string,
            )

            # Step 5: Return complete response with relevance flag
            response = {
                "question": question,
                "rephrased_question": cosmos_data["rephrased_question"],
                "answer": answer,
                "source_documents": relevant_docs if is_relevant else [],
                "is_relevant": is_relevant,
                "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            for key, value in response.items():
                cosmos_data[key] = value
            # self.azure_cosmos.upload_to_cosmos(cosmos_data, "chat_history")
            logger.info("RAG query completed successfully")
            return response

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise

    def _is_query_relevant(
        self, question: str, relevant_docs: List[Dict[str, Any]]
    ) -> bool:
        """
        Determine if the query is relevant to the uploaded documents

        Args:
            question: User's question
            relevant_docs: Retrieved documents

        Returns:
            True if query is relevant, False otherwise
        """
        # Define irrelevant query patterns
        irrelevant_patterns = [
            r"\b(hi|hello|hey|good morning|good afternoon|good evening)\b",
            r"\b(how are you|how do you do)\b",
            r"\b(what is your name|who are you)\b",
            r"\b(thank you|thanks)\b",
            r"\b(bye|goodbye|see you)\b",
            r"\b(what time|what day|what date)\b",
            r"\b(weather|temperature)\b",
            r"\b(joke|funny|humor)\b",
            r"\b(help|support)\b",
            r"\b(menu|options)\b",
        ]

        # Check if query matches irrelevant patterns
        question_lower = question.lower().strip()
        for pattern in irrelevant_patterns:
            if re.search(pattern, question_lower):
                return False

        # Check if we have relevant documents with good scores
        if not relevant_docs:
            return False

        # Check if any document has a good relevance score
        max_score = max(doc.get("score", 0) for doc in relevant_docs)
        return max_score > 0.3  # Adjust threshold as needed

    async def generate_answer(
        self,
        query: str,
        context_docs: List[Dict[str, Any]],
        is_relevant: bool = True,
        previous_convo_string: str = "",
    ) -> str:
        """
        Generate an answer using OpenAI's chat model and retrieved context
        Supports both standard OpenAI and Azure OpenAI

        Args:
            query: User's question
            context_docs: Retrieved documents for context
            is_relevant: Whether the query is relevant to documents

        Returns:
            Generated answer
        """
        try:
            if not is_relevant:
                return """Thanks for your question! 😊 Unfortunately, I couldn’t find any relevant information in the uploaded documents to answer your question accurately. If you have any specific document that mentions this topic, feel free to upload it and I’ll gladly help!"""

            # Step 1: Prepare the context from retrieved documents
            context = "\n\n".join(
                [
                    f"Document: {os.path.basename(doc['filename'])} (Page {doc['page_number']})\n{doc['content']}"
                    for doc in context_docs
                ]
            )

            # Step 2: Create the prompt for the chat model
            system_prompt = """ You are an expert assistant with access to a set of uploaded legal Tender documents.\n Follow these rules strictly while responding:
            1. Context and Relevance: Answers must ONLY be based on the provided context (documents) and previous chat history. - Do NOT use outside knowledge or assumptions.

            2. Document References: Always cite which document and page number supports your statements. Clearly distinguish between documents that support **factual claims** vs **inferred interpretations**.

            3. Inference vs Factual Claims: If the answer is found **explicitly** in the documents, clearly present it as a fact.
            -  If the answer is **inferred** (not directly stated), clearly label it using phrases like:
            -  "Based on inference..."
            -  "While not explicitly stated, this can be interpreted as..."
            -  "It is reasonable to infer from..."
            ❗ NEVER present inferred information as if it is directly written in the document.
  
            4. Language Guidelines:
            - Use cautious language when inferring. Avoid overconfident terms like “will result in” unless the document clearly states it.
            - Prefer terms like “may lead to,” “could result in,” or “might imply” for inferred content.


            5. Multiple Document References: Provide document references only if the answer is explicitly found within the documents, include references at the end of the answer Also , below format given between triple backticks. If clarification is needed, do NOT mention document references prematurely. Ask for clarification clearly without citing any documents.
            
            ```
            References:
            - filename1, Pages: number of pages (like 1 and 5)
            - filename2, Pages: number of pages (like 6 and 35)

            ```
            4. Out-of-Scope Queries: If the user's query does not pertain to any content within the uploaded documents, explicitly state that the query is outside the scope of available documents. ***Do NOT mention any document references in such cases.*** Use this response template when information is unavailable:
            "Thank you for your question! However, after reviewing the provided documents, I couldn’t find relevant information to accurately answer your query. If this topic is covered in any other document, please upload it, and I'll be happy to assist further.

            Past Conversation (If any):
            {previous_conversation}"""

            user_prompt = f"""Context Documents:
            {context}
            
            Question: {query}
            
            Please provide a detailed answer based on the context documents above."""

            # Step 3: Call OpenAI's chat completion API
            if self.use_azure_openai:
                # For Azure OpenAI, use the deployment name as the model
                # print("\n===== SYSTEM PROMPT =====\n")
                # print(system_prompt.format(previous_conversation=previous_convo_string))
                # print("\n===== USER PROMPT =====\n")
                # print(user_prompt)
                response = self.openai_chat_client.chat.completions.create(
                    model=self.azure_openai_chat_deployment,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt.format(
                                previous_conversation=previous_convo_string
                            ),
                        },
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=1000,
                    temperature=0.1,  # Low temperature for more focused answers
                )
            else:
                # For standard OpenAI, use the model name
                response = self.openai_chat_client.chat.completions.create(
                    model=self.openai_chat_model,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt.format(
                                previous_conversation=previous_convo_string
                            ),
                        },
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=1000,
                    temperature=0.1,  # Low temperature for more focused answers
                )

            # Step 4: Extract and return the generated answer
            answer = response.choices[0].message.content
            answer = answer.replace("```", "")

            logger.info("Generated answer using RAG pipeline")
            return answer

        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            raise


# Example usage and testing functions
async def main():
    """
    Example usage of the Azure RAG Pipeline
    This demonstrates how to use the pipeline for document indexing and querying
    """

    # Initialize the RAG pipeline
    rag_pipeline = AzureRAGPipeline()

    # Step 1: Create the search index
    # # print("Creating search index...")
    # await rag_pipeline.create_search_index()

    # pdf_path = "502-UCV-Amendment_in_agreement_Bihar2.pdf"
    # blob_name = pdf_path
    # rag_pipeline.upload_pdf_to_blob(pdf_path, blob_name)
    # await rag_pipeline.index_document(blob_name)

    # Step 2: Upload a PDF to blob storage (replace with your PDF path)
    for folder_name in os.listdir("dot-docs"):
        if folder_name[0] != ".":
            for file_name in os.listdir(f"dot-docs/{folder_name}"):
                # print(folder_name, file_name)
                pdf_path = f"dot-docs/{folder_name}/{file_name}"
                blob_name = f"Categories/{folder_name}/{file_name}"
                rag_pipeline.upload_pdf_to_blob(pdf_path, blob_name)

                # Step 3: Index the document
                # print("Indexing document...")
                await rag_pipeline.index_document(blob_name)

    # # Step 4: Query the system
    # # print("Querying the system...")
    # # response = await rag_pipeline.query("What is the main topic of the document?")
    # response = await rag_pipeline.query("Summarize 502-UCV-Amendment_in_agreement_Bihar2 ")
    # # print(f"Answer: {response['answer']}")
    # # print(f"Sources: {len(response['source_documents'])} documents")


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())
