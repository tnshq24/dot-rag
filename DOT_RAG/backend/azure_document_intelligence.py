import traceback
import os
from azure.core.credentials import AzureKeyCredential

from DOT_RAG.backend.azure_blob_storage import AzureBlobStorage

class AzureDocumentIntelligence(AzureBlobStorage):
    def __init__(self, logger):
        self.logger = logger
        super().__init__(logger=logger)
        self.__initialize_services()

    def __initialize_services(self):
        from azure.ai.documentintelligence import DocumentIntelligenceClient
        AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT = self._get_env_variables(
            "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"
        )
        AZURE_DOCUMENT_INTELLIGENCE_API_KEY = self._get_env_variables(
            "AZURE_DOCUMENT_INTELLIGENCE_API_KEY"
        )
        if not AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT:
            raise Exception("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT environment variable not set")
        if not AZURE_DOCUMENT_INTELLIGENCE_API_KEY:
            raise Exception("AZURE_DOCUMENT_INTELLIGENCE_API_KEY environment variable not set")

        try:
            self.AZURE_DOCUMENT_INTELLIGENCE_CLIENT = DocumentIntelligenceClient(
                endpoint=AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT,
                credential=AzureKeyCredential(AZURE_DOCUMENT_INTELLIGENCE_API_KEY),
            )
            self.logger.info("AZURE_DOCUMENT_INTELLIGENCE_CLIENT initalized successfully")
        except Exception as e:
            self.logger.error("AZURE_DOCUMENT_INTELLIGENCE_CLIENT initalization failed")
            self.logger.error(traceback.format_exc())
            raise e

    def _extract_using_document_intelligence(self, blob_name: str, return_raw: bool = False):
        """
        Extract text content from a PDF stored in Azure Blob Storage

        Args:
            blob_name: Name of the PDF blob in storage

        Returns:
            List of dictionaries containing page text and metadata
        """
        from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
        try:
            # Get the blob client for the PDF file
            blob_client = self.get_azure_blob_client(blob_name=blob_name)

            pdf_reader = self.AZURE_DOCUMENT_INTELLIGENCE_CLIENT.begin_analyze_document(
                "prebuilt-read", AnalyzeDocumentRequest(url_source=blob_client.url)
            )
            pdf_reader = pdf_reader.result()
            if return_raw:
                return pdf_reader
            
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
                        }
                    )
            dir_name = "/Users/sakhiagarwal/Downloads/dot_rag_pipeline 2/extracted_text-with-filter"
            filename = blob_name.split("/")[-1].replace(".pdf", "")
            file_path = os.path.join(dir_name, f"{filename}.txt")
            os.makedirs(dir_name, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                for page in pages_content:
                    f.write(f"Page {page['page_number']}:\n{page['content']}\n\n")
            print(f"Extracted text saved to {file_path}")
            return pages_content
        except Exception as e:
            self.logger.error(f"Error extracting text from PDF: {str(e)}")
            raise



