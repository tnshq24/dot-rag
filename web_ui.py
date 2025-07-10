from flask import Flask, render_template, request, jsonify, session
import asyncio
import re
from azure_rag_pipeline import AzureRAGPipeline
import os
import tempfile
import traceback

app = Flask(__name__)

# Initialize the RAG pipeline
rag_pipeline = None

def initialize_rag_pipeline():
    """Initialize the RAG pipeline"""
    global rag_pipeline
    try:
        rag_pipeline = AzureRAGPipeline()
        return True
    except Exception as e:
        print(f"Error initializing RAG pipeline: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure ENV.txt file exists in the current directory")
        print("2. Check that all required environment variables are set:")
        print("   - AZURE_SEARCH_SERVICE_NAME")
        print("   - AZURE_SEARCH_ADMIN_KEY")
        print("   - AZURE_STORAGE_CONNECTION_STRING")
        print("   - AZURE_BLOB_CONTAINER_NAME")
        print("   - AZURE_OPENAI_ENDPOINT")
        print("   - AZURE_OPENAI_API_KEY")
        print("   - AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        print("   - AZURE_OPENAI_CHAT_DEPLOYMENT")
        print("   - USE_AZURE_OPENAI")
        return False

def clean_response(response_text):
    """
    Clean and format the response text by:
    1. Removing markdown bold formatting (**text** -> text)
    2. Removing extra whitespace and newlines
    3. Making the response more readable
    """
    if not response_text:
        return ""
    
    # Remove markdown bold formatting (**text** -> text)
    cleaned = re.sub(r'\*\*(.*?)\*\*', r'\1', response_text)
    
    # Remove any remaining asterisks
    cleaned = re.sub(r'\*', '', cleaned)
    
    # Clean up extra whitespace and newlines
    cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned)  # Remove multiple empty lines
    cleaned = re.sub(r' +', ' ', cleaned)  # Remove multiple spaces
    
    # Strip leading/trailing whitespace
    cleaned = cleaned.strip()
    
    return cleaned

@app.route('/')
def index():
    """Main page with chat interface"""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests"""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'error': 'Please provide a question'}), 400
        
        if not rag_pipeline:
            return jsonify({'error': 'RAG pipeline not initialized'}), 500
        
        # Run the async query function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            response = loop.run_until_complete(rag_pipeline.query(question, top_k=5))
            print(response)
            # Clean and format the answer
            # clean_answer = clean_response(response['answer'])
            
            return jsonify({
                'answer': response['answer'],
                'question': question,
                'timestamp': response.get('timestamp', ''),
                'source_documents': response.get('source_documents', [])
            })
            
        finally:
            loop.close()
            
    except Exception as e:
        return jsonify({'error': f'Error processing request: {traceback.format_exc()}'}), 500

async def upload_and_index_documents(file, tmp_path, metadata):
    # await rag_pipeline.create_search_index()
    blob_name = file.filename  # You may want to make this unique per user/session
    blob_url = rag_pipeline.upload_pdf_to_blob_with_metadata(tmp_path, blob_name, metadata)
    await rag_pipeline.index_document(blob_name)
    print("Indexing Done")
    return True

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    if 'pdfs' not in request.files:
        return jsonify({'error': 'No PDF files provided.'}), 400

    files = request.files.getlist('pdfs')
    if len(files) == 0 or len(files) > 3:
        return jsonify({'error': 'You must upload between 1 and 3 PDF files.'}), 400

    # Get metadata fields with new names
    metadata = {
        'filename': request.form.get('field1', ''),
        'project_code': request.form.get('field2', ''),
        'label_tag': request.form.get('field3', ''),
    }

    results = []
    for file in files:
        if file.filename == '':
            results.append({'filename': '', 'status': 'No filename'})
            continue
        if not file.filename.lower().endswith('.pdf'):
            results.append({'filename': file.filename, 'status': 'Not a PDF'})
            continue

        # Save to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            file.save(tmp)
            tmp_path = tmp.name

        # Upload to Azure Blob Storage and index
        try:
            blob_name = file.filename  # You may want to make this unique per user/session

            # Index the document with metadata
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(upload_and_index_documents(file=file, tmp_path=tmp_path, metadata=metadata))
            loop.close()
            results.append({'filename': file.filename, 'status': 'Uploaded and indexed'})
        except Exception as e:
            results.append({'filename': file.filename, 'status': f'Error: {str(e)}'})
        finally:
            os.remove(tmp_path)

    return jsonify({'results': results, 'metadata': metadata})

@app.route('/view_pdf/<blob_name>')
def view_pdf(blob_name):
    """Serve PDF files with proper content type for viewing in browser"""
    try:
        if not rag_pipeline:
            return jsonify({'error': 'RAG pipeline not initialized'}), 500
        
        blob_name = blob_name.replace("@","/")
        
        print(f"Attempting to view PDF: {blob_name}")
        
        # Get the blob client for the PDF file
        blob_client = rag_pipeline.blob_service_client.get_blob_client(
            container=rag_pipeline.blob_container_name,
            blob=blob_name
        )
        
        # Check if blob exists
        try:
            blob_properties = blob_client.get_blob_properties()
            print(f"Found blob: {blob_name}, size: {blob_properties.size} bytes")
        except Exception as e:
            print(f"Blob not found: {blob_name}, error: {str(e)}")
            return jsonify({'error': f'PDF file "{blob_name}" not found'}), 404
        
        # Download the blob content
        blob_data = blob_client.download_blob()
        
        # Return the PDF with proper content type
        from flask import Response
        response = Response(
            blob_data.readall(),
            mimetype='application/pdf',
            headers={'Content-Disposition': f'inline; filename="{blob_name}"'}
        )
        
        print(f"Successfully serving PDF: {blob_name}")
        return response
        
    except Exception as e:
        print(f"Error viewing PDF {blob_name}: {str(e)}")
        return jsonify({'error': f'Error viewing PDF: {str(e)}'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'pipeline_initialized': rag_pipeline is not None})

# Initialize the RAG pipeline when the app starts
if initialize_rag_pipeline():
    print("✅ RAG pipeline initialized successfully")
else:
    print("❌ Failed to initialize RAG pipeline")

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=8000) 