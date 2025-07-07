from flask import Flask, render_template, request, jsonify
import asyncio
import re
from azure_rag_pipeline import AzureRAGPipeline

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
            
            # Clean and format the answer
            clean_answer = clean_response(response['answer'])
            
            return jsonify({
                'answer': clean_answer,
                'question': question,
                'timestamp': response.get('timestamp', '')
            })
            
        finally:
            loop.close()
            
    except Exception as e:
        return jsonify({'error': f'Error processing request: {str(e)}'}), 500

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
    app.run(debug=True, host='0.0.0.0', port=5000) 