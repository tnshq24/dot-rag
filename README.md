# Azure RAG Pipeline Web UI

A simple web interface for interacting with the Azure RAG (Retrieval-Augmented Generation) pipeline chatbot.

## 🚀 Quick Deploy to Azure

This project is ready for deployment to Azure Web App. See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.

### Quick Start:
1. Upload code to GitHub
2. Create Azure Web App (Linux, Python 3.11)
3. Configure environment variables
4. Connect GitHub repository
5. Deploy!

**Live Demo**: [Your App URL will be here after deployment]

## Features

- 🤖 Clean chat interface for the Azure RAG pipeline
- 📝 Automatic response formatting (removes markdown formatting)
- 📱 Responsive design that works on desktop and mobile
- ⚡ Real-time chat with loading indicators
- 🎨 Modern UI with gradient design

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Configuration

Make sure you have your `ENV.txt` file with all the required Azure and OpenAI credentials:

```
AZURE_SEARCH_SERVICE_NAME=your_search_service_name
AZURE_SEARCH_ADMIN_KEY=your_search_admin_key
AZURE_STORAGE_CONNECTION_STRING=your_storage_connection_string
AZURE_BLOB_CONTAINER_NAME=your_container_name
AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o
USE_AZURE_OPENAI=true
```

### 3. Run the Web Application

```bash
python web_ui.py
```

The web interface will be available at: `http://localhost:5000`

## Usage

1. **Open the Web Interface**: Navigate to `http://localhost:5000` in your browser
2. **Ask Questions**: Type your questions in the chat input and press Enter or click Send
3. **View Responses**: The chatbot will provide clean, formatted answers based on your indexed documents

## Response Formatting

The web UI automatically:
- Removes markdown bold formatting (`**text**` → `text`)
- Cleans up extra whitespace and formatting
- Preserves the structure of lists and numbered items
- Makes responses more readable

## Example

**Input**: "find me the village name of district bastar and sub district kondagaon"

**Output**: 
```
Based on the provided context documents, the villages in the Bastar district and Kondagaon sub-district are as follows:

1. Nahkanar
   - Village Code: 448719
   - Population: 586

2. Pala
   - Village Code: 448662
   - Population: 744

... (and so on)
```

## File Structure

```
microsoft_rag/
├── azure_rag_pipeline.py    # Main RAG pipeline implementation
├── web_ui.py               # Flask web application
├── requirements.txt         # Python dependencies
├── templates/
│   └── index.html         # Chat interface HTML template
└── ENV.txt                # Environment variables (not in repo)
```

## Troubleshooting

- **Pipeline Initialization Error**: Check your `ENV.txt` file and ensure all credentials are correct
- **Network Error**: Make sure the Flask server is running and accessible
- **Empty Responses**: Verify that your documents are properly indexed in Azure AI Search

## API Endpoints

- `GET /` - Main chat interface
- `POST /chat` - Send chat messages (JSON: `{"question": "your question"}`)
- `GET /health` - Health check endpoint

## Development

To modify the response formatting, edit the `clean_response()` function in `web_ui.py`. The function currently:
- Removes markdown bold formatting
- Cleans up whitespace
- Makes responses more readable 