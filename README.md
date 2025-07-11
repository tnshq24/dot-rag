# Azure RAG Pipeline Web UI

A modern ChatGPT-style web interface for interacting with the Azure RAG (Retrieval-Augmented Generation) pipeline chatbot.

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

- 🤖 **ChatGPT-style Interface**: Modern, responsive UI with dark theme
- 👤 **User Authentication**: Login system with session management
- 💬 **Chat History**: Persistent conversation history with Cosmos DB
- 📁 **Resizable Sidebar**: Collapsible left panel for chat sessions
- 📤 **PDF Upload**: Upload and index PDF documents (for authenticated users)
- 📱 **Responsive Design**: Works seamlessly on desktop and mobile
- ⚡ **Real-time Chat**: Live responses with loading indicators
- 🎨 **Modern UI**: Clean, professional interface with smooth animations

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Configuration

Make sure you have your `ENV.txt` file with all the required Azure, OpenAI, and Cosmos DB credentials:

```
# Azure Search Service Configuration
AZURE_SEARCH_SERVICE_NAME=your_search_service_name
AZURE_SEARCH_ADMIN_KEY=your_search_admin_key

# Azure Storage Configuration
AZURE_STORAGE_CONNECTION_STRING=your_storage_connection_string
AZURE_BLOB_CONTAINER_NAME=your_container_name

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o
USE_AZURE_OPENAI=true

# Cosmos DB Configuration (for session management)
COSMOS_ENDPOINT=your_cosmos_db_endpoint
COSMOS_KEY=your_cosmos_db_key

# Flask Secret Key (for session management)
SECRET_KEY=your_secret_key_here_change_this_in_production
```

### 3. Run the Web Application

```bash
python web_ui.py
```

The web interface will be available at: `http://localhost:8000`

## Usage

### Authentication
- **Default Credentials**: 
  - Email: `admin@xyz.com`
  - Password: `admin`
- Click on the user avatar in the bottom-left to login
- Upload PDF functionality is only available for authenticated users

### Chat Interface
1. **Start a New Chat**: Click the "New Chat" button in the sidebar
2. **Ask Questions**: Type your questions in the chat input and press Enter
3. **View History**: Previous chat sessions are available in the left sidebar
4. **Upload PDFs**: Click "Upload PDF" button (only for logged-in users)

### Session Management
- Each browser session gets a unique `session_id`
- Each conversation gets a unique `conversation_id`
- Chat history is persisted in Cosmos DB
- Sessions are automatically managed and displayed in the sidebar

## API Endpoints

### Chat & Authentication
- `GET /` - Main chat interface
- `POST /chat` - Send chat messages with session tracking
- `POST /login` - User authentication
- `POST /logout` - User logout
- `GET /check_auth` - Check authentication status

### Session Management
- `GET /chat_history` - Get user's chat history
- `GET /user_sessions` - Get user's chat sessions

### File Management
- `POST /upload_pdf` - Upload and index PDF documents
- `GET /view_pdf/<blob_name>` - View PDF files

### Health Check
- `GET /health` - Health check endpoint

## Chat API Payload

The `/chat` endpoint now accepts extended parameters for session tracking:

```json
{
  "question": "Your question here",
  "user_id": "abc123",
  "conversation_id": "564121",
  "session_id": "session1"
}
```

**Example Session Flow**:
```json
// First two questions in the same session
{"user_id": "abc", "conversation_id": "564121", "session_id": "session1"}
{"user_id": "abc", "conversation_id": "89789456454", "session_id": "session1"}

// After page refresh
{"user_id": "abc", "conversation_id": "5375342", "session_id": "session2"}
```

## UI Features

### Left Sidebar
- **Collapsible**: Click the hamburger menu to collapse/expand
- **Chat History**: Displays previous chat sessions
- **New Chat**: Start fresh conversations
- **User Section**: Login/logout functionality with user avatar

### Main Chat Area
- **Modern Design**: Dark theme with ChatGPT-like styling
- **Auto-scroll**: Automatically scrolls to new messages
- **Loading States**: Visual feedback during processing
- **Markdown Support**: Rich text formatting for responses
- **Reference Documents**: Links to source PDFs when available

### Responsive Design
- **Desktop**: Full sidebar and chat area
- **Mobile**: Collapsible sidebar with touch-friendly interface
- **Tablet**: Adaptive layout for medium screens

## File Structure

```
microsoft_rag/
├── azure_rag_pipeline.py    # Main RAG pipeline implementation
├── web_ui.py               # Flask web application with session management
├── requirements.txt         # Python dependencies
├── templates/
│   └── index.html         # ChatGPT-style chat interface
├── ENV.txt                # Environment variables (not in repo)
└── DEPLOYMENT.md          # Deployment instructions
```

## Troubleshooting

- **Pipeline Initialization Error**: Check your `ENV.txt` file and ensure all credentials are correct
- **Cosmos DB Connection**: Verify your Cosmos DB endpoint and key are correct
- **Cosmos DB 'Incorrect padding' Error**: This usually means your `COSMOS_KEY` in `ENV.txt` is not copied correctly. Make sure there are no extra spaces, line breaks, or missing characters. Copy the key exactly as provided by Azure.
- **Authentication Issues**: Ensure the SECRET_KEY is set for session management
- **Network Error**: Make sure the Flask server is running and accessible
- **Empty Responses**: Verify that your documents are properly indexed in Azure AI Search

## Development

### Adding New Features
- **UI Changes**: Modify `templates/index.html` for frontend updates
- **Backend Logic**: Edit `web_ui.py` for server-side functionality
- **Session Management**: Use the Cosmos DB functions for data persistence

### Customization
- **Styling**: Update CSS in the `<style>` section of `index.html`
- **Authentication**: Modify the `authenticate_user()` function in `web_ui.py`
- **Session Logic**: Customize the session management functions as needed 