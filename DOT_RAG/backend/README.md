# Backend API Documentation

## Overview

This backend provides a Flask-based API with JWT authentication for the DOT RAG chatbot application. All routes (except `/health` and `/login`) are protected with JWT authentication.

## File Structure

```
backend/
├── app.py              # Main Flask application with JWT auth
├── main.py             # RAG pipeline implementation
├── startup.py          # Azure Web App startup script
├── web.config          # Azure Web App configuration
├── test_routes.py      # Route testing script
└── README.md           # This file
```

## Authentication

### JWT Token Flow

1. **Login**: `POST /login`
   - Returns JWT token on successful authentication
   - Token expires in 24 hours

2. **Protected Routes**: All routes except `/health` and `/login`
   - Require `Authorization: Bearer <token>` header
   - Token is validated on each request

3. **Logout**: `POST /logout`
   - Invalidates the current session
   - Frontend should clear stored token

### Authentication Middleware

The `@require_auth` decorator is applied to all protected routes:

```python
@require_auth
def protected_route():
    # Access user info from request context
    user_id = request.user_id
    user_email = request.user_email
    # ... route logic
```

## API Endpoints

### Public Routes

- `GET /health` - Health check endpoint
- `POST /login` - User authentication

### Protected Routes (Require JWT)

- `POST /logout` - User logout
- `GET /check_auth` - Check authentication status
- `POST /chat` - Send chat message
- `POST /upload_pdf` - Upload PDF files
- `GET /available_files` - Get available files
- `GET /user_sessions` - Get user sessions
- `GET /session_messages` - Get session messages
- `POST /delete_session` - Delete session
- `POST /view_highlights` - View PDF highlights
- `GET /view_pdf/<blob_name>` - View PDF file

## Environment Variables

Required environment variables for Azure deployment:

```
AZURE_SEARCH_SERVICE_NAME=your_search_service_name
AZURE_SEARCH_ADMIN_KEY=your_search_admin_key
AZURE_STORAGE_CONNECTION_STRING=your_storage_connection_string
AZURE_BLOB_CONTAINER_NAME=your_container_name
AZURE_OPENAI_ENDPOINT=your_openai_endpoint
AZURE_OPENAI_API_KEY=your_openai_api_key
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=your_embedding_deployment
AZURE_OPENAI_CHAT_DEPLOYMENT=your_chat_deployment
USE_AZURE_OPENAI=true
SECRET_KEY=your_secret_key
JWT_SECRET=your_jwt_secret
```

## CORS Configuration

CORS is configured to allow all origins (`"*"`) as requested. In production, this should be restricted to specific domains.

## Testing

Use the provided test script to verify all routes:

```bash
python test_routes.py https://your-backend-url.azurewebsites.net
```

## Deployment

### Azure Web App

1. Set environment variables in Azure Portal
2. Deploy using Azure CLI or GitHub Actions
3. The `startup.py` script handles the Flask app startup
4. `web.config` provides IIS configuration

### Local Development

```bash
# From the root directory (dot-rag)
python -m DOT_RAG.backend.app

# Or from the backend directory
cd DOT_RAG/backend
python -m backend.app
```

The server will run on `http://localhost:5001`

## Security Notes

1. JWT tokens expire after 24 hours
2. All sensitive routes are protected
3. CORS is configured for development (should be restricted in production)
4. Environment variables should be properly secured in production

## Error Handling

All endpoints return consistent error responses:

```json
{
  "error": "Error message description"
}
```

HTTP status codes:
- 200: Success
- 400: Bad Request
- 401: Unauthorized (invalid/missing JWT)
- 500: Internal Server Error 