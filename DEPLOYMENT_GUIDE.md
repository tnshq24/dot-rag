# Deployment Guide

## Backend Deployment (Azure Web App)

### Prerequisites
1. Azure subscription
2. Azure CLI installed
3. Python 3.8+ environment

### Steps

1. **Prepare the backend code**
   ```bash
   cd dot-rag/DOT_RAG
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set environment variables in Azure Web App**
   - Go to Azure Portal
   - Navigate to your Web App
   - Go to Configuration > Application settings
   - Add the following environment variables:
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

4. **Deploy to Azure Web App**
   ```bash
   # Using Azure CLI
   az webapp up --name your-webapp-name --resource-group your-resource-group --runtime "PYTHON:3.9"
   ```

5. **Test the backend**
   ```bash
   curl https://your-webapp-name.azurewebsites.net/health
   ```

## Frontend Deployment (Vercel)

### Prerequisites
1. Vercel account
2. Vercel CLI installed
3. Node.js 18+

### Steps

1. **Navigate to frontend directory**
   ```bash
   cd dot-rag/fe_2
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Set environment variables**
   - Create a `.env.local` file:
     ```
     NEXT_PUBLIC_API_URL=https://your-backend-url.azurewebsites.net
     ```

4. **Deploy to Vercel**
   ```bash
   # Install Vercel CLI if not already installed
   npm i -g vercel

   # Deploy
   vercel deploy
   ```

5. **Update API URL in production**
   - After getting the backend URL, update the constants file:
   ```typescript
   // In fe_2/constants/index.ts
   export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "https://your-backend-url.azurewebsites.net"
   ```

## Testing Routes

After deployment, test all routes:

### Backend Routes (Protected with JWT)
- `POST /login` - Login and get JWT token
- `POST /logout` - Logout (requires JWT)
- `GET /check_auth` - Check authentication (requires JWT)
- `POST /chat` - Send chat message (requires JWT)
- `POST /upload_pdf` - Upload PDF (requires JWT)
- `GET /available_files` - Get available files (requires JWT)
- `GET /user_sessions` - Get user sessions (requires JWT)
- `GET /session_messages` - Get session messages (requires JWT)
- `POST /delete_session` - Delete session (requires JWT)
- `POST /view_highlights` - View PDF highlights (requires JWT)
- `GET /view_pdf/<blob_name>` - View PDF (requires JWT)
- `GET /health` - Health check (public)

### Frontend Routes
- `/` - Main chat interface
- `/login` - Login page
- All other routes handled by Next.js

## CORS Configuration

The backend is configured with CORS set to `"*"` for all origins as requested. In production, you should restrict this to your specific frontend domain.

## Authentication Flow

1. User logs in via `/login` endpoint
2. Backend returns JWT token
3. Frontend stores token in localStorage
4. All subsequent requests include token in Authorization header
5. Backend validates token for protected routes

## Troubleshooting

### Backend Issues
1. Check Azure Web App logs
2. Verify environment variables are set correctly
3. Ensure all Azure services are properly configured

### Frontend Issues
1. Check Vercel deployment logs
2. Verify API URL is correct
3. Check browser console for CORS errors

### Authentication Issues
1. Verify JWT token is being sent in headers
2. Check token expiration
3. Ensure JWT_SECRET is set correctly 