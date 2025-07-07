# Azure Web App Deployment Guide

This guide will help you deploy your Azure RAG Chatbot to Azure Web App using Linux environment.

## Prerequisites

1. **Azure Account** - You need an active Azure subscription
2. **GitHub Account** - To store your code repository
3. **Azure CLI** (optional) - For command-line deployment

## Step 1: Prepare Your Code for GitHub

### 1.1 Create GitHub Repository

1. Go to [GitHub.com](https://github.com) and sign in
2. Click the "+" icon in the top right and select "New repository"
3. Name your repository (e.g., `azure-rag-chatbot`)
4. Make it **Public** (Azure Web App can access public repos)
5. Don't initialize with README (we'll push our existing code)
6. Click "Create repository"

### 1.2 Upload Code to GitHub

Run these commands in your project directory:

```bash
# Initialize git repository
git init

# Add all files
git add .

# Commit the files
git commit -m "Initial commit: Azure RAG Chatbot"

# Add your GitHub repository as remote
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Push to GitHub
git push -u origin main
```

## Step 2: Create Azure Web App

### 2.1 Using Azure Portal

1. **Sign in to Azure Portal**
   - Go to [portal.azure.com](https://portal.azure.com)
   - Sign in with your Azure account

2. **Create Web App**
   - Click "Create a resource"
   - Search for "Web App"
   - Click "Create"

3. **Configure Web App**
   - **Subscription**: Choose your subscription
   - **Resource Group**: Create new or use existing
   - **Name**: Choose a unique name (e.g., `my-rag-chatbot`)
   - **Publish**: Code
   - **Runtime stack**: Python 3.11
   - **Operating System**: Linux
   - **Region**: Choose closest to you
   - **Pricing Plan**: Basic B1 (or higher for production)

4. **Click "Review + create" then "Create"**

### 2.2 Using Azure CLI (Alternative)

```bash
# Login to Azure
az login

# Create resource group
az group create --name my-rag-rg --location eastus

# Create web app
az webapp create \
  --resource-group my-rag-rg \
  --plan my-rag-plan \
  --name my-rag-chatbot \
  --runtime "PYTHON:3.11"
```

## Step 3: Configure Environment Variables

### 3.1 In Azure Portal

1. Go to your Web App in Azure Portal
2. In the left menu, click "Configuration"
3. Click "New application setting" for each environment variable:

```
AZURE_SEARCH_SERVICE_NAME = your_search_service_name
AZURE_SEARCH_ADMIN_KEY = your_search_admin_key
AZURE_STORAGE_CONNECTION_STRING = your_storage_connection_string
AZURE_BLOB_CONTAINER_NAME = your_container_name
AZURE_OPENAI_ENDPOINT = your_azure_openai_endpoint
AZURE_OPENAI_API_KEY = your_azure_openai_api_key
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = text-embedding-ada-002
AZURE_OPENAI_CHAT_DEPLOYMENT = gpt-4o
USE_AZURE_OPENAI = true
```

4. Click "Save" after adding all variables

### 3.2 Using Azure CLI

```bash
az webapp config appsettings set \
  --resource-group my-rag-rg \
  --name my-rag-chatbot \
  --settings \
    AZURE_SEARCH_SERVICE_NAME="your_search_service_name" \
    AZURE_SEARCH_ADMIN_KEY="your_search_admin_key" \
    AZURE_STORAGE_CONNECTION_STRING="your_storage_connection_string" \
    AZURE_BLOB_CONTAINER_NAME="your_container_name" \
    AZURE_OPENAI_ENDPOINT="your_azure_openai_endpoint" \
    AZURE_OPENAI_API_KEY="your_azure_openai_api_key" \
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT="text-embedding-ada-002" \
    AZURE_OPENAI_CHAT_DEPLOYMENT="gpt-4o" \
    USE_AZURE_OPENAI="true"
```

## Step 4: Connect GitHub to Azure Web App

### 4.1 Using Azure Portal

1. In your Web App, go to "Deployment Center"
2. Choose "GitHub" as source
3. Click "Authorize" and sign in to GitHub
4. Select your repository and branch (main)
5. Click "Save"

### 4.2 Using Azure CLI

```bash
az webapp deployment source config \
  --resource-group my-rag-rg \
  --name my-rag-chatbot \
  --repo-url https://github.com/YOUR_USERNAME/YOUR_REPO_NAME \
  --branch main \
  --manual-integration
```

## Step 5: Monitor Deployment

1. **Check Deployment Status**
   - Go to "Deployment Center" in your Web App
   - You should see deployment progress

2. **Check Logs**
   - Go to "Log stream" to see real-time logs
   - Look for any errors during startup

3. **Test Your App**
   - Once deployed, visit: `https://your-app-name.azurewebsites.net`
   - Test the chat functionality

## Step 6: Troubleshooting

### Common Issues:

1. **Environment Variables Not Set**
   - Double-check all environment variables are configured
   - Ensure no typos in variable names

2. **Import Errors**
   - Check that all packages are in `requirements.txt`
   - Verify Python version compatibility

3. **RAG Pipeline Initialization Failed**
   - Check Azure credentials are correct
   - Verify Azure services are accessible

4. **App Not Starting**
   - Check "Log stream" for error messages
   - Verify `gunicorn` is in requirements.txt

### Debug Commands:

```bash
# Check app logs
az webapp log tail --resource-group my-rag-rg --name my-rag-chatbot

# Restart app
az webapp restart --resource-group my-rag-rg --name my-rag-chatbot

# Check app settings
az webapp config appsettings list --resource-group my-rag-rg --name my-rag-chatbot
```

## Step 7: Custom Domain (Optional)

1. Go to "Custom domains" in your Web App
2. Add your domain name
3. Configure DNS records as instructed

## Security Best Practices

1. **Use Azure Key Vault** for sensitive environment variables
2. **Enable HTTPS** (automatic with Azure Web App)
3. **Set up monitoring** with Application Insights
4. **Regular updates** of dependencies

## Cost Optimization

1. **Use Basic plan** for development
2. **Scale down** when not in use
3. **Monitor usage** in Azure Portal

## Next Steps

1. **Set up CI/CD** with GitHub Actions
2. **Add monitoring** with Application Insights
3. **Implement authentication** if needed
4. **Add custom domain** for production use

Your Azure RAG Chatbot should now be live at: `https://your-app-name.azurewebsites.net` 