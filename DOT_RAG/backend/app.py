import os
import asyncio
import traceback
import tempfile
import jwt
from datetime import datetime, timedelta
from functools import wraps
from flask import Flask, render_template, request, jsonify, session, send_file, Response
from flask_cors import CORS

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from DOT_RAG.frontend.utility import (authenticate_user, generate_user_id,
                                               extract_refs_dict,
                                               get_relevant_sources, get_highlighted_pdf_content, extract_refs_dict_v2)
from DOT_RAG.backend.main import RunAzureRagPipeline

app = Flask(__name__)

# Configure CORS with proper settings for credentials
CORS(app, 
     resources={r"/*": {
         "origins": ["*"],  # Allow all origins for now
         "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
         "allow_headers": ["Content-Type", "Authorization"],
         "supports_credentials": True
     }})

app.secret_key = os.environ.get("SECRET_KEY", "your-secret-key-here")
JWT_SECRET = os.environ.get("JWT_SECRET", "your-jwt-secret-here")

# Initialize RAG pipeline
try:
    rag_pipeline = RunAzureRagPipeline()
    print("✅ RAG pipeline initialized successfully")
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
    print("❌ Failed to initialize RAG pipeline")

def generate_jwt_token(user_id, email):
    """Generate JWT token for user"""
    payload = {
        'user_id': user_id,
        'email': email,
        'exp': datetime.utcnow() + timedelta(hours=24),  # Token expires in 24 hours
        'iat': datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET, algorithm='HS256')

def verify_jwt_token(token):
    """Verify JWT token and return payload"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def require_auth(f):
    """Decorator to require JWT authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        
        if not auth_header:
            return jsonify({"error": "No authorization header"}), 401
        
        try:
            token = auth_header.split(' ')[1]  # Bearer <token>
            payload = verify_jwt_token(token)
            
            if not payload:
                return jsonify({"error": "Invalid or expired token"}), 401
            
            # Add user info to request context
            request.user_id = payload['user_id']
            request.user_email = payload['email']
            
            return f(*args, **kwargs)
        except IndexError:
            return jsonify({"error": "Invalid authorization header format"}), 401
        except Exception as e:
            return jsonify({"error": f"Authentication error: {str(e)}"}), 401
    
    return decorated_function

@app.route("/")
def index():
    """Main page with chat interface"""
    return render_template("index.html")

@app.route("/login", methods=["POST"])
def login():
    """Handle user login and return JWT token"""
    try:
        data = request.get_json()
        email = data.get("email", "").strip()
        password = data.get("password", "").strip()

        if not email or not password:
            return jsonify({"error": "Email and password are required"}), 400

        if authenticate_user(email, password):
            user_id = generate_user_id(email)
            token = generate_jwt_token(user_id, email)
            
            return jsonify({
                "success": True, 
                "user_id": user_id, 
                "email": email,
                "token": token
            })
        else:
            return jsonify({"error": "Invalid credentials"}), 401
    except Exception as e:
        return jsonify({"error": f"Login error: {str(e)}"}), 500

@app.route("/logout", methods=["POST"])
@require_auth
def logout():
    """Handle user logout"""
    return jsonify({"success": True})

@app.route("/check_auth")
@require_auth
def check_auth():
    """Check if user is authenticated"""
    isadmin = "admin" in request.user_email
    return jsonify({
        "authenticated": True,
        "user_id": request.user_id,
        "email": request.user_email,
        "isadmin": isadmin,
    })

@app.route("/chat_history")
@require_auth
def chat_history():
    """Get chat history for authenticated user"""
    user_id = request.user_id
    history = rag_pipeline.get_cosmo_user_chat_history(user_id)
    return jsonify({"history": history})

@app.route("/user_sessions")
@require_auth
def user_sessions():
    """Get all sessions for authenticated user"""
    user_id = request.user_id
    sessions = rag_pipeline.get_cosmo_user_sessions(user_id)
    return jsonify({"sessions": sessions})

@app.route("/view_highlights", methods=["POST"])
@require_auth
def view_highlights():
    source = request.get_json()
    if not source:
        return jsonify({"error": "No data provided"}), 400
    
    # Validate required fields
    if not source.get("filename") or not source.get("page_number") or not source.get("content"):
        missing_fields = []
        if not source.get("filename"):
            missing_fields.append("filename")
        if not source.get("page_number"):
            missing_fields.append("page_number")
        if not source.get("content"):
            missing_fields.append("content")
        return jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"}), 400

    try:
        pages_content = rag_pipeline._extract_text_from_pdf_blob(source.get("filename"))
        if len(pages_content) == 0:
            pages_content = rag_pipeline._extract_using_document_intelligence(
                blob_name=source.get("filename"),
                return_raw=True
            )
            source["pages_content"] = pages_content

        # Check if RAG pipeline is initialized
        if not rag_pipeline:
            return jsonify({"error": "RAG pipeline not initialized"}), 500
        
        output_pdf_io, found = get_highlighted_pdf_content(rag_pipeline=rag_pipeline, source=source)
        
        # Create response with page number in header
        response = send_file(
            output_pdf_io,
            mimetype='application/pdf',
            as_attachment=False,
            download_name=source["filename"]
        )
        response.headers['X-Page-Number'] = str(min(source["page_number"]))
        return response
    except Exception as e:
        print(f"Error in first attempt: {str(e)}")
        try:
            output_pdf_io, found = get_highlighted_pdf_content(
                rag_pipeline=rag_pipeline, source=source,
                try_highlight=False
            )
            # Create response with page number in header
            response = send_file(
                output_pdf_io,
                mimetype='application/pdf',
                as_attachment=False,
                download_name=source["filename"]
            )
            response.headers['X-Page-Number'] = str(min(source["page_number"]))
            return response
        except Exception as e:
            print(f"Error in second attempt: {str(e)}")
            try:
                output_pdf_io, found = get_highlighted_pdf_content(
                    rag_pipeline=rag_pipeline, source=source,
                    try_highlight=False
                )
                # Create response with page number in header
                response = send_file(
                    output_pdf_io,
                    mimetype='application/pdf',
                    as_attachment=False,
                    download_name=source["filename"]
                )
                return response
            except Exception as e:
                print(f"Error in view_highlights: {str(e)}")
                # Try to return a simple error response
                try:
                    return jsonify({"error": f"Error processing highlights: {str(e)}"}), 500
                except:
                    # If even JSON response fails, return a simple text response
                    return f"Error processing highlights: {str(e)}", 500

@app.route("/chat", methods=["POST"])
@require_auth
def chat():
    """Handle chat requests"""
    try:
        data = request.get_json()
        question = data.get("question", "").strip()
        user_id = request.user_id  # Get from JWT token
        conversation_id = data.get("conversation_id", "").strip()
        session_id = data.get("session_id", "").strip()
        file_names = data.get("file_names", [])  # New parameter for selected files

        if not question:
            return jsonify({"error": "Please provide a question"}), 400

        if not rag_pipeline:
            return jsonify({"error": "RAG pipeline not initialized"}), 500

        # Run the async query function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # If file_names is provided, use the first one as file_name parameter
            file_name = file_names[0] if file_names and len(file_names) > 0 else None
            
            response = loop.run_until_complete(
                rag_pipeline.query(
                    question,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    session_id=session_id,
                    file_name=file_name,  # Pass the selected file name
                    top_k=8,
                )
            )

            file_names = []
            for file in response["source_documents"]:
                file_names.append(file["filename"])
            
            result_v2 = extract_refs_dict_v2(response["references"])
            relevant_sources = get_relevant_sources(result=result_v2, response=response)

            # Save chat message to Cosmos DB if user is authenticated
            if user_id and conversation_id and session_id:
                timestamp = datetime.now().isoformat()
                rag_pipeline.save_cosmo_chat_message(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    session_id=session_id,
                    question=question,
                    answer=response["answer"],
                    timestamp=timestamp,
                    rephrased_question=response["rephrased_question"],
                    retrieved_documents=response["source_documents"],
                    source_documents=relevant_sources,
                )
            return jsonify(
                {
                    "answer": response["answer"],
                    "question": question,
                    "timestamp": response.get("timestamp", ""),
                    "source_documents": relevant_sources,
                }
            )
        finally:
            loop.close()
    except Exception as e:
        return (
            jsonify({"error": f"Error processing request: {traceback.format_exc()}"}),
            500,
        )

@app.route("/upload_pdf", methods=["POST"])
@require_auth
def upload_pdf():
    if "pdfs" not in request.files:
        return jsonify({"error": "No PDF files provided."}), 400

    files = request.files.getlist("pdfs")
    if len(files) == 0 or len(files) > 1:
        return jsonify({"error": "You must upload only 1 files."}), 400

    # Get blob_kwargs fields with new names
    blob_kwargs = {
        "from_ui": True,
        "meta_data": {
            "filename": request.form.get("field1", ""),
            "project_code": request.form.get("field2", ""),
            "label_tag": request.form.get("field3", "")
        }
    }

    results = []
    for file in files:
        if file.filename == "":
            results.append({"filename": "", "status": "No filename"})
            continue
        if not file.filename.lower().endswith(".pdf"):
            results.append({"filename": file.filename, "status": "Not a PDF"})
            continue

        # Save to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            file.save(tmp)
            tmp_path = tmp.name

        # Upload to Azure Blob Storage and index
        try:
            blob_name = file.filename
            # Index the document with metadata
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(
                rag_pipeline.run(
                    upload_to_blob=True,
                    pdf_path=tmp_path,
                    blob_name=blob_name,
                    index_document=True,
                    blob_kwargs=blob_kwargs,
                )
            )
            loop.close()
            results.append(
                {"filename": file.filename, "status": "Uploaded and indexed"}
            )
            status_code = 200
        except Exception as e:
            status_code = 400
            results.append({"filename": file.filename, "status": f"Error: {str(e)}"})
        finally:
            os.remove(tmp_path)
    return jsonify({"results": results, "metadata": blob_kwargs["meta_data"]}, status_code)

@app.route("/view_pdf/<blob_name>")
@require_auth
def view_pdf(blob_name):
    """Serve PDF files with proper content type for viewing in browser"""
    try:
        if not rag_pipeline:
            return jsonify({"error": "RAG pipeline not initialized"}), 500

        blob_name = blob_name.replace("@", "/")
        blob_data = rag_pipeline.get_pdf_content_from_blob(blob_name=blob_name)
        # Return the PDF with proper content type
        response = Response(
            blob_data,
            mimetype="application/pdf",
            headers={"Content-Disposition": f'inline; filename="{blob_name}"'},
        )
        return response

    except Exception as e:
        return jsonify({"error": f"Error viewing PDF: {str(e)}"}), 500

@app.route("/health")
def health():
    """Health check endpoint"""
    return jsonify(
        {"status": "healthy", "pipeline_initialized": rag_pipeline is not None}
    )

@app.route("/session_messages")
@require_auth
def session_messages():
    """Get all messages for a given session_id (for authenticated user)"""
    user_id = request.user_id
    session_id = request.args.get("session_id")
    if not session_id:
        return jsonify({"error": "Missing session_id"}), 400
    try:
        items = rag_pipeline.get_cosmo_user_sessions_message(user_id=user_id, session_id=session_id)
        return jsonify({"messages": items})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/delete_session", methods=["POST"])
@require_auth
def delete_session():
    """Delete all messages for a given session_id (for authenticated user)"""
    user_id = request.user_id
    data = request.get_json()
    session_id = data.get("session_id")
    if not session_id:
        return jsonify({"error": "Missing session_id"}), 400
    try:
        # Get all messages for this session
        status = rag_pipeline.delete_cosmo_chat_message(user_id=user_id, session_id=session_id)
        if status:
            return jsonify({"success": True})
        else:
            return jsonify({"error": "Error deleting chat message"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/available_files")
@require_auth
def available_files():
    """Get all available files for authenticated user"""
    try:
        # Run the async get_available_files function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            files = loop.run_until_complete(rag_pipeline.get_available_files())
            return jsonify({"files": files})
        finally:
            loop.close()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Run the Flask app
    app.run(debug=True, host="0.0.0.0", port=5001) 