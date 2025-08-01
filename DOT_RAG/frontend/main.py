import os
import asyncio
import traceback
import tempfile
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session, send_file, Response

from DOT_RAG.frontend.utility import (authenticate_user, generate_user_id,
                                               extract_refs_dict,
                                               get_relevant_sources, get_highlighted_pdf_content, extract_refs_dict_v2)
# from frontend.utility import (authenticate_user, generate_user_id,
#                                                extract_refs_dict,
#                                                get_relevant_sources, get_highlighted_pdf_content, extract_refs_dict_v2)
from DOT_RAG.backend.main import RunAzureRagPipeline


app = Flask(__name__)
# Configure CORS with proper settings for credentials
from flask_cors import CORS

# Configure CORS to allow credentials and specific origin
CORS(app, 
     resources={r"/*": {
         "origins": ["*"],  # Next.js development server
         "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
         "allow_headers": ["Content-Type", "Authorization"],
         "supports_credentials": True  # Important for session cookies
     }})

app.secret_key = os.environ.get("SECRET_KEY", "your-secret-key-here")



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


@app.route("/")
def index():
    """Main page with chat interface"""
    return render_template("index.html")

@app.route("/login", methods=["POST"])
def login():
    """Handle user login"""
    try:
        data = request.get_json()
        email = data.get("email", "").strip()
        password = data.get("password", "").strip()

        if not email or not password:
            return jsonify({"error": "Email and password are required"}), 400

        if authenticate_user(email, password):
            user_id = generate_user_id(email)
            session["user_id"] = user_id
            session["user_email"] = email
            session["logged_in"] = True
            return jsonify({"success": True, "user_id": user_id, "email": email})
        else:
            return jsonify({"error": "Invalid credentials"}), 401
    except Exception as e:
        return jsonify({"error": f"Login error: {str(e)}"}), 500

@app.route("/logout", methods=["POST"])
def logout():
    """Handle user logout"""
    session.clear()
    return jsonify({"success": True})

@app.route("/check_auth")
def check_auth():
    """Check if user is authenticated"""
    if session.get("logged_in"):
        if "admin" in session.get("user_email"):
            isadmin = True
        else:
            isadmin = False
        return jsonify(
            {
                "authenticated": True,
                "user_id": session.get("user_id"),
                "email": session.get("user_email"),
                "isadmin": isadmin,
            }
        )
    return jsonify({"authenticated": False})

@app.route("/chat_history")
def chat_history():
    """Get chat history for authenticated user"""
    if not session.get("logged_in"):
        return jsonify({"error": "Not authenticated"}), 401

    user_id = session.get("user_id")
    history = rag_pipeline.get_cosmo_user_chat_history(user_id)
    return jsonify({"history": history})

@app.route("/user_sessions")
def user_sessions():
    """Get all sessions for authenticated user"""
    if not session.get("logged_in"):
        return jsonify({"error": "Not authenticated"}), 401

    user_id = session.get("user_id")
    sessions = rag_pipeline.get_cosmo_user_sessions(user_id)
    return jsonify({"sessions": sessions})


@app.route("/view_highlights", methods=["POST"])
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
def chat():
    """Handle chat requests"""
    try:
        data = request.get_json()
        question = data.get("question", "").strip()
        user_id = data.get("user_id", "").strip()
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
            # print(response)

            # if "references" in response and response["references"]:
            #     response["answer"] += f"\n\nReferences:\n{response['references']}"
                
            file_names = []
            for file in response["source_documents"]:
                file_names.append(file["filename"])
            # result = extract_refs_dict(response["references"])
            result_v2 = extract_refs_dict_v2(response["references"])

            #print("Results : ", result)
            #print("Results V2 : ", result_v2)

            relevant_sources = get_relevant_sources(result=result_v2, response=response)
            # print(relevant_sources)

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
            blob_name = (
                file.filename
            )  # You may want to make this unique per user/session
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
def session_messages():
    """Get all messages for a given session_id (for authenticated user)"""
    if not session.get("logged_in"):
        return jsonify({"error": "Not authenticated"}), 401
    user_id = session.get("user_id")
    session_id = request.args.get("session_id")
    if not session_id:
        return jsonify({"error": "Missing session_id"}), 400
    try:
        items = rag_pipeline.get_cosmo_user_sessions_message(user_id=user_id, session_id=session_id)
        return jsonify({"messages": items})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/delete_session", methods=["POST"])
def delete_session():
    """Delete all messages for a given session_id (for authenticated user)"""
    if not session.get("logged_in"):
        return jsonify({"error": "Not authenticated"}), 401
    user_id = session.get("user_id")
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
def available_files():
    """Get all available files for authenticated user"""
    if not session.get("logged_in"):
        return jsonify({"error": "Not authenticated"}), 401

    try:
        # Run the async get_available_files function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            files = loop.run_until_complete(rag_pipeline.get_available_files())
            # print(f"Available files from backend: {files}")
            return jsonify({"files": files})
        finally:
            loop.close()
    except Exception as e:
        # print(f"Error in available_files endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Run the Flask app
    app.run(debug=True, host="0.0.0.0", port=5001)
