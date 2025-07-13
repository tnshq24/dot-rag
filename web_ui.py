from flask import Flask, render_template, request, jsonify, session
import asyncio
import re
from azure_rag_pipeline import AzureRAGPipeline
import os
import tempfile
import traceback
import uuid
from datetime import datetime
from azure.cosmos import CosmosClient, PartitionKey
import hashlib
from dotenv import load_dotenv
from difflib import SequenceMatcher

from collections import defaultdict

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "your-secret-key-here")

# Initialize the RAG pipeline
rag_pipeline = None

# Cosmos DB configuration
COSMOS_ENDPOINT = os.environ.get("AZURE_COSMOS_DB_URI")
COSMOS_KEY = os.environ.get("AZURE_COSMOS_DB_KEY")
COSMOS_DATABASE_NAME = "chatbot_db"
COSMOS_CONTAINER_NAME = "chat_sessions"

# Initialize Cosmos DB client
cosmos_client = None
database = None
container = None


def extract_refs_dict(text: str) -> dict[str, list[int]]:
    """
    Return {filename: [pages, …], …} from
    • in-line refs like “… (foo.pdf, Page 3)”
    • bulleted refs like “- foo.pdf, Pages 3 and 20.”
    """
    pattern = re.compile(
        r"(?:\(\s*|^\s*-\s*)"  # “( …”  or  “- …” at line start
        r"(?P<file>[^,()]+?\.pdf)"  # filename ending in .pdf
        r"\s*,\s*Page(?:s)?\s*"  # “, Page ” / “, Pages ”
        r"(?P<nums>[^)\n\.]+)",  # everything up to “)” / eol / period
        flags=re.I | re.M,
    )

    refs = defaultdict(list)

    for m in pattern.finditer(text):
        filename = m.group("file").strip()
        pages = [int(n) for n in re.findall(r"\d+", m.group("nums"))]
        refs[filename].extend(pages)

    # deduplicate & sort each page list
    return {f: sorted(set(ps)) for f, ps in refs.items()}


def initialize_cosmos_db():
    """Initialize Cosmos DB connection"""
    global cosmos_client, database, container
    try:
        if COSMOS_ENDPOINT and COSMOS_KEY:
            cosmos_client = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)
            database = cosmos_client.create_database_if_not_exists(COSMOS_DATABASE_NAME)
            container = database.create_container_if_not_exists(
                id=COSMOS_CONTAINER_NAME, partition_key=PartitionKey(path="/user_id")
            )
            print("✅ Cosmos DB initialized successfully")
            return True
        else:
            print(
                "⚠️ Cosmos DB credentials not found. Session management will be disabled."
            )
            return False
    except Exception as e:
        if "Incorrect padding" in str(e):
            print(
                "❌ Failed to initialize Cosmos DB: Incorrect padding.\nThis usually means your COSMOS_KEY in ENV.txt is not copied correctly. Please ensure there are no extra spaces, line breaks, or missing characters. Copy the key exactly as provided by Azure."
            )
        else:
            print(f"❌ Failed to initialize Cosmos DB: {e}")
        return False


def generate_user_id(email):
    """Generate a consistent user ID from email"""
    return hashlib.md5(email.encode()).hexdigest()


def authenticate_user(email, password):
    """Simple authentication - in production, use proper auth"""
    # For now, hardcoded credentials as requested
    if email == "admin@xyz.com" and password == "admin":
        return True
    elif email == "user1@xyz.com" and password == "user1":
        return True
    return False


# def get_or_create_session_id():
#     """Get existing session ID or create new one"""
#     if 'session_id' not in session:
#         session['session_id'] = str(uuid.uuid4())
#     return session['session_id']


def save_chat_message(
    user_id,
    conversation_id,
    session_id,
    question,
    answer,
    timestamp,
    rephrased_question,
    retrieved_documents,
    source_documents=None,
):
    """Save chat message to Cosmos DB"""
    if not container:
        return False

    try:
        chat_message = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "conversation_id": conversation_id,
            "session_id": session_id,
            "question": question,
            "rephrased_question": rephrased_question,
            "answer": answer,
            "timestamp": timestamp,
            "source_documents": source_documents or [],
            "retrieved_documents": retrieved_documents,
            "type": "chat_message",
        }

        container.create_item(chat_message)
        return True
    except Exception as e:
        print(f"Error saving chat message: {e}")
        return False


def get_user_chat_history(user_id, limit=50):
    """Get chat history for a user"""
    if not container:
        return []

    try:
        query = f"SELECT * FROM c WHERE c.user_id = '{user_id}' AND c.type = 'chat_message' ORDER BY c.timestamp DESC OFFSET 0 LIMIT {limit}"
        items = list(
            container.query_items(query=query, enable_cross_partition_query=True)
        )
        return items
    except Exception as e:
        print(f"Error getting chat history: {e}")
        return []


def get_user_sessions(user_id):
    """Get all sessions for a user"""
    if not container:
        return []

    try:
        query = f"SELECT DISTINCT c.session_id FROM c WHERE c.user_id = '{user_id}' AND c.type = 'chat_message' ORDER BY c.timestamp DESC "
        items = list(
            container.query_items(query=query, enable_cross_partition_query=True)
        )
        for idx, session in enumerate(items):
            try:
                query_2 = f"SELECT c.question FROM c WHERE c.user_id = '{user_id}' AND c.session_id = '{session['session_id']}' AND c.type = 'chat_message' ORDER BY c.timestamp ASC"
                items_2 = list(
                    container.query_items(
                        query=query_2, enable_cross_partition_query=True
                    )
                )[0]
                items[idx]["question"] = items_2["question"]
            except:
                items[idx]["question"] = "Unknown Question"
        return items
    except Exception as e:
        print(f"Error getting user sessions: {e}")
        return []


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
    cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", response_text)

    # Remove any remaining asterisks
    cleaned = re.sub(r"\*", "", cleaned)

    # Clean up extra whitespace and newlines
    cleaned = re.sub(r"\n\s*\n", "\n\n", cleaned)  # Remove multiple empty lines
    cleaned = re.sub(r" +", " ", cleaned)  # Remove multiple spaces

    # Strip leading/trailing whitespace
    cleaned = cleaned.strip()

    return cleaned


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
    history = get_user_chat_history(user_id)
    return jsonify({"history": history})


@app.route("/user_sessions")
def user_sessions():
    """Get all sessions for authenticated user"""
    if not session.get("logged_in"):
        return jsonify({"error": "Not authenticated"}), 401

    user_id = session.get("user_id")
    sessions = get_user_sessions(user_id)
    return jsonify({"sessions": sessions})


@app.route("/chat", methods=["POST"])
def chat():
    """Handle chat requests"""
    try:
        data = request.get_json()
        question = data.get("question", "").strip()
        user_id = data.get("user_id", "").strip()
        conversation_id = data.get("conversation_id", "").strip()
        session_id = data.get("session_id", "").strip()

        if not question:
            return jsonify({"error": "Please provide a question"}), 400

        if not rag_pipeline:
            return jsonify({"error": "RAG pipeline not initialized"}), 500

        # Run the async query function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            response = loop.run_until_complete(
                rag_pipeline.query(
                    question,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    session_id=session_id,
                    top_k=5,
                )
            )
            # print(response)
            file_names = []
            for file in response["source_documents"]:
                file_names.append(file["filename"])
            # print(file_names)
            # print(response["answer"])
            # to_consider = [ans for ans in response["answer"].split("References:") if ".pdf" in ans]
            result = {}
            # if "reference" in response["answer"].lower():
            result = extract_refs_dict(response["answer"])
            # print(response["source_documents"])
            # print(result)
            relevant_sources = []
            seen_files = set()

            for filename in result:
                cleaned_source_filename = filename.split("/")[-1].strip()
                for doc in response["source_documents"]:
                    cleaned_retrieved_filename = doc["filename"].split("/")[-1].strip()
                    if (
                        cleaned_retrieved_filename == cleaned_source_filename
                        and cleaned_source_filename not in seen_files
                    ):
                        relevant_sources.append(doc)
                        seen_files.add(cleaned_source_filename)
                        break

            # print(relevant_sources)
            # print("To Consider : ", to_consider)
            # pdf_to_consider = [ans for ans in response["answer"].split() if ans in file_names]
            # if len(pdf_to_consider)!=len(to_consider):
            #     pdf_to_consider = {file for file in file_names for s_file in to_consider if SequenceMatcher(None, file, s_file).ratio() > 0.9}

            # print("PDF to Consider : ", pdf_to_consider)
            # reference_sources = []
            # added_file = []
            # for file in response["source_documents"]:
            #     if  file["filename"] in pdf_to_consider and file["filename"] not in added_file:
            #         reference_sources.append(file)
            #         added_file.append(file["filename"])

            # print(reference_sources)
            # Save chat message to Cosmos DB if user is authenticated
            if user_id and conversation_id and session_id:
                timestamp = datetime.now().isoformat()
                save_chat_message(
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


async def upload_and_index_documents(file, tmp_path, metadata):
    # await rag_pipeline.create_search_index()
    blob_name = file.filename  # You may want to make this unique per user/session
    blob_url = rag_pipeline.upload_pdf_to_blob_with_metadata(
        tmp_path, blob_name, metadata
    )

    # Extract the blob path from the returned URL to use for indexing
    # The URL format is: https://fabricbckp.blob.core.windows.net/dot-docs/Categories/{project_code}/{project_code}_{filename}.pdf
    # We need to extract: Categories/{project_code}/{project_code}_{filename}.pdf
    blob_path = blob_url.split("/dot-docs/")[-1]

    await rag_pipeline.index_document(blob_path)
    print("Indexing Done")
    return True


@app.route("/upload_pdf", methods=["POST"])
def upload_pdf():
    if "pdfs" not in request.files:
        return jsonify({"error": "No PDF files provided."}), 400

    files = request.files.getlist("pdfs")
    if len(files) == 0 or len(files) > 3:
        return jsonify({"error": "You must upload between 1 and 3 PDF files."}), 400

    # Get metadata fields with new names
    metadata = {
        "filename": request.form.get("field1", ""),
        "project_code": request.form.get("field2", ""),
        "label_tag": request.form.get("field3", ""),
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
            import asyncio

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(
                upload_and_index_documents(
                    file=file, tmp_path=tmp_path, metadata=metadata
                )
            )
            loop.close()
            results.append(
                {"filename": file.filename, "status": "Uploaded and indexed"}
            )
        except Exception as e:
            results.append({"filename": file.filename, "status": f"Error: {str(e)}"})
        finally:
            os.remove(tmp_path)

    return jsonify({"results": results, "metadata": metadata})


@app.route("/view_pdf/<blob_name>")
def view_pdf(blob_name):
    """Serve PDF files with proper content type for viewing in browser"""
    try:
        if not rag_pipeline:
            return jsonify({"error": "RAG pipeline not initialized"}), 500

        blob_name = blob_name.replace("@", "/")

        print(f"Attempting to view PDF: {blob_name}")

        # Get the blob client for the PDF file
        blob_client = rag_pipeline.blob_service_client.get_blob_client(
            container=rag_pipeline.blob_container_name, blob=blob_name
        )

        # Check if blob exists
        try:
            blob_properties = blob_client.get_blob_properties()
            print(f"Found blob: {blob_name}, size: {blob_properties.size} bytes")
        except Exception as e:
            print(f"Blob not found: {blob_name}, error: {str(e)}")
            return jsonify({"error": f'PDF file "{blob_name}" not found'}), 404

        # Download the blob content
        blob_data = blob_client.download_blob()

        # Return the PDF with proper content type
        from flask import Response

        response = Response(
            blob_data.readall(),
            mimetype="application/pdf",
            headers={"Content-Disposition": f'inline; filename="{blob_name}"'},
        )

        print(f"Successfully serving PDF: {blob_name}")
        return response

    except Exception as e:
        print(f"Error viewing PDF {blob_name}: {str(e)}")
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
    if not container:
        return jsonify({"messages": []})
    try:
        query = f"SELECT * FROM c WHERE c.user_id = '{user_id}' AND c.session_id = '{session_id}' AND c.type = 'chat_message' ORDER BY c.timestamp ASC"
        items = list(
            container.query_items(query=query, enable_cross_partition_query=True)
        )
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
    if not container:
        return jsonify({"success": False})
    try:
        # Get all messages for this session
        query = f"SELECT c.id FROM c WHERE c.user_id = '{user_id}' AND c.session_id = '{session_id}' AND c.type = 'chat_message'"
        items = list(
            container.query_items(query=query, enable_cross_partition_query=True)
        )
        for item in items:
            container.delete_item(item["id"], partition_key=user_id)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Initialize the RAG pipeline when the app starts
if initialize_rag_pipeline():
    print("✅ RAG pipeline initialized successfully")
else:
    print("❌ Failed to initialize RAG pipeline")

# Initialize Cosmos DB
initialize_cosmos_db()

if __name__ == "__main__":
    # Run the Flask app
    app.run(debug=True, host="0.0.0.0", port=8000)
