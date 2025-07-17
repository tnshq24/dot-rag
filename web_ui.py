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

from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import send_file
try:
    import fitz
except ImportError:
    print("PyMuPDF not found. Installing...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PyMuPDF"])
    import fitz

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "your-secret-key-here")

# Initialize the RAG pipeline
rag_pipeline = None

# Cosmos DB configuration
COSMOS_ENDPOINT = os.environ.get("AZURE_COSMOS_DB_URI")
COSMOS_KEY = os.environ.get("AZURE_COSMOS_DB_KEY")
COSMOS_DATABASE_NAME = os.environ.get("AZURE_COSMOS_DB_DATABASE_NAME", "chatbot_db")
COSMOS_CONTAINER_NAME = os.environ.get("AZURE_COSMOS_DB_CHAT_HISTORY_CONTAINER", "chat_sessions")

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
                "❌ Failed to initialize Cosmos DB: Incorrect padding.\nThis usually means your AZURE_COSMOS_DB_KEY in ENV.txt is not copied correctly. Please ensure there are no extra spaces, line breaks, or missing characters. Copy the key exactly as provided by Azure."
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

# @app.route("/view_highlights", methods=["POST"])
# def view_highlights():
#     try:
#         source = request.get_json()
#         if not source:
#             return jsonify({"error": "No data provided"}), 400
        
#         print(f"View highlights request: {source}")
#         print(f"Filename: {source.get('filename')}")
#         print(f"Page number: {source.get('page_number')}")
#         print(f"Content length: {len(source.get('content', ''))}")
        
#         # Validate required fields
#         if not source.get("filename") or not source.get("page_number") or not source.get("content"):
#             missing_fields = []
#             if not source.get("filename"):
#                 missing_fields.append("filename")
#             if not source.get("page_number"):
#                 missing_fields.append("page_number")
#             if not source.get("content"):
#                 missing_fields.append("content")
#             return jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"}), 400
        
#         # Check if RAG pipeline is initialized
#         if not rag_pipeline:
#             return jsonify({"error": "RAG pipeline not initialized"}), 500
        
#         final_sources = []
#         try:
#             blob_client = rag_pipeline.blob_service_client.get_blob_client(
#                 container=rag_pipeline.blob_container_name, blob=source["filename"]
#             )
#             # Download the PDF content from blob storage
#             pdf_content = blob_client.download_blob().readall()
#         except Exception as e:
#             print(f"Error accessing blob storage: {str(e)}")
#             return jsonify({"error": f"Error accessing PDF file: {str(e)}"}), 500

#         target_page=int(source["page_number"])-1

#         # Create a BytesIO object to read the PDF content
#         try:
#             doc = fitz.open(stream=pdf_content, filetype="pdf")
#         except Exception as e:
#             print(f"Error opening PDF: {str(e)}")
#             return jsonify({"error": f"Error processing PDF: {str(e)}"}), 500
#         found = False
        
#         try:
#             for page_num, page in enumerate(doc):
#                 if target_page is not None and page_num != target_page:
#                     continue

#                 # Get page text
#                 page_text = page.get_text()
#                 if not page_text.strip():
#                     continue

#                 # Try to find and highlight the content
#                 try:
#                     chunks = [source["content"]] + rag_pipeline.chunk_text(text=page_text)
#                     vectorizer = TfidfVectorizer()
#                     vect_text = vectorizer.fit_transform(chunks)
#                     similarities = cosine_similarity(vect_text[0:1], vect_text[1:]).flatten()
#                     best_match_index = similarities.argmax() + 1
#                     similar_text = chunks[best_match_index]
                    
#                     # Search for the text in the page
#                     text_instances = page.search_for(similar_text)
#                     if text_instances:
#                         for inst in text_instances:
#                             highlight = page.add_highlight_annot(inst)
#                             highlight.update()
#                         found = True
#                         print(f"Successfully highlighted text on page {page_num + 1}")
#                         break
#                     else:
#                         print(f"No text instances found on page {page_num + 1}")
#                 except Exception as e:
#                     print(f"Error highlighting on page {page_num + 1}: {str(e)}")
#                     # Continue to next page if highlighting fails
#                     continue
#         except Exception as e:
#             print(f"Error in highlighting process: {str(e)}")
#             # Continue without highlighting

#         try:
#             output_pdf_io = BytesIO()
#             doc.save(output_pdf_io)
#             doc.close()
#             output_pdf_io.seek(0)
#         except Exception as e:
#             print(f"Error saving PDF: {str(e)}")
#             return jsonify({"error": f"Error saving PDF: {str(e)}"}), 500
        
#         # Create response with page number in header
#         response = send_file(
#             output_pdf_io,
#             mimetype='application/pdf',
#             as_attachment=False,
#             download_name=source["filename"]
#         )
#         response.headers['X-Page-Number'] = str(target_page+1)
        
#         if found:
#             print("Returning highlighted PDF")
#         else:
#             print("Returning PDF without highlighting (highlighting failed)")
        
#         return response
#     except Exception as e:
#         print(f"Error in first attempt: {str(e)}")
#         try:
#             blob_client = rag_pipeline.blob_service_client.get_blob_client(
#                 container=rag_pipeline.blob_container_name, blob=source["filename"]
#             )
#             # Download the PDF content from blob storage
#             pdf_content = blob_client.download_blob().readall()

#             target_page=int(source["page_number"])-1

#             # Create a BytesIO object to read the PDF content
#             doc = fitz.open(stream=pdf_content, filetype="pdf")
#             output_pdf_io = BytesIO()
#             doc.save(output_pdf_io)
#             doc.close()
#             output_pdf_io.seek(0)
#             # Create response with page number in header
#             response = send_file(
#                 output_pdf_io,
#                 mimetype='application/pdf',
#                 as_attachment=False,
#                 download_name=source["filename"]
#             )
#             response.headers['X-Page-Number'] = str(target_page+1)
#         except Exception as e2:
#             print(f"Error in second attempt: {str(e2)}")
#             # Final fallback - just return the original PDF without highlighting
#             blob_client = rag_pipeline.blob_service_client.get_blob_client(
#                 container=rag_pipeline.blob_container_name, blob=source["filename"]
#             )
#             # Download the PDF content from blob storage
#             pdf_content = blob_client.download_blob().readall()

#             target_page=int(source["page_number"])-1

#             # Create a BytesIO object to read the PDF content
#             doc = fitz.open(stream=pdf_content, filetype="pdf")
#             output_pdf_io = BytesIO()
#             doc.save(output_pdf_io)
#             doc.close()
#             output_pdf_io.seek(0)
#             # Create response with page number in header
#             response = send_file(
#                 output_pdf_io,
#                 mimetype='application/pdf',
#                 as_attachment=False,
#                 download_name=source["filename"]
#             )
#         return response
#     except Exception as e:
#         print(f"Error in view_highlights: {str(e)}")
#         # Try to return a simple error response
#         try:
#             return jsonify({"error": f"Error processing highlights: {str(e)}"}), 500
#         except:
#             # If even JSON response fails, return a simple text response
#             return f"Error processing highlights: {str(e)}", 500


@app.route("/view_highlights", methods=["POST"])
def view_highlights():
    try:
        source = request.get_json()
        if not source:
            return jsonify({"error": "No data provided"}), 400
        
        print(f"View highlights request: {source}")
        print(f"Filename: {source.get('filename')}")
        print(f"Page number: {source.get('page_number')}")
        print(f"Content length: {len(source.get('content', ''))}")
        
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
        
        # Check if RAG pipeline is initialized
        if not rag_pipeline:
            return jsonify({"error": "RAG pipeline not initialized"}), 500
        
        all_content = source["content"]
        all_pages = source["page_number"]


        try:
            blob_client = rag_pipeline.blob_service_client.get_blob_client(
                container=rag_pipeline.blob_container_name, blob=source["filename"]
            )
            # Download the PDF content from blob storage
            pdf_content = blob_client.download_blob().readall()
        except Exception as e:
            print(f"Error accessing blob storage: {str(e)}")
            return jsonify({"error": f"Error accessing PDF file: {str(e)}"}), 500
        
        # Create a BytesIO object to read the PDF content
        try:
            doc = fitz.open(stream=pdf_content, filetype="pdf")
        except Exception as e:
            print(f"Error opening PDF: {str(e)}")
            return jsonify({"error": f"Error processing PDF: {str(e)}"}), 500
        found = False

        for idx, content in enumerate(all_content):

            target_page = all_pages[idx] - 1
            try:
                for page_num, page in enumerate(doc):
                    if target_page is not None and page_num != target_page:
                        continue

                    # Get page text
                    page_text = page.get_text()
                    if not page_text.strip():
                        continue

                    # Try to find and highlight the content
                    try:
                        chunks = [content] + rag_pipeline.chunk_text(text=page_text)
                        vectorizer = TfidfVectorizer()
                        vect_text = vectorizer.fit_transform(chunks)
                        similarities = cosine_similarity(vect_text[0:1], vect_text[1:]).flatten()
                        best_match_index = similarities.argmax() + 1
                        similar_text = chunks[best_match_index]
                        
                        # Search for the text in the page
                        text_instances = page.search_for(similar_text)
                        if text_instances:
                            for inst in text_instances:
                                highlight = page.add_highlight_annot(inst)
                                highlight.update()
                            found = True
                            print(f"Successfully highlighted text on page {page_num + 1}")
                            break
                        else:
                            print(f"No text instances found on page {page_num + 1}")
                    except Exception as e:
                        print(f"Error highlighting on page {page_num + 1}: {str(e)}")
                        # Continue to next page if highlighting fails
                        continue
            except Exception as e:
                print(f"Error in highlighting process: {str(e)}")
                # Continue without highlighting

        try:
            output_pdf_io = BytesIO()
            doc.save(output_pdf_io)
            doc.close()
            output_pdf_io.seek(0)
        except Exception as e:
            print(f"Error saving PDF: {str(e)}")
            return jsonify({"error": f"Error saving PDF: {str(e)}"}), 500
        
        # Create response with page number in header
        response = send_file(
            output_pdf_io,
            mimetype='application/pdf',
            as_attachment=False,
            download_name=source["filename"]
        )
        response.headers['X-Page-Number'] = str(all_pages[0])
        
        if found:
            print("Returning highlighted PDF")
        else:
            print("Returning PDF without highlighting (highlighting failed)")
        
        return response
    except Exception as e:
        print(f"Error in first attempt: {str(e)}")
        try:
            blob_client = rag_pipeline.blob_service_client.get_blob_client(
                container=rag_pipeline.blob_container_name, blob=source["filename"]
            )
            # Download the PDF content from blob storage
            pdf_content = blob_client.download_blob().readall()

            target_page=int(source["page_number"])-1

            # Create a BytesIO object to read the PDF content
            doc = fitz.open(stream=pdf_content, filetype="pdf")
            output_pdf_io = BytesIO()
            doc.save(output_pdf_io)
            doc.close()
            output_pdf_io.seek(0)
            # Create response with page number in header
            response = send_file(
                output_pdf_io,
                mimetype='application/pdf',
                as_attachment=False,
                download_name=source["filename"]
            )
            response.headers['X-Page-Number'] = str(target_page+1)
        except Exception as e2:
            print(f"Error in second attempt: {str(e2)}")
            # Final fallback - just return the original PDF without highlighting
            blob_client = rag_pipeline.blob_service_client.get_blob_client(
                container=rag_pipeline.blob_container_name, blob=source["filename"]
            )
            # Download the PDF content from blob storage
            pdf_content = blob_client.download_blob().readall()

            target_page=int(source["page_number"])-1

            # Create a BytesIO object to read the PDF content
            doc = fitz.open(stream=pdf_content, filetype="pdf")
            output_pdf_io = BytesIO()
            doc.save(output_pdf_io)
            doc.close()
            output_pdf_io.seek(0)
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
            print("Results : ", result)
            # print(response["source_documents"])
            # print(result)
            relevant_sources = {}
            seen_files = set()

            for filename in result:
                cleaned_source_filename = filename.split("/")[-1].strip()
                allowed_pages = result[filename]
                for doc in response["source_documents"]:
                    cleaned_retrieved_filename = doc["filename"].split("/")[-1].strip()
                    page_number = doc["page_number"]
                    if (
                        (cleaned_retrieved_filename == cleaned_source_filename) and (page_number in allowed_pages)
                    ):
                        if cleaned_retrieved_filename in relevant_sources:
                            relevant_sources[cleaned_retrieved_filename]["content"].append(doc["content"])
                            relevant_sources[cleaned_retrieved_filename]["page_number"].append(doc["page_number"])
                            print(relevant_sources[cleaned_retrieved_filename]["page_number"])
                        else:
                            relevant_sources[cleaned_retrieved_filename] = doc
                            relevant_sources[cleaned_retrieved_filename]["content"] = [doc["content"]]
                            relevant_sources[cleaned_retrieved_filename]["page_number"] = [doc["page_number"]]
                        # relevant_sources.append(doc)
                        # seen_files.add(cleaned_source_filename)
            relevant_sources = [relevant_sources[file_name] for file_name in relevant_sources]
                        

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

@app.route("/test_highlights")
def test_highlights():
    """Test endpoint for PDF highlighting"""
    try:
        # Test if fitz is working
        import fitz
        fitz_status = "PyMuPDF (fitz) is available"
    except ImportError as e:
        fitz_status = f"PyMuPDF (fitz) import error: {str(e)}"
    
    try:
        # Test if RAG pipeline is available
        rag_status = "RAG pipeline is available" if rag_pipeline else "RAG pipeline is not available"
    except Exception as e:
        rag_status = f"RAG pipeline error: {str(e)}"
    
    return jsonify({
        "fitz_status": fitz_status,
        "rag_status": rag_status,
        "pipeline_initialized": rag_pipeline is not None
    })


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
