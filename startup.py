import os
import sys
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Add DOT_RAG directory to path
dot_rag_dir = current_dir / "DOT_RAG"
sys.path.insert(0, str(dot_rag_dir))

# Import the Flask app from backend
from DOT_RAG.backend.app import app

if __name__ == "__main__":
    # Get port from environment variable (Azure Web App requirement)
    port = int(os.environ.get("PORT", 8000))
    
    # Run the Flask app
    app.run(host="0.0.0.0", port=port, debug=False) 