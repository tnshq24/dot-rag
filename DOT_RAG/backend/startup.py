import os
import sys
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

# Import the Flask app
from backend.app import app

if __name__ == "__main__":
    # Get port from environment variable (Azure Web App requirement)
    port = int(os.environ.get("PORT", 5001))
    
    # Run the Flask app
    app.run(host="0.0.0.0", port=port) 