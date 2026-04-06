"""
server/app.py — Re-exports the FastAPI app from the root app module.
This file exists for OpenEnv validator compatibility.
"""
import sys
import os

# Add parent directory to path so we can import from root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

def main():
    import uvicorn
    # Use the same port that Hugging Face Spaces expect
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
