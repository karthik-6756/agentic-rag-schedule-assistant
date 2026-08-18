# Agentic RAG Schedule Assistant - Gemini version

This version removes the OpenAI API completely.

Environment variables on Render:
- GEMINI_API_KEY = your Google AI Studio key
- GEMINI_MODEL = gemini-2.5-flash

Build command:
pip install -r requirements.txt

Start command:
streamlit run app.py --server.address 0.0.0.0 --server.port $PORT
