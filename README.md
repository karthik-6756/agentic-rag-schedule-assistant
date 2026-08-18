# Agentic RAG Schedule Assistant

Streamlit + OpenAI tool calling + ChromaDB.

Run:
`pip install -r requirements.txt`
`streamlit run app.py`

Set `OPENAI_API_KEY`.

Deploy on Render as a Python Web Service:
Build: `pip install -r requirements.txt`
Start: `streamlit run app.py --server.address 0.0.0.0 --server.port $PORT`

The app contains the two required tools: `get_schedule` and `update_schedule`.
