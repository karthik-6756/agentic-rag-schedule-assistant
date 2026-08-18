import os
import json
import re
from pathlib import Path
from datetime import date

import numpy as np
import streamlit as st
import chromadb
from sklearn.feature_extraction.text import HashingVectorizer
from google import genai
from google.genai import types

BASE = Path(__file__).parent
SCHEDULE_FILE = BASE / "schedule.json"
CHROMA_DIR = BASE / "chroma_db"

st.set_page_config(page_title="Agentic RAG Schedule Assistant", page_icon="📅", layout="wide")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY is not configured in Render Environment Variables.")
    st.stop()

client = genai.Client(api_key=GEMINI_API_KEY)
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Local vectorization: no OpenAI embedding API is needed.
vectorizer = HashingVectorizer(
    n_features=512,
    alternate_sign=False,
    norm="l2",
    lowercase=True,
    token_pattern=r"(?u)\b\w+\b",
)

chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = chroma.get_or_create_collection(name="schedule")

def load_events():
    return json.loads(SCHEDULE_FILE.read_text(encoding="utf-8"))

def save_events(events):
    SCHEDULE_FILE.write_text(json.dumps(events, indent=2), encoding="utf-8")

def event_text(e):
    return (
        f"{e['title']} | {e['date']} | {e['start_time']}-{e['end_time']} | "
        f"{e['type']} | {e.get('description', '')}"
    )

def vectorize(texts):
    return vectorizer.transform(texts).toarray().astype(np.float32).tolist()

def rebuild_index():
    global collection
    try:
        chroma.delete_collection("schedule")
    except Exception:
        pass
    collection = chroma.get_or_create_collection(name="schedule")
    es = load_events()
    if not es:
        return
    docs = [event_text(e) for e in es]
    collection.upsert(
        ids=[e["id"] for e in es],
        documents=docs,
        embeddings=vectorize(docs),
        metadatas=[
            {"event_id": e["id"], "date": e["date"], "title": e["title"], "type": e["type"]}
            for e in es
        ],
    )

def ensure_index():
    if collection.count() != len(load_events()):
        rebuild_index()

ensure_index()

def get_schedule(
    date_from: str = "",
    date_to: str = "",
    time_from: str = "",
    time_to: str = "",
    query: str = "",
    top_k: int = 8,
):
    """Retrieve schedule events using date/time filters and ChromaDB vector search.
    Dates are YYYY-MM-DD and times are HH:MM. Use this for schedule questions,
    availability, conflicts, or to identify an existing event before changing it.
    """
    es = load_events()
    filtered = []
    for e in es:
        if date_from and e["date"] < date_from:
            continue
        if date_to and e["date"] > date_to:
            continue
        if time_from and e["end_time"] <= time_from:
            continue
        if time_to and e["start_time"] >= time_to:
            continue
        filtered.append(e)

    if query:
        search_space = filtered if filtered else es
        if not search_space:
            return {"events": [], "count": 0}
        query_vec = vectorize([query])[0]
        result = collection.query(
            query_embeddings=[query_vec],
            n_results=min(max(int(top_k or 8), 1), len(search_space)),
        )
        ids = result.get("ids", [[]])[0]
        by_id = {e["id"]: e for e in search_space}
        ranked = [by_id[i] for i in ids if i in by_id]
        return {"events": ranked, "count": len(ranked)}

    return {"events": filtered[:max(int(top_k or 8), 1)], "count": len(filtered)}

def update_schedule(
    action: str,
    event_id: str = "",
    title: str = "",
    event_date: str = "",
    start_time: str = "",
    end_time: str = "",
    event_type: str = "meeting",
    description: str = "",
):
    """Add, update, or remove a schedule entry.
    For update/remove, event_id must identify an existing event.
    """
    es = load_events()

    if action == "add":
        if not all([title, event_date, start_time, end_time]):
            return {"success": False, "error": "title, event_date, start_time and end_time are required"}
        nums = [int(re.sub(r"\D", "", e["id"]) or 0) for e in es]
        new_event = {
            "id": event_id or f"evt-{max(nums + [0]) + 1:03d}",
            "title": title,
            "date": event_date,
            "start_time": start_time,
            "end_time": end_time,
            "type": event_type or "meeting",
            "description": description or "",
        }
        es.append(new_event)
        save_events(es)
        rebuild_index()
        return {"success": True, "event": new_event}

    target = next((e for e in es if e["id"] == event_id), None)
    if not target:
        return {"success": False, "error": f"Event {event_id} not found"}

    if action == "remove":
        es.remove(target)
        save_events(es)
        rebuild_index()
        return {"success": True, "removed": target}

    if action == "update":
        for key, value in {
            "title": title, "date": event_date, "start_time": start_time,
            "end_time": end_time, "type": event_type, "description": description
        }.items():
            if value:
                target[key] = value
        save_events(es)
        rebuild_index()
        return {"success": True, "event": target}

    return {"success": False, "error": "action must be add, update, or remove"}

SYSTEM_PROMPT = f"""
You are an Agentic RAG Schedule Assistant.
Today's date is {date.today().isoformat()}.
Use exactly two tools: get_schedule and update_schedule.
For schedule questions, availability, conflicts, or existing-event references, use get_schedule.
For an existing event update/removal without an event_id, first use get_schedule to identify it, then use update_schedule.
For a new event, use update_schedule when the date and time are clear.
Resolve relative dates such as tomorrow, Friday, and next Monday.
For availability, inspect the returned events and determine whether the requested time overlaps them.
Never claim a mutation succeeded unless update_schedule returned success=true.
Keep answers concise and clear.
"""

def run_agent(user_text):
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[get_schedule, update_schedule],
        temperature=0.1,
    )
    response = client.models.generate_content(
        model=MODEL,
        contents=user_text,
        config=config,
    )
    return response.text

st.title("📅 Agentic RAG Schedule Assistant")
st.caption("Gemini Agent • ChromaDB RAG • Local vectors • get_schedule + update_schedule")

with st.sidebar:
    st.subheader("Project")
    st.metric("Stored events", len(load_events()))
    if st.button("Rebuild Chroma index"):
        rebuild_index()
        st.success("ChromaDB index rebuilt.")
    st.divider()
    st.write("Try:")
    st.code("What do I have scheduled tomorrow?")
    st.code("Am I free Friday afternoon?")
    st.code("Add a meeting on August 25 at 3 PM.")
    st.code("Move my meeting from 3 PM to 4 PM.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Ask about your schedule or change it...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Agent is retrieving and deciding..."):
            try:
                answer = run_agent(prompt)
            except Exception as exc:
                answer = (
                    "I couldn't process that request. Check GEMINI_API_KEY and Render logs.\n\n"
                    f"Error: `{type(exc).__name__}: {exc}`"
                )
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
