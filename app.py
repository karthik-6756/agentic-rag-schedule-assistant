import os,json,re
from pathlib import Path
from datetime import date
import streamlit as st
import chromadb
from openai import OpenAI

BASE=Path(__file__).parent
DATA=BASE/"schedule.json"
DB=BASE/"chroma_db"
if not os.getenv("OPENAI_API_KEY"):
    st.error("Set OPENAI_API_KEY in the deployment environment."); st.stop()
ai=OpenAI()
db=chromadb.PersistentClient(path=str(DB))
col=db.get_or_create_collection("schedule")

def events(): return json.loads(DATA.read_text())
def text(e): return f"{e['title']} | {e['date']} | {e['start_time']}-{e['end_time']} | {e['type']} | {e.get('description','')}"
def emb(xs): return [x.embedding for x in ai.embeddings.create(model="text-embedding-3-small",input=xs).data]
def rebuild():
    es=events()
    if es: col.upsert(ids=[e["id"] for e in es],documents=[text(e) for e in es],embeddings=emb([text(e) for e in es]),metadatas=[{"date":e["date"],"title":e["title"],"type":e["type"]} for e in es])
if col.count()!=len(events()): rebuild()

def get_schedule(date_from=None,date_to=None,time_from=None,time_to=None,query=None,top_k=8):
    es=events(); out=[]
    for e in es:
        if date_from and e["date"]<date_from: continue
        if date_to and e["date"]>date_to: continue
        if time_from and e["end_time"]<=time_from: continue
        if time_to and e["start_time"]>=time_to: continue
        out.append(e)
    if query:
        r=col.query(query_embeddings=emb([query])[0:1],n_results=min(top_k,max(1,col.count())))
        ids=set(r["ids"][0]) if r.get("ids") else set()
        return [e for e in (out or es) if e["id"] in ids][:top_k]
    return out[:top_k]

def update_schedule(action,event_id=None,title=None,event_date=None,start_time=None,end_time=None,event_type=None,description=None):
    es=events()
    if action=="add":
        if not all([title,event_date,start_time,end_time]): return {"success":False,"error":"title, event_date, start_time and end_time are required"}
        nums=[int(re.sub(r"\D","",e["id"]) or 0) for e in es]
        e={"id":event_id or f"evt-{max(nums+[0])+1:03d}","title":title,"date":event_date,"start_time":start_time,"end_time":end_time,"type":event_type or "meeting","description":description or ""}
        es.append(e)
        DATA.write_text(json.dumps(es,indent=2)); rebuild(); return {"success":True,"event":e}
    target=next((e for e in es if e["id"]==event_id),None)
    if not target: return {"success":False,"error":"Event not found"}
    if action=="remove": es.remove(target)
    elif action=="update":
        for k,v in {"title":title,"date":event_date,"start_time":start_time,"end_time":end_time,"type":event_type,"description":description}.items():
            if v is not None: target[k]=v
    else: return {"success":False,"error":"action must be add, update or remove"}
    DATA.write_text(json.dumps(es,indent=2)); rebuild()
    return {"success":True,"event":target} if action=="update" else {"success":True,"removed":target}

TOOLS=[{"type":"function","function":{"name":"get_schedule","description":"Retrieve schedule using date/time filters and semantic RAG. Use for schedule questions, availability, and before changing an indirectly referenced event.","parameters":{"type":"object","properties":{"date_from":{"type":"string"},"date_to":{"type":"string"},"time_from":{"type":"string"},"time_to":{"type":"string"},"query":{"type":"string"},"top_k":{"type":"integer"}}}}},
{"type":"function","function":{"name":"update_schedule","description":"Add, update, or remove a schedule entry. For existing events, use get_schedule first unless event_id is already known.","parameters":{"type":"object","properties":{"action":{"type":"string","enum":["add","update","remove"]},"event_id":{"type":"string"},"title":{"type":"string"},"event_date":{"type":"string"},"start_time":{"type":"string"},"end_time":{"type":"string"},"event_type":{"type":"string","enum":["meeting","workshop","task","appointment"]},"description":{"type":"string"}},"required":["action"]}}}]

def agent(q):
    sys=f"""You are an Agentic RAG Schedule Assistant. Today is {date.today().isoformat()}.
Use only the schedule in the tools. Resolve relative dates such as tomorrow and Friday.
For schedule questions, availability, conflicts, or indirect references, call get_schedule.
For an existing event change/removal, retrieve first unless exact event_id is known.
For adding, call update_schedule. Never claim a mutation succeeded unless success=true.
For availability, inspect overlapping events and clearly say free/busy."""
    msgs=[{"role":"system","content":sys},{"role":"user","content":q}]
    trace=[]
    for _ in range(6):
        r=ai.chat.completions.create(model=os.getenv("OPENAI_MODEL","gpt-4.1-mini"),messages=msgs,tools=TOOLS,tool_choice="auto",temperature=0)
        m=r.choices[0].message
        if not m.tool_calls: return m.content,trace
        msgs.append(m)
        for c in m.tool_calls:
            a=json.loads(c.function.arguments or "{}"); trace.append({"tool":c.function.name,"arguments":a})
            result=get_schedule(**a) if c.function.name=="get_schedule" else update_schedule(**a)
            msgs.append({"role":"tool","tool_call_id":c.id,"content":json.dumps(result)})
    return "Unable to complete the request.",trace

st.set_page_config(page_title="Agentic RAG Schedule Assistant",page_icon="📅",layout="wide")
st.title("📅 Agentic RAG Schedule Assistant")
st.caption("30-day schedule • ChromaDB • RAG • two agent tools")
st.sidebar.metric("Events",len(events()))
if st.sidebar.button("Rebuild Chroma index"): rebuild(); st.sidebar.success("Rebuilt")
st.sidebar.write("Examples")
for x in ["What do I have scheduled tomorrow?","Am I free Friday afternoon?","Add a meeting on August 15 at 3 PM.","Move my meeting from 2 PM to 4 PM."]: st.sidebar.code(x)
if "chat" not in st.session_state: st.session_state.chat=[]
for m in st.session_state.chat:
    with st.chat_message(m["role"]): st.markdown(m["content"])
q=st.chat_input("Ask about your schedule or change it...")
if q:
    st.session_state.chat.append({"role":"user","content":q})
    with st.chat_message("user"): st.markdown(q)
    with st.chat_message("assistant"):
        with st.spinner("Retrieving and deciding..."): ans,trace=agent(q)
        st.markdown(ans)
        with st.expander("Agent tool trace"):
            for t in trace: st.json(t)
    st.session_state.chat.append({"role":"assistant","content":ans})
