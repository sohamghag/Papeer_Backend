from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage,HumanMessage
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os 
from fastapi import FastAPI,HTTPException
from contextlib import asynccontextmanager
import uuid
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from rag_graph import graph 
from supabase import create_client
import tempfile
from pathlib import Path
from fastapi import UploadFile, File, Form
from loader import load_document
from vector_store import add_paper
from vector_store import list_papers
from vector_store import delete_collection

class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str

class HistoryMessage(BaseModel):
    role: str   # "user" | "assistant"
    content: str

class RenameRequest(BaseModel):
    title: str

load_dotenv()
api_keys_kimi=os.getenv("KIMI_API_KEY")
kimi_llm = ChatOpenAI(
    api_key=api_keys_kimi,
    base_url="https://api.moonshot.ai/v1",
    model="moonshot-v1-32k"
)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
POSTGRES_URL = os.getenv("POSTGRES_URL")
app_state: dict = {}



@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Build the checkpointer + compile the graph once at startup.
    AsyncPostgresSaver needs an open connection pool for the app's lifetime.
    """
    print("[lifespan] Connecting to Postgres checkpointer...")
 
    async with AsyncPostgresSaver.from_conn_string(POSTGRES_URL) as checkpointer:
        # Creates the checkpoint tables in Supabase if they don't exist yet.
        # Safe to call every startup — it's a no-op if tables already exist.
        await checkpointer.setup()
 
        compiled_graph = graph.compile(checkpointer=checkpointer)
        app_state["graph"] = compiled_graph
 
        print("[lifespan] Graph compiled with Postgres checkpointer. Ready.")
        yield
    print("[lifespan] Shutting down.")

app = FastAPI(title="Papeer API",lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

async def generate_session_title(message: str) -> str:
    response = kimi_llm.invoke([
        {"role": "system", "content": "Generate a short 3-5 word title summarizing this message. No quotes, no punctuation at the end. Just the title text."},
        {"role": "user", "content": message}
    ])
    return response.content.strip()

async def create_new_session(title: str = "New Chat") -> str:
    """Creates a new session row and returns the session_id."""
    session_id = str(uuid.uuid4())
    supabase_client.table("sessions").insert({
        "session_id": session_id,
        "title": title,
    }).execute()
    return {"session_id": session_id, "title": title}

@app.get("/api/health")
async def health():
    return {"status": "ok"}

@app.post("/api/session")
async def create_session():
    try:
        result = await create_new_session()
        session_id = result["session_id"]
        title = result["title"]
        return {"session_id": session_id, "title": title}
    except Exception as e:
        print(f"[/session] ERROR: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/api/conversations")
async def load_conversations():
    try:
        response = supabase_client.table("sessions").select("*").order("updated_at", desc=True).execute()
        return {"conversations": response.data}
    except Exception as e:
        print(f"[/conversations] ERROR: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat(req: ChatRequest):
    graph = app_state.get("graph")
    if not graph:
        raise HTTPException(status_code=503, detail="Graph not ready")
    session_id = req.session_id
    try:
        if not session_id:
            session_data = await create_new_session()
            session_id = session_data["session_id"]

        config = {"configurable": {"thread_id": session_id}}
        initial_state = {
            "session_id": session_id,
            "messages": [HumanMessage(content=req.message)],
        }
        result = await graph.ainvoke(initial_state, config=config)

        # ALWAYS check — regardless of whether session was just created or already existed
        existing = supabase_client.table("sessions").select("title").eq("session_id", session_id).execute()
        current_title = existing.data[0]["title"] if existing.data else "New Chat"

        if current_title == "New Chat":
            new_title = await generate_session_title(req.message)
            supabase_client.table("sessions").update({"title": new_title}).eq("session_id", session_id).execute()

    except Exception as e:
        print(f"[/chat] ERROR: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "session_id": session_id,
        "route": result.get("route"),
        "answer": result.get("answer"),
    }

@app.get("/api/history/{session_id}")
async def get_history(session_id: str):
    graph = app_state.get("graph")
    if not graph:
        raise HTTPException(status_code=503, detail="Graph not ready")

    config = {"configurable": {"thread_id": session_id}}
    state = await graph.aget_state(config)
    if not state or not state.values:
        return {"session_id": session_id, "messages": []}

    all_messages = state.values.get("messages", [])

    clean_history: list[HistoryMessage] = []
    pending_assistant_content = None

    print("messages",all_messages)

    for msg in all_messages:
        if isinstance(msg, HumanMessage):
            # flush any pending assistant reply from the previous turn first
            if pending_assistant_content is not None:
                clean_history.append(HistoryMessage(role="assistant", content=pending_assistant_content))
                pending_assistant_content = None
            clean_history.append(HistoryMessage(role="user", content=msg.content))

        elif isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            # keep overwriting — only the LAST one before the next Human message survives
            pending_assistant_content = msg.content

    # flush the final trailing assistant message after the loop ends
    if pending_assistant_content is not None:
        clean_history.append(HistoryMessage(role="assistant", content=pending_assistant_content))
    
    print("Clean_History",clean_history)
    return {"session_id": session_id, "messages": clean_history}

@app.patch("/api/conversations/{session_id}")
async def rename_conversation(session_id: str, req: RenameRequest):
    try:
        supabase_client.table("sessions").update({"title": req.title}).eq("session_id", session_id).execute()
        return {"session_id": session_id, "title": req.title}
    except Exception as e:
        print(f"[/conversations PATCH] ERROR: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.delete("/api/conversations/{session_id}")
async def delete_conversation(session_id: str):
    try:
        graph = app_state.get("graph")
        checkpointer = graph.checkpointer  # access the checkpointer LangGraph is using
        await checkpointer.adelete_thread(session_id)
        delete_collection(session_id)
        supabase_client.table("sessions").delete().eq("session_id", session_id).execute()
        return {"deleted": session_id}
    except Exception as e:
        print(f"[/conversations DELETE] ERROR: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}  

@app.post("/api/upload")
async def upload_documents(session_id: str = Form(...),
    file: UploadFile = File(...),):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")
    tmp_path = None
    try:
        if not session_id:
            session_data = await create_new_session()
            session_id = session_data["session_id"]

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
        tmp.close()
        docs = load_document(tmp_path)  

        # load_document stamps title from the temp filename — overwrite with the real name
        original_title = Path(file.filename).stem
        for doc in docs:
            doc.metadata["title"] = original_title

        add_paper(docs, session_id)

        return {
            "filename": file.filename,
            "chunks_added": len(docs),
            "session_id": session_id,
        }
    except Exception as e:
        print(f"[/api/upload] ERROR: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)

@app.get("/api/documents/{session_id}")
async def get_documents(session_id: str):
    try:
        titles = list_papers(session_id)
        return {"session_id": session_id, "documents": titles}
    except Exception as e:
        print(f"[/api/documents] ERROR: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class LoadUrlRequest(BaseModel):
    session_id: str | None = None
    url: str

@app.post("/api/load-url")
async def load_url(req: LoadUrlRequest):
    try:
        session_id = req.session_id
        if not session_id:
            session_data = await create_new_session()
            session_id = session_data["session_id"]

        docs = load_document(req.url)  
        add_paper(docs, session_id)

        title = docs[0].metadata.get("title", req.url) if docs else req.url

        return {
            "url": req.url,
            "title": title,
            "chunks_added": len(docs),
            "session_id": session_id,
        }
    except Exception as e:
        print(f"[/api/load-url] ERROR: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=str(e))