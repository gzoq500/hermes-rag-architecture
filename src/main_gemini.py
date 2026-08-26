from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse, Response
import uvicorn
import urllib.request
import json
import time
import httpx
import sqlite3
from pymilvus import MilvusClient
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MILVUS_URI = "https://in01-8fb19767351ff95.aws-ap-southeast-1.vectordb.zillizcloud.com:19532"
MILVUS_TOKEN = "<REPLACE_WITH_ZILLIZ_TOKEN>"
COLLECTION_NAME = "hermes_gemini_memory"
DB_PATH = "/root/.hermes/state.db"

GEMINI_API_KEY = "<REPLACE_WITH_GOOGLE_GEMINI_KEY>"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:embedContent?key={GEMINI_API_KEY}"

milvus_client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)
ROUTER_URL = "http://127.0.0.1:20130/v1/chat/completions"

def chunk_text_intelligently(text: str, max_chars=8000) -> list[str]:
    if len(text) <= max_chars: return [text]
    chunks = []
    lines = text.split("\n")
    current_chunk = ""
    for line in lines:
        if len(current_chunk) + len(line) > max_chars:
            chunks.append(current_chunk)
            current_chunk = line + "\n"
        else:
            current_chunk += line + "\n"
    if current_chunk: chunks.append(current_chunk)
    return chunks

def get_embedding(text: str) -> list[float]:
    body = json.dumps({"model": "models/gemini-embedding-2", "content": {"parts": [{"text": text}]}}).encode("utf-8")
    req = urllib.request.Request(GEMINI_URL, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())["embedding"]["values"]
    except: return None

def search_memory_sync(query: str, session_id: str = None, limit: int = 3):
    try:
        vec = get_embedding(query)
        if not vec: return []
        filter_expr = f"session_id == \"{session_id}\"" if session_id else ""
        results = milvus_client.search(
            collection_name=COLLECTION_NAME, data=[vec], limit=limit,
            filter=filter_expr, output_fields=["text"]
        )
        if results and len(results) > 0:
            return [hit["entity"]["text"] for hit in results[0]]
    except: pass
    return []

def get_real_session_id(user_query: str) -> str:
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        query_snippet = user_query[:100] + "%"
        cur.execute("SELECT session_id FROM messages WHERE role=user AND content LIKE ? ORDER BY id DESC LIMIT 1", (query_snippet,))
        row = cur.fetchone()
        conn.close()
        if row: return row[0]
    except: pass
    return "default_session"

def background_upsert(session_id: str, role: str, content: str):
    if not content or len(content) < 15: return
    chunks = chunk_text_intelligently(content, max_chars=8000)
    ts = int(time.time())
    for i, chunk_data in enumerate(chunks):
        chunk_text = f"{role.upper()}: {chunk_data}"
        if len(chunks) > 1: chunk_text = f"[{role.upper()} PART {i+1}/{len(chunks)}]:\n{chunk_data}"
        try:
            vec = get_embedding(chunk_text)
            if vec:
                doc_id = f"{session_id}_{ts}_{i}"
                milvus_client.insert(collection_name=COLLECTION_NAME, data=[{
                    "id": doc_id, "vector": vec, "text": chunk_text, "session_id": session_id, "timestamp": ts
                }])
        except: pass

@app.api_route("/{path_name:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH", "TRACE"])
async def catch_all(request: Request, path_name: str, background_tasks: BackgroundTasks):
    
    # -------------------------------------------------------------
    # JALUR KHUSUS INFERENCE LLM (CHAT DARI TELEGRAM HERMES)
    # -------------------------------------------------------------
    if path_name == "v1/chat/completions" and request.method == "POST":
        raw_body = await request.body()
        try: payload = json.loads(raw_body.decode("utf-8"))
        except: return JSONResponse({"error": "Invalid JSON"}, status_code=400)
            
        messages = payload.get("messages", [])
        
        # Patch mutlak: CB/CBCN Wajib system prompt pertama
        if len(messages) > 0 and messages[0].get("role") != "system":
            payload["messages"].insert(0, {"role": "system", "content": "You are a helpful AI."})
            messages = payload["messages"]
                
        user_id = payload.get("user", "")
        if messages and messages[-1].get("role") == "user":
            user_query = str(messages[-1].get("content", ""))
            
            if not user_id or user_id == "default_session":
                user_id = get_real_session_id(user_query)
                
            background_tasks.add_task(background_upsert, user_id, "user", user_query)
            
            if len(user_query) > 10:
                rag_texts = search_memory_sync(user_query, session_id=user_id, limit=3)
                if rag_texts:
                    recalled_context = "\\n\\n[ARCHIVED MEMORY RECALLED FROM ZILLIZ CLOUD]\\n"
                    for t in rag_texts: recalled_context += f"- {t}\\n"
                    recalled_context += "[END OF ARCHIVED MEMORY]\\n"
                    
                    if messages[0].get("role") == "system":
                        messages[0]["content"] = str(messages[0]["content"]) + recalled_context

        headers = dict(request.headers)
        headers.pop("host", None)
        headers.pop("content-length", None)
        
        if payload.get("stream"):
            async def stream_forwarder():
                ai_response_buffer = ""
                async with httpx.AsyncClient() as client:
                    async with client.stream("POST", ROUTER_URL, json=payload, headers=headers, timeout=120) as r:
                        async for chunk in r.aiter_bytes():
                            yield chunk
                            try:
                                for line in chunk.decode("utf-8").split("\n"):
                                    if line.startswith("data: ") and line != "data: [DONE]":
                                        data_json = json.loads(line[6:])
                                        if "choices" in data_json and len(data_json["choices"]) > 0:
                                            delta = data_json["choices"][0].get("delta", {})
                                            if "content" in delta:
                                                ai_response_buffer += delta["content"]
                            except: pass
                if ai_response_buffer:
                    background_tasks.add_task(background_upsert, user_id, "assistant", ai_response_buffer)
            return StreamingResponse(stream_forwarder(), media_type="text/event-stream")
        
        async with httpx.AsyncClient() as client:
            try:
                r = await client.post(ROUTER_URL, json=payload, headers=headers, timeout=120)
                resp_text = r.text
                try:
                    resp_json = json.loads(resp_text)
                    if "choices" in resp_json and len(resp_json["choices"]) > 0:
                        ai_text = resp_json["choices"][0].get("message", {}).get("content", "")
                        if ai_text: background_tasks.add_task(background_upsert, user_id, "assistant", ai_text)
                except: pass
                return Response(content=resp_text, status_code=r.status_code, media_type=r.headers.get("content-type", "application/json"))
            except Exception as e:
                return JSONResponse({"error": f"9Router Proxy Error: {str(e)}"}, status_code=502)

    # -------------------------------------------------------------
    # JALUR SEMUA ENDPOINT LAIN (Termasuk Tombol Tes Model di UI)
    # -------------------------------------------------------------
    else:
        target_url = f"http://127.0.0.1:20130/{path_name}"
        if request.url.query: target_url += f"?{request.url.query}"
            
        async with httpx.AsyncClient() as client:
            try:
                raw_body = await request.body()
                
                # SANGAT KRUSIAL: 9Router ternyata me-re-route "Tes Model" ke rute internalnya sendiri 
                # (contoh /api/providers/test), lalu 9Router secara internal memanggil fungsi axios
                # untuk menembak ke provider asli (cbai).
                # Jadi, Proxy kita TIDAK BISA mencegat "Tes Model" dari luar.
                # Proxy hanya bisa meneruskannya, dan kalau gagal, itu berarti 9Router asli-nya lah
                # yang gagal mengirim System Prompt ke cbai.
                
                headers = dict(request.headers)
                headers.pop("host", None)
                headers.pop("accept-encoding", None) 
                
                r = await client.request(
                    request.method, target_url, 
                    content=raw_body if len(raw_body) > 0 else None, 
                    headers=headers, timeout=30, follow_redirects=False 
                )
                
                resp_headers = dict(r.headers)
                resp_headers.pop("content-length", None)
                resp_headers.pop("transfer-encoding", None) 
                resp_headers.pop("content-encoding", None) 
                
                return Response(content=r.content, status_code=r.status_code, headers=resp_headers)
            except Exception as e:
                return JSONResponse({"error": f"UI Proxy Error: {str(e)}"}, status_code=502)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=20128)
