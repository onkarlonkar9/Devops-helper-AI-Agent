# app/agent.py
import os
import json
import time
import requests
from chromadb import PersistentClient
from datetime import datetime
from sentence_transformers import SentenceTransformer
import hashlib

# -----------------------------
# CONFIGURATION
# -----------------------------
DB_DIR = "./chroma_db1"
STATIC_COLLECTION = "devops_mini"           # Static docs
MEMORY_COLLECTION = "memory_devops_mini"    # Dynamic long-term memory
OLLAMA_API = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"
MAX_CONTEXT_DOCS = 5
MEMORY_SIZE = 3  # short-term buffer

# -----------------------------
# INITIALIZE
# -----------------------------
print("[+] Connecting to Chroma...")
client = PersistentClient(path=DB_DIR)

# Static Knowledge Base
try:
    static_collection = client.get_collection(STATIC_COLLECTION)
    print(f"[+] Static collection '{STATIC_COLLECTION}' loaded.")
except Exception:
    raise RuntimeError(f"[!] Could not load static collection '{STATIC_COLLECTION}'")

# Persistent Memory
try:
    memory_collection = client.get_collection(MEMORY_COLLECTION)
    print(f"[+] Memory collection '{MEMORY_COLLECTION}' loaded.")
except Exception:
    memory_collection = client.create_collection(MEMORY_COLLECTION)
    print(f"[+] Memory collection '{MEMORY_COLLECTION}' created.")

embedder = SentenceTransformer("all-mpnet-base-v2")

print("[+] DevOps AI Agent ready. Type 'exit' to quit.\n")

conversation_history = []

# -----------------------------
# HELPERS
# -----------------------------
def hash_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

# -----------------------------
# MEMORY HANDLING
# -----------------------------
def add_memory(user_id: str, text: str, role="user"):
    """Store conversation chunks into memory DB."""
    embedding = embedder.encode([text])[0].tolist()
    mem_id = f"{user_id}_{role}_{hash_text(text)[:8]}"
    metadata = {
        "user_id": user_id,
        "role": role,
        "timestamp": datetime.now().isoformat()
    }
    memory_collection.upsert(
        documents=[text],
        embeddings=[embedding],
        ids=[mem_id],
        metadatas=[metadata]
    )

def recall_memory(user_id: str, query: str, top_k=3):
    """Retrieve semantically related memories."""
    try:
        results = memory_collection.query(
            query_texts=[query],
            n_results=top_k,
            where={"user_id": user_id}
        )
        docs = results.get("documents", [[]])[0]
        return "\n".join(docs)
    except Exception as e:
        print(f"[!] Memory recall failed: {e}")
        return ""

# -----------------------------
# SMART SEARCH
# -----------------------------
def search_docs(query_text):
    """Semantic search on static KB."""
    try:
        results = static_collection.query(query_texts=[query_text], n_results=MAX_CONTEXT_DOCS)
        docs = results["documents"][0]
        metadatas = results.get("metadatas", [[]])[0]
        context_blocks = []
        for i, doc in enumerate(docs):
            source = metadatas[i].get("source", "unknown") if metadatas else "unknown"
            context_blocks.append(f"[Source: {source}]\n{doc}")
        return "\n\n".join(context_blocks)
    except Exception as e:
        return f"[ERROR] search_docs failed: {e}"

# -----------------------------
# QUERY REPHRASING
# -----------------------------
def rephrase_query(prompt):
    """Optional step for clarity."""
    if len(prompt.split()) < 3:
        return prompt
    refined_prompt = f"Rephrase this for clarity: {prompt}"
    try:
        res = requests.post(
            OLLAMA_API,
            json={"model": MODEL_NAME, "prompt": refined_prompt, "stream": False},
        )
        if res.ok:
            return res.json()["response"].strip()
    except:
        pass
    return prompt

# -----------------------------
# QUERY OLLAMA
# -----------------------------
def query_ollama(prompt, context=""):
    payload = {
        "model": MODEL_NAME,
        "prompt": f"""You are a DevOps expert assistant.
Use the provided context (memory + docs) to give accurate, step-by-step answers.
If context is missing, answer with your own knowledge.

Context:
{context}

User question:
{prompt}

Answer clearly:""",
        "stream": True,
        "options": {"temperature": 0.6, "num_predict": 512, "top_k": 30},
    }

    try:
        with requests.post(OLLAMA_API, json=payload, stream=True) as response:
            if response.status_code != 200:
                return f"[ERROR] Ollama returned {response.status_code}"

            answer = ""
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    token = data.get("response", "")
                    print(token, end="", flush=True)
                    answer += token
                except json.JSONDecodeError:
                    continue
            print()
            return answer.strip()
    except Exception as e:
        return f"[ERROR] Ollama request failed: {str(e)}"

# -----------------------------
# CONTEXT BUILDER
# -----------------------------
def build_context(user_id, user_query):
    """Merge short-term history, semantic memory, and doc search."""
    # Short-term memory
    short_term = "\n".join(
        [f"User: {q}\nAgent: {a}" for q, a in conversation_history[-MEMORY_SIZE:]]
    )
    # Long-term memory (semantic)
    long_term = recall_memory(user_id, user_query, top_k=3)
    # Static docs
    docs = search_docs(user_query)
    return f"Short-term:\n{short_term}\n\nLong-term memory:\n{long_term}\n\nDocs:\n{docs}"

# -----------------------------
# MAIN LOOP
# -----------------------------
USER_ID = "onkar"

while True:
    query = input("🧩 You: ").strip()
    if query.lower() in ["exit", "quit"]:
        print("👋 Exiting agent.")
        break
    if not query:
        continue

    print("[+] Thinking...\n")

    refined_query = rephrase_query(query)
    context = build_context(USER_ID, refined_query)

    print("🤖 Agent:", end=" ", flush=True)
    answer = query_ollama(refined_query, context)

    # Track conversation
    conversation_history.append((query, answer))

    # Save to persistent memory
    add_memory(USER_ID, query, role="user")
    add_memory(USER_ID, answer, role="agent")

    print("\n" + "-" * 80)

