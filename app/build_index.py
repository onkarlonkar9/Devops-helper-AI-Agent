# ===============================================
# build_index.py — Build ChromaDB index for Jarvis
# ===============================================

import os
import chromadb
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

# -----------------------------
# CONFIG
# -----------------------------
DB_DIR = "./chroma_db1"
COLLECTION_NAME = "devops_mini"          # For all-MiniLM-L6-v2 (384-dim)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"     # Compact + fast
DATA_DIR = "./data"                      # Folder containing your text or markdown docs

# -----------------------------
# INIT
# -----------------------------
print("[+] Connecting to Chroma...")
client = PersistentClient(path=DB_DIR)

# Load embedding model
print(f"[+] Loading embedding model: {EMBEDDING_MODEL}")
model = SentenceTransformer(EMBEDDING_MODEL)

# Create or load collection
try:
    collection = client.get_collection(COLLECTION_NAME)
    print(f"[+] Existing collection '{COLLECTION_NAME}' loaded.")
except Exception:
    print(f"[!] Collection '{COLLECTION_NAME}' not found — creating new one...")
    collection = client.create_collection(COLLECTION_NAME)
    print(f"[+] Created new collection '{COLLECTION_NAME}'.")

# -----------------------------
# BUILD INDEX
# -----------------------------
print(f"[+] Scanning '{DATA_DIR}' for documents...")
docs = []
doc_ids = []

for root, _, files in os.walk(DATA_DIR):
    for f in files:
        if f.endswith((".txt", ".md")):
            path = os.path.join(root, f)
            with open(path, "r", encoding="utf-8") as file:
                text = file.read().strip()
                if text:
                    docs.append(text)
                    doc_ids.append(path)
                    print(f"    └── Added: {path}")

if not docs:
    print("[!] No text files found in data folder.")
    exit(0)

print(f"[+] Found {len(docs)} files. Building embeddings...")

embeddings = model.encode(docs, show_progress_bar=True)

# Add docs to Chroma
collection.add(
    documents=docs,
    embeddings=embeddings.tolist(),
    ids=doc_ids
)

print(f"[✓] Successfully built index for '{COLLECTION_NAME}' using {EMBEDDING_MODEL}")
print("[✓] Database directory:", DB_DIR)
print("[✓] Total documents added:", len(docs))
print("--------------------------------------------------")
print("Jarvis Static Knowledge Base is ready 🚀")

