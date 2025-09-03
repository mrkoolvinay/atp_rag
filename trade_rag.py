import os
import hashlib
import glob
import numpy as np
import faiss
from tqdm import tqdm
import joblib
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import pandas as pd

load_dotenv(override=True)

# -------------------------------
# Config
# -------------------------------
CSV_FOLDER = "stock_csvs"
EMBEDDING_CACHE_FILE = "stock_embeddings.pkl"
FAISS_INDEX_FILE = "faiss_stock_hnsw.index"
PROCESSED_LOG = "processed_files.txt"
TOP_K = 200
RECENT_DAYS = 2            # kept for compatibility, not used in pandas path
BATCH_SIZE = 200           # controls both text conversion batching and embedding batching
MAX_CHUNK_LENGTH = 3000    # token-safe split for GPT context

# -------------------------------
# Load local embedding model
# -------------------------------
print("Loading local embedding model (SentenceTransformers)...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")  # fast + small
EMBED_DIM = embed_model.get_sentence_embedding_dimension()
print(f"Embedding model loaded (dim={EMBED_DIM})\n")

# -------------------------------
# Helper Functions
# -------------------------------
def get_embedding(texts):
    """
    Generate embeddings locally using SentenceTransformers in one call.
    Returns float32 numpy array of shape (N, EMBED_DIM).
    """
    embs = embed_model.encode(
        texts,
        batch_size=64,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=False,   # keep raw vectors; FAISS HNSWFlat works fine
    )
    return embs.astype(np.float32)

def load_embeddings_cache(path):
    """
    Backward-compatible loader:
    - New format: {"hashes": <set or list>, "texts": <list>}
    - Old format: <list> (assumed to be texts only)
    """
    if not os.path.exists(path):
        return set(), [], False
    try:
        data = joblib.load(path)
        if isinstance(data, dict):
            hashes = data.get("hashes", set())
            # ensure set type
            if isinstance(hashes, list):
                hashes = set(hashes)
            elif not isinstance(hashes, set):
                hashes = set(hashes)
            texts = data.get("texts", [])
            if not isinstance(texts, list):
                texts = list(texts)
            return hashes, texts, True
        elif isinstance(data, list):
            # old format: only texts were stored
            return set(), data, True
        else:
            # unknown format -> start fresh
            print("Cache format unknown; starting fresh.")
            return set(), [], False
    except Exception as e:
        print(f"Failed to load cache: {e}")
        return set(), [], False

def save_embeddings_cache(path, hashes, texts):
    joblib.dump({"hashes": list(hashes), "texts": texts}, path)

def load_or_init_index():
    hashes, texts, valid = load_embeddings_cache(EMBEDDING_CACHE_FILE)
    index = None
    if os.path.exists(FAISS_INDEX_FILE):
        try:
            index = faiss.read_index(FAISS_INDEX_FILE)
        except Exception as e:
            print(f"Failed to read FAISS index: {e}")
            index = None
    return index, hashes, texts

def get_new_csvs(folder):
    all_csvs = sorted(glob.glob(os.path.join(folder, "*.csv")))
    processed = set()
    if os.path.exists(PROCESSED_LOG):
        with open(PROCESSED_LOG, "r", encoding="utf-8") as f:
            processed = set(line.strip() for line in f if line.strip())
    return [f for f in all_csvs if f not in processed]

def mark_as_processed(files):
    if not files:
        return
    with open(PROCESSED_LOG, "a", encoding="utf-8") as f:
        for file in files:
            f.write(file + "\n")

# -------------------------------
# Text + Hash helpers (pandas rows)
# -------------------------------
def row_to_text(row):
    # join all non-NaN values with space; keeps your progress bars unchanged
    return " ".join(str(v) for v in row.values if pd.notna(v))

def hash_row(row) -> str:
    # stable row hash for deduplication
    row_str = "|".join(str(v) for v in row.values if pd.notna(v))
    return hashlib.md5(row_str.encode("utf-8")).hexdigest()

# -------------------------------
# Ensure FAISS matches model dim (auto-rebuild if needed)
# -------------------------------
def ensure_faiss_index(index, texts):
    """
    If index is None, create new one. If index dim != model dim, rebuild from texts.
    """
    def new_index():
        # HNSWFlat with M=32 is a good default; adjust if needed
        return faiss.IndexHNSWFlat(EMBED_DIM, 32)

    if index is None:
        idx = new_index()
        if texts:
            print("Building FAISS index from existing texts...")
            # embed in batches with progress bar
            all_embs = []
            for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Re-embedding cached texts"):
                batch = texts[i:i + BATCH_SIZE]
                embs = get_embedding(batch)
                all_embs.append(embs)
            all_embs = np.vstack(all_embs) if all_embs else np.zeros((0, EMBED_DIM), dtype=np.float32)
            if all_embs.shape[0] > 0:
                idx.add(all_embs)
            faiss.write_index(idx, FAISS_INDEX_FILE)
            print(f"FAISS rebuilt with {idx.ntotal} vectors.\n")
        return idx

    # If existing index dim mismatches, rebuild
    if getattr(index, "d", None) != EMBED_DIM:
        print(f"FAISS index dim ({index.d}) != model dim ({EMBED_DIM}). Rebuilding index...")
        idx = new_index()
        if texts:
            all_embs = []
            for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Re-embedding cached texts"):
                batch = texts[i:i + BATCH_SIZE]
                embs = get_embedding(batch)
                all_embs.append(embs)
            all_embs = np.vstack(all_embs) if all_embs else np.zeros((0, EMBED_DIM), dtype=np.float32)
            if all_embs.shape[0] > 0:
                idx.add(all_embs)
        faiss.write_index(idx, FAISS_INDEX_FILE)
        print(f"FAISS rebuilt with {idx.ntotal} vectors.\n")
        return idx

    return index

# -------------------------------
# Incremental CSV Update (keeps your progress bars)
# -------------------------------
def incremental_update(csv_path, index, seen_hashes, texts):
    df = pd.read_csv(csv_path)

    # 1) Dedup via hashes (progress bar unchanged)
    new_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Filtering duplicates using hash lookup"):
        row_hash = hash_row(row)
        if row_hash not in seen_hashes:
            seen_hashes.add(row_hash)
            new_rows.append(row)

    if not new_rows:
        print("No new rows to embed.")
        return index, seen_hashes, texts

    print(f"{len(new_rows)} new rows to embed")

    # 2) Convert rows to text (progress bar unchanged)
    new_texts = []
    for row in tqdm(new_rows, desc="Converting rows to text", unit="row"):
        new_texts.append(row_to_text(row))

    # 3) Generate embeddings in batches (preserve a single progress bar name)
    embeddings = []
    for i in tqdm(range(0, len(new_texts), BATCH_SIZE), desc="Generating embeddings", unit="batch"):
        batch = new_texts[i:i + BATCH_SIZE]
        embs = get_embedding(batch)
        embeddings.append(embs)
    embeddings_np = np.vstack(embeddings).astype("float32")

    # 4) Ensure FAISS exists and matches model dim
    index = ensure_faiss_index(index, texts)

    # 5) Add to FAISS
    index.add(embeddings_np)

    # 6) Update texts and checkpoint after this CSV (atomic-ish)
    texts.extend(new_texts)
    save_embeddings_cache(EMBEDDING_CACHE_FILE, seen_hashes, texts)
    faiss.write_index(index, FAISS_INDEX_FILE)
    print("Checkpoint saved (cache + FAISS)\n")

    return index, seen_hashes, texts

# -------------------------------
# Retrieve rows
# -------------------------------
def retrieve_rows(prompt, index, texts, top_k=TOP_K):
    if index is None or index.ntotal == 0:
        print("FAISS index is empty; nothing to retrieve.")
        return []

    print("Generating embedding for the prompt...")
    query_emb = get_embedding([prompt]).reshape(1, -1)

    k = min(top_k, index.ntotal)
    print(f"Searching top {k} most relevant rows in FAISS index...")
    D, I = index.search(query_emb, k)
    retrieved = [texts[i] for i in I[0] if 0 <= i < len(texts)]
    print(f"Retrieved {len(retrieved)} rows from FAISS")

    print("Chunking rows for GPT (token-safe)...")
    chunks = []
    current_chunk, current_len = [], 0
    for row_text in tqdm(retrieved, desc="Creating GPT chunks"):
        current_chunk.append(row_text)
        current_len += len(row_text.split())
        if current_len > MAX_CHUNK_LENGTH:
            chunks.append("\n".join(current_chunk))
            current_chunk, current_len = [], 0
    if current_chunk:
        chunks.append("\n".join(current_chunk))

    print(f"Created {len(chunks)} chunks for GPT prompt\n")
    return chunks

# -------------------------------
# Generate trades using GPT (unchanged)
# -------------------------------
from openai import OpenAI
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

def generate_trades(prompt, chunks):
    context = "\n".join(chunks)
    full_prompt = f"""
You are a professional intraday stock trading assistant. Here is recent stock data (with ATR & VWAP):
{context}

{prompt}

Provide top 5 intraday trades (short and long) including:
- Entry
- Stop Loss
- Target (R:R >= 1:2)
- Exit Strategy
- previous day close price

Also, include strategy to trade NIFTY Options (PE/CE)
Output clearly and concisely.
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": full_prompt}],
        temperature=0
    )
    return response.choices[0].message.content

# -------------------------------
# Main RAG bot
# -------------------------------
def rag_trading_bot(prompt):
    print("Loading embeddings cache and FAISS index...")
    index, hashes, texts = load_or_init_index()
    # ensure types
    if isinstance(hashes, list):
        hashes = set(hashes)
    print(f"Loaded {len(texts)} existing texts\n")

    # ensure FAISS matches model before processing new CSVs
    index = ensure_faiss_index(index, texts)

    new_csvs = get_new_csvs(CSV_FOLDER)
    if not new_csvs:
        print("No new CSV files to process")
    else:
        print(f"Found {len(new_csvs)} new CSV files to process\n")

    for csv_path in new_csvs:
        index, hashes, texts = incremental_update(csv_path, index, hashes, texts)

    mark_as_processed(new_csvs)
    print("CSV processing complete. Retrieving relevant rows...\n")

    chunks = retrieve_rows(prompt, index, texts)
    print(f"Retrieved {len(chunks)} chunks for GPT prompt\n")

    trades = generate_trades(prompt, chunks)
    print("Generated trades successfully\n")
    return trades

# -------------------------------
# Example Usage
# -------------------------------
if __name__ == "__main__":
    user_prompt = "Suggest top 5 intraday trades for next trading day with all the stocks given to you."
    result = rag_trading_bot(user_prompt)
    print(result)
