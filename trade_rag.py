import os
import hashlib
import glob
import numpy as np
import faiss
from tqdm import tqdm
import joblib
from dotenv import load_dotenv

load_dotenv(override=True)

from sentence_transformers import SentenceTransformer
import pandas as pd
import re
import tiktoken
from openai import OpenAI
import json
from gemini_api import generate_from_gemini
import google.generativeai as genai

# ------------------------------- 
# Config
# ------------------------------- 
CSV_FOLDER = "stock_csvs"
EMBEDDING_CACHE_FILE = "stock_embeddings.pkl"
FAISS_INDEX_FILE = "faiss_stock_hnsw.index"
PROCESSED_LOG = "processed_files.txt"
TOP_K = 5000
BATCH_SIZE = 200
MAX_CHUNK_LENGTH = 1000
MAX_LLM_CONTEXT_TOKENS = 4000

# API Keys and Clients
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ------------------------------- 
# Load local embedding model
# ------------------------------- 
print("Loading local embedding model (SentenceTransformers)...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
EMBED_DIM = embed_model.get_sentence_embedding_dimension()
print(f"Embedding model loaded (dim={EMBED_DIM})")

# Initialize tiktoken tokenizer
tokenizer = tiktoken.encoding_for_model("gpt-4")

# ------------------------------- 
# Helper Functions
# ------------------------------- 
def get_embedding(texts):
    embs = embed_model.encode(
        texts,
        batch_size=64,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=False,
    )
    return embs.astype(np.float32)

def load_embeddings_cache(path):
    if not os.path.exists(path):
        return set(), [], False
    try:
        data = joblib.load(path)
        if isinstance(data, dict):
            hashes = data.get("hashes", set())
            if not isinstance(hashes, set):
                hashes = set(hashes)
            texts = data.get("texts", [])
            return hashes, texts, True
        return set(), data, True  # Old format
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
            f.write("\n".join(files) + "\n")

# ------------------------------- 
# Text + Hash helpers
# ------------------------------- 
def symbol_df_to_texts(symbol_df, max_tokens_per_doc=500):
    symbol = symbol_df['symbol'].iloc[0]
    texts = []
    current_text_parts = [f"Stock: {symbol}"]
    current_tokens = len(tokenizer.encode(current_text_parts[0]))
    for _, row in symbol_df.iterrows():
        ohlcv_data = [f"{col.lower()}: {row[col]}" for col in ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'interval'] if pd.notna(row[col])]
        row_text = f"  {', '.join(ohlcv_data)}"
        row_tokens = len(tokenizer.encode(row_text))
        if current_tokens + row_tokens > max_tokens_per_doc:
            if len(current_text_parts) > 1:
                texts.append("\n".join(current_text_parts))
            current_text_parts = [f"Stock: {symbol}", row_text]
            current_tokens = len(tokenizer.encode(current_text_parts[0])) + row_tokens
        else:
            current_text_parts.append(row_text)
            current_tokens += row_tokens
    if len(current_text_parts) > 1:
        texts.append("\n".join(current_text_parts))
    return texts

def hash_symbol_data(symbol_df):
    symbol = symbol_df['symbol'].iloc[0]
    sorted_df = symbol_df.sort_values(by='timestamp').reset_index(drop=True)
    data_str = sorted_df.to_string(index=False)
    return hashlib.md5((symbol + data_str).encode("utf-8")).hexdigest()

# ------------------------------- 
# Index and Data Processing
# ------------------------------- 
def ensure_faiss_index(index, texts):
    def new_index():
        return faiss.IndexHNSWFlat(EMBED_DIM, 32)
    if index is None:
        idx = new_index()
        if texts:
            print("Building FAISS index from existing texts...")
            all_embs = np.vstack([get_embedding(texts[i:i + BATCH_SIZE]) for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Re-embedding cached texts")])
            if all_embs.shape[0] > 0:
                idx.add(all_embs)
            faiss.write_index(idx, FAISS_INDEX_FILE)
            print(f"FAISS rebuilt with {idx.ntotal} vectors.\n")
        return idx
    if getattr(index, "d", None) != EMBED_DIM:
        print(f"FAISS index dim ({index.d}) != model dim ({EMBED_DIM}). Rebuilding index...")
        idx = new_index()
        if texts:
            all_embs = np.vstack([get_embedding(texts[i:i + BATCH_SIZE]) for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Re-embedding cached texts")])
            if all_embs.shape[0] > 0:
                idx.add(all_embs)
        faiss.write_index(idx, FAISS_INDEX_FILE)
        print(f"FAISS rebuilt with {idx.ntotal} vectors.\n")
        return idx
    return index

def incremental_update(csv_path, index, seen_hashes, texts):
    df = pd.read_csv(csv_path)
    new_texts = []
    for symbol, group_df in tqdm(df.groupby('symbol'), desc=f"Processing symbols from {os.path.basename(csv_path)}"):
        symbol_hash = hash_symbol_data(group_df)
        if symbol_hash not in seen_hashes:
            seen_hashes.add(symbol_hash)
            symbol_texts = symbol_df_to_texts(group_df)
            new_texts.extend(symbol_texts)
    if not new_texts:
        print(f"No new or updated symbols to embed from {os.path.basename(csv_path)}.")
        return index, seen_hashes, texts
    print(f"Found {len(new_texts)} new text documents to embed from {os.path.basename(csv_path)}")
    embeddings_np = np.vstack([get_embedding(new_texts[i:i + BATCH_SIZE]) for i in tqdm(range(0, len(new_texts), BATCH_SIZE), desc="Generating embeddings for new documents", unit="batch")])
    index = ensure_faiss_index(index, texts)
    index.add(embeddings_np)
    texts.extend(new_texts)
    save_embeddings_cache(EMBEDDING_CACHE_FILE, seen_hashes, texts)
    faiss.write_index(index, FAISS_INDEX_FILE)
    print("Checkpoint saved (cache + FAISS)\n")
    return index, seen_hashes, texts

def process_new_data(index, hashes, texts):
    """Checks for and processes new CSV files."""
    index = ensure_faiss_index(index, texts)
    new_csvs = get_new_csvs(CSV_FOLDER)
    if not new_csvs:
        print("No new CSV files to process")
        return index, hashes, texts, False
    print(f"Found {len(new_csvs)} new CSV files to process\n")
    for csv_path in new_csvs:
        index, hashes, texts = incremental_update(csv_path, index, hashes, texts)
    mark_as_processed(new_csvs)
    print("CSV processing complete.")
    return index, hashes, texts, True

# ------------------------------- 
# RAG Core Logic
# ------------------------------- 
def retrieve_rows(prompt, index, texts, top_k=TOP_K, screen_all=False, num_stocks_to_screen=None):
    if not texts:
        print("No texts available to retrieve.")
        return []
    if screen_all:
        print("Screening all available stocks. Bypassing similarity search.")
        retrieved = list(texts)
    else:
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
        if num_stocks_to_screen is not None and num_stocks_to_screen > 0:
            chunks_by_symbol = {}
            ordered_symbols = []
            for chunk in retrieved:
                try:
                    symbol = chunk.split("Stock: ")[1].split("\n")[0].strip()
                    if symbol not in chunks_by_symbol:
                        chunks_by_symbol[symbol] = []
                        ordered_symbols.append(symbol)
                    chunks_by_symbol[symbol].append(chunk)
                except IndexError:
                    continue
            symbols_to_include = ordered_symbols[:num_stocks_to_screen]
            selected_chunks = []
            for symbol in symbols_to_include:
                selected_chunks.extend(chunks_by_symbol[symbol])
            retrieved = selected_chunks
            print(f"Filtered to {len(retrieved)} chunks for {len(symbols_to_include)} unique symbols, providing full context for each.")
    print("Chunking rows for GPT (token-safe)...")
    full_text = "\n\n---\n\n".join(retrieved)
    tokens = tokenizer.encode(full_text)
    chunks = [tokenizer.decode(tokens[i:i + MAX_CHUNK_LENGTH]) for i in tqdm(range(0, len(tokens), MAX_CHUNK_LENGTH), desc="Creating GPT chunks")]
    print(f"Created {len(chunks)} chunks for GPT prompt\n")
    return chunks

def generate_trades(prompt, chunks, model_choice="gemini"):
    print(f"DEBUG: Starting Map-Reduce process with {len(chunks)} chunks using {model_choice}.")
    map_prompt_template = """You are a specialized stock analyst. Your task is to analyze the following stock data based on the user's request and extract potential intraday trading opportunities.

User's request: \"{user_prompt}\"\n**Analysis Instructions:**
1.  **If the user's request is generic (e.g., 'screen all stocks'), you MUST look for common, reliable technical patterns.** These include:
    *   **High-Volume Breakouts:** Identify stocks breaking above a key resistance level on significantly higher-than-average volume.
    *   **Support/Resistance Bounces:** Look for stocks bouncing off a well-defined support or resistance level.
    *   **Consolidation Patterns:** Find stocks that have been trading in a tight range and are showing signs of a potential breakout (e.g., a bull flag).
2.  **If the user's request is specific, prioritize that.** For example, if they ask for 'cheap stocks showing momentum', focus on low-priced stocks with recent upward price movement and high volume.
3.  **For each identified trade, you must provide a clear, data-driven justification.** Reference specific price levels, volume spikes, or other indicators from the data.

Stock data chunk:
---
{chunk}
---

Based on the data and the request, identify potential trades. For each trade, provide the stock symbol, entry price, target, stop-loss, and a brief justification.
Output your findings as a JSON object with a single key \"trades\" which contains a list of trade objects. Each trade object should have the following keys: \"symbol\", \"entry\", \"target\", \"stop_loss\", \"justification\".

Example format:
```json
{{
  \"trades\": [
    {{
      \"symbol\": \"AAPL\",
      \"entry\": 150.5,
      \"target\": 152.0,
      \"stop_loss\": 149.75,
      \"justification\": \"Breakout above recent resistance at 150.0 on high volume, suggesting further upside.\"
    }}
  ]
}}
```

If you don't find any valid trading opportunities in this chunk that match the user's request, return a JSON with an empty list: `{{ \"trades\": [] }}`.
    """ # Same prompt as before
    all_potential_trades = []
    for i, chunk in enumerate(chunks):
        print(f"Map Step: Processing chunk {i + 1}/{len(chunks)}")
        map_prompt = map_prompt_template.format(user_prompt=prompt, chunk=chunk)
        try:
            if model_choice == 'gemini':
                model = genai.GenerativeModel('gemini-2.0-flash-lite')
                response = model.generate_content(map_prompt)
                content = response.text
            else: # openai
                response = openai_client.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=[{"role": "user", "content": map_prompt}],
                    temperature=0.2,
                    response_format={"type": "json_object"}
                )
                content = response.choices[0].message.content
            try:
                match = re.search(r'\{.*\}', content, re.DOTALL)
                if match:
                    json_str = match.group(0)
                    data = json.loads(json_str)
                    if "trades" in data and isinstance(data["trades"], list):
                        all_potential_trades.extend(data["trades"])
                        print(f"  -> Extracted {len(data['trades'])} potential trades from chunk {i + 1}.")
                else:
                    print(f"  -> No JSON found in response from chunk {i+1}.")
            except json.JSONDecodeError:
                print(f"  -> Warning: Could not decode JSON from chunk {i + 1}. Content: {content[:200]}...")
        except Exception as e:
            print(f"  -> Error processing chunk {i + 1} in map step: {e}")
    if not all_potential_trades:
        return "After analyzing the data, I could not find any trading opportunities that match your criteria."
    print(f"Reduce Step: Consolidating {len(all_potential_trades)} potential trades.")
    reduce_prompt_template = """You are a senior portfolio manager. You have received a list of potential trading ideas from your team of analysts. Your task is to synthesize these ideas, remove duplicates, rank them based on the user's original request, and present a final, consolidated report.

Client's original request: \"{user_prompt}\"\nHere is the list of potential trades identified by your analysts (in JSON format):
---
{trades_json}
---

Please provide a single, clear, and concise response that directly answers the client's request. Do not just list all the trades you received. 
Your final output should be a polished, human-readable report. For example, if the client asked for the "top 10 stocks", provide a ranked list of the best 10 opportunities from the provided data, explaining the strategy and rationale as requested.

**IMPORTANT**: Conclude your report with the following disclaimer, exactly as written:
"**Disclaimer**: These are AI-generated trading ideas based on technical analysis of historical data. They are not financial advice. All trading involves risk, and you should conduct your own research and consult with a qualified financial advisor before making any investment decisions."
    """ # Same as before
    unique_trades = []
    seen = set()
    for trade in all_potential_trades:
        if isinstance(trade, dict) and "symbol" in trade and "entry" in trade:
            identifier = (trade["symbol"], trade["entry"])
            if identifier not in seen:
                unique_trades.append(trade)
                seen.add(identifier)
    print(f"  -> Reduced to {len(unique_trades)} unique trades.")
    trades_json = json.dumps(unique_trades, indent=2)
    reduce_prompt = reduce_prompt_template.format(user_prompt=prompt, trades_json=trades_json)
    try:
        if model_choice == 'gemini':
            model = genai.GenerativeModel('gemini-2.0-flash-lite')
            final_response = model.generate_content(reduce_prompt)
            return final_response.text
        else:
            final_response = openai_client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": "user", "content": reduce_prompt}],
                temperature=0.8,
            )
            return final_response.choices[0].message.content
    except Exception as e:
        print(f"Error in reduce step: {e}")
        return "I encountered an error when trying to consolidate the trades into a final report."

def rag_trading_bot(prompt, index, texts, model_choice="gemini"):
    if prompt.lower().strip() in ["list symbols", "list stocks"]:
        symbols = set()
        for text in texts:
            if text.startswith("Stock: "):
                try:
                    symbol = text.split("Stock: ")[1].split("\n")[0].strip()
                    symbols.add(symbol)
                except IndexError:
                    continue
        if symbols:
            return "Here are the stock symbols I can access:\n" + "\n".join(sorted(list(symbols)))
        else:
            return "Could not find any stock symbols in the processed data."
    screen_all = False
    num_stocks_to_screen = None
    if "screen all" in prompt.lower() or "all stocks" in prompt.lower():
        screen_all = True
        print(f"User requested to screen all available stocks.")
    else:
        match = re.search(r"screen (\d+) stocks", prompt.lower())
        if match:
            num_stocks_to_screen = int(match.group(1))
            if num_stocks_to_screen > 1000:
                return f"I can only screen up to 1000 stocks at a time."
            print(f"User requested to screen {num_stocks_to_screen} stocks.")
    print("Retrieving relevant rows...\n")
    chunks = retrieve_rows(prompt, index, texts, screen_all=screen_all, num_stocks_to_screen=num_stocks_to_screen)
    if not chunks:
        return "I couldn't find any relevant stock data for your query."
    trades = generate_trades(prompt, chunks, model_choice=model_choice)
    print("Generated trades successfully\n")
    return trades

if __name__ == "__main__":
    print("Loading embeddings cache and FAISS index...")
    index, hashes, texts = load_or_init_index()
    if isinstance(hashes, list):
        hashes = set(hashes)
    print(f"Loaded {len(texts)} existing texts\n")
    index, hashes, texts, _ = process_new_data(index, hashes, texts)
    user_prompt = "screen all stocks"
    result = rag_trading_bot(user_prompt, index, texts)
    print(result)