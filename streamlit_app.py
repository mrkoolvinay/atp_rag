import streamlit as st
import os
import html
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(override=True)

from trade_rag import rag_trading_bot, load_or_init_index, get_new_csvs, process_new_data, ensure_faiss_index

# Page configuration
st.set_page_config(
    page_title="AlgoTrader RAG Chatbot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
        color: #2c3e50;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left-color: #2196f3;
        color: #1565c0;
    }
    .bot-message {
        background-color: #f3e5f5;
        border-left-color: #9c27b0;
        color: #7b1fa2;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        text-align: center;
        color: #2c3e50;
    }
    /* Ensure all text is visible */
    .stMarkdown, .stText, .stButton > button {
        color: #2c3e50 !important;
    }
    /* Chat message text colors */
    .chat-message strong {
        color: #1a237e !important;
    }
    /* Footer text */
    .footer-text {
        color: #34495e !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'rag_system_ready' not in st.session_state:
    st.session_state.rag_system_ready = False
if 'csv_status' not in st.session_state:
    st.session_state.csv_status = {}

# --- Backend Functions ---
def initialize_system():
    with st.spinner("Initializing RAG system..."):
        index, hashes, texts = load_or_init_index()
        index = ensure_faiss_index(index, texts)
        st.session_state.faiss_index = index
        st.session_state.texts = texts
        st.session_state.hashes = hashes
        st.session_state.rag_system_ready = True
        
        # Update CSV status after initialization
        st.session_state.csv_status = {
            'total_texts': len(texts),
            'index_vectors': index.ntotal if index else 0,
            'new_csvs': len(get_new_csvs("stock_csvs")),
            'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    return True

def process_new_csvs():
    with st.spinner("Processing new CSV files..."):
        index, hashes, texts, processed = process_new_data(
            st.session_state.faiss_index,
            st.session_state.hashes,
            st.session_state.texts
        )
        if processed:
            st.session_state.faiss_index = index
            st.session_state.texts = texts
            st.session_state.hashes = hashes
            
            # Update CSV status after processing
            st.session_state.csv_status = {
                'total_texts': len(texts),
                'index_vectors': index.ntotal if index else 0,
                'new_csvs': len(get_new_csvs("stock_csvs")),
                'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.success("Processed new CSV files!")
        else:
            st.info("No new CSV files to process.")

def chat_with_rag(user_input, model_choice):
    with st.spinner(f"Analyzing your request with {model_choice}..."):
        response = rag_trading_bot(
            prompt=user_input, 
            index=st.session_state.faiss_index, 
            texts=st.session_state.texts,
            model_choice=model_choice.lower()
        )
        return response

# --- UI Components ---

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    if st.button("🔄 Re-initialize System", use_container_width=True):
        initialize_system()
        st.rerun()
    if st.button("📁 Process New CSVs", use_container_width=True):
        process_new_csvs()
        st.rerun()
    
    # Display system status
    if st.session_state.csv_status:
        st.markdown("### Current Status")
        st.metric("Total Texts", st.session_state.csv_status.get('total_texts', 0))
        st.metric("Index Vectors", st.session_state.csv_status.get('index_vectors', 0))
        st.metric("New CSVs", st.session_state.csv_status.get('new_csvs', 0))
        
        if 'last_updated' in st.session_state.csv_status:
            st.caption(f"Last updated: {st.session_state.csv_status['last_updated']}")

    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# Main content
st.markdown("<h1>🤖 AlgoTrader RAG Chatbot</h1>", unsafe_allow_html=True)

# System Initialization
if not st.session_state.rag_system_ready:
    initialize_system()
    st.rerun()

# Model Selection
st.markdown("### 🧠 Choose AI Model")
model_choice = st.radio(
    "Select the model to use for analysis:",
    ("Gemini", "OpenAI"),
    index=0, # Default to Gemini
    horizontal=True,
)

# Chat Interface
st.markdown("### 💬 Chat with Your Trading Assistant")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input (using st.text_input with a send button)
user_input = st.text_input(
    "Ask me about trading strategies, stock analysis, or options trading:",
    key="user_input",
    placeholder="screen all stocks",
    max_chars=2000
)

send_button = st.button("🚀 Send", use_container_width=True)

if send_button and user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response = chat_with_rag(user_input, model_choice)
        st.markdown(response)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
