import streamlit as st
import os
from datetime import datetime
import pandas as pd
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

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'csv_status' not in st.session_state:
    st.session_state.csv_status = {}

if 'current_input' not in st.session_state:
    st.session_state.current_input = ""

if 'auto_send' not in st.session_state:
    st.session_state.auto_send = False

if 'system_initialized' not in st.session_state:
    st.session_state.system_initialized = False

if 'rag_system_ready' not in st.session_state:
    st.session_state.rag_system_ready = False

if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = None

if 'texts' not in st.session_state:
    st.session_state.texts = []

if 'hashes' not in st.session_state:
    st.session_state.hashes = set()

if 'processing_query' not in st.session_state:
    st.session_state.processing_query = False

def initialize_system():
    """Initialize the RAG system and check CSV status"""
    try:
        with st.spinner("Initializing RAG system..."):
            index, hashes, texts = load_or_init_index()
            index = ensure_faiss_index(index, texts)
            
            # Check for new CSV files
            new_csvs = get_new_csvs("stock_csvs")
            
            # Cache the RAG system components
            st.session_state.faiss_index = index
            st.session_state.texts = texts
            st.session_state.hashes = hashes
            st.session_state.rag_system_ready = True
            
            # Update CSV status
            st.session_state.csv_status = {
                'total_texts': len(texts),
                'index_vectors': index.ntotal if index else 0,
                'new_csvs': len(new_csvs),
                'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            return True
    except Exception as e:
        st.error(f"Error initializing system: {str(e)}")
        return False

def process_new_csvs():
    """Process any new CSV files"""
    try:
        with st.spinner("Processing new CSV files..."):
            index = st.session_state.faiss_index
            hashes = st.session_state.hashes
            texts = st.session_state.texts

            index, hashes, texts, processed = process_new_data(index, hashes, texts)

            if processed:
                # Update cached components
                st.session_state.faiss_index = index
                st.session_state.texts = texts
                st.session_state.hashes = hashes
                
                # Update status
                st.session_state.csv_status = {
                    'total_texts': len(texts),
                    'index_vectors': index.ntotal if index else 0,
                    'new_csvs': 0,
                    'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                st.success(f"Processed new CSV files!")
            else:
                st.info("No new CSV files to process.")
            return True
    except Exception as e:
        st.error(f"Error processing CSV files: {str(e)}")
        return False

def chat_with_rag(user_input):
    """Process user input through RAG system"""
    try:
        # Prevent duplicate processing
        if st.session_state.processing_query:
            return "⏳ Already processing a query. Please wait..."
        
        st.session_state.processing_query = True
        
        with st.spinner("Analyzing your request using cached RAG system..."):
            if not st.session_state.get('rag_system_ready'):
                st.session_state.processing_query = False
                return "❌ RAG system not ready. Please wait for initialization to complete."
            
            index = st.session_state.get('faiss_index')
            texts = st.session_state.get('texts')
            
            if index is None or not texts:
                st.session_state.processing_query = False
                return "❌ RAG system components not available. Please reinitialize."
            
            response = rag_trading_bot(user_input, index, texts)
            
            st.session_state.processing_query = False
            return response
            
    except Exception as e:
        st.session_state.processing_query = False
        return f"❌ Error processing your request: {str(e)}"

# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-header">📊 System Status</div>', unsafe_allow_html=True)
    
    # Initialize button
    if st.button("🔄 Initialize System", use_container_width=True):
        if initialize_system():
            st.success("System initialized successfully!")
    
    # Process CSV button
    if st.button("📁 Process New CSVs", use_container_width=True):
        if process_new_csvs():
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
    st.markdown("### 💡 Sample Questions")
    sample_questions = [
        "Suggest top 5 intraday trades for next trading day",
        "What are the best options trading strategies?",
        "Show me stocks with high volatility for day trading",
        "Recommend swing trading opportunities",
        "Analyze market sentiment for NIFTY options"
    ]
    
    for question in sample_questions:
        if st.button(question, key=f"sample_{hash(question)}"):
            st.session_state.current_input = question
            st.session_state.auto_send = True
            st.rerun()

# Main content
st.markdown('<div class="main-header">🤖 AlgoTrader RAG Chatbot</div>', unsafe_allow_html=True)

# Show system status
if st.session_state.system_initialized:
    st.success("✅ System Ready - You can start asking questions!")
else:
    st.warning("⏳ System is initializing... Please wait.")



# Initialize system on first load (only once)
if 'system_initialized' not in st.session_state:
    st.session_state.system_initialized = False

if not st.session_state.system_initialized:
    with st.spinner("🚀 Starting up RAG system for the first time..."):
        if initialize_system():
            st.session_state.system_initialized = True
            st.success("✅ RAG system initialized successfully!")
            st.rerun()

# Chat interface
st.markdown("### 💬 Chat with Your Trading Assistant")

# Show current input if it exists
if st.session_state.current_input and not st.session_state.auto_send:
    st.info(f"📝 **Current Question:** {st.session_state.current_input}")
    if st.button("🚀 Send This Question", key="send_current"):
        user_input = st.session_state.current_input
        st.session_state.current_input = ""
        
        # Add user message to chat history
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        })
        
        # Get bot response
        bot_response = chat_with_rag(user_input)
        
        # Add bot response to chat history
        st.session_state.chat_history.append({
            'role': 'bot',
            'content': bot_response,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        })
        
        st.rerun()



# Display chat history
st.markdown("### 📝 Chat History")

if st.session_state.chat_history:
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>👤 You ({message['timestamp']}):</strong><br>
                {message['content']}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message bot-message">
                <strong>🤖 Assistant ({message['timestamp']}):</strong><br>
                {message['content'].replace(chr(10), '<br>')}
            </div>
            """, unsafe_allow_html=True)
else:
    st.info("No chat history yet. Start a conversation!")

# Clear chat button
if st.session_state.chat_history:
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div class="footer-text" style='text-align: center;'>
    <p>Powered by RAG (Retrieval-Augmented Generation) | Built with Streamlit</p>
    <p>Your AI-powered trading assistant for stock analysis and options trading</p>
</div>
""", unsafe_allow_html=True)

# Fixed input box at bottom (ChatGPT style)
st.markdown("---")
st.markdown("### 💬 Ask Your Question")

# Auto-send if sample question was selected
if st.session_state.auto_send and st.session_state.current_input:
    # Automatically process the sample question
    user_input = st.session_state.current_input
    st.session_state.auto_send = False
    st.session_state.current_input = ""
    
    # Add user message to chat history
    st.session_state.chat_history.append({
        'role': 'user',
        'content': user_input,
        'timestamp': datetime.now().strftime("%H:%M:%S")
    })
    
    # Get bot response
    bot_response = chat_with_rag(user_input)
    
    # Add bot response to chat history
    st.session_state.chat_history.append({
        'role': 'bot',
        'content': bot_response,
        'timestamp': datetime.now().strftime("%H:%M:%S")
    })
    
    # Clear auto-send flag and rerun
    st.rerun()

# User input at bottom
user_input = st.text_input(
    "Ask me about trading strategies, stock analysis, or options trading:",
    key="user_input",
    placeholder="Context: I am looking for the top 10 performing stocks for the next trading day that can be traded intraday for profits. I have attached 1-minute candle data for the last 7 working days. Requirements: Backtesting: Test your strategy thoroughly on the attached data using proper backtesting methods. Only suggest stocks where the strategy is consistently profitable. Selection Criteria: Identify stocks that meet a risk to reward ratio of 1:2. Provide only those stocks where trades can be executed with disciplined stop-loss and target levels. Output Format: For each stock, provide: Stock name/symbol Last close price Entry price Exit price (target) Stop-loss price Risk to reward ratio (must be 1:2) Accuracy/confidence level based on backtesting results Strategy Details: Explain the technical indicators and parameters used (e.g., RSI, EMA, MACD, Bollinger Bands). Provide clear entry and exit rules with conditions. Explain how stop-loss and target are calculated to maintain the 1:2 risk to reward. Share any filters or conditions applied (e.g., volume spikes, trend direction, volatility). Provide trade management rules like when to scale out or exit early if needed. Validation: Show the backtesting performance summary (win %, number of trades, average profit/loss, drawdown, etc.) Justify why these stocks are the best candidates for the next day’s intraday trading. Constraints: Only use the last 7 working days of 1-minute candles attached. Ensure realistic trade execution, slippage considerations, and transaction costs are accounted for. Do not include stocks with insufficient data or inconsistent patterns.",
    value=""
)

# Send button below input
col1, col2 = st.columns([1, 4])
with col1:
    send_button = st.button("🚀 Send", use_container_width=True)

# Process user input
if send_button and user_input:
    # Add user message to chat history
    st.session_state.chat_history.append({
        'role': 'user',
        'content': user_input,
        'timestamp': datetime.now().strftime("%H:%M:%S")
    })
    
    # Get bot response
    bot_response = chat_with_rag(user_input)
    
    # Add bot response to chat history
    st.session_state.chat_history.append({
        'role': 'bot',
        'content': bot_response,
        'timestamp': datetime.now().strftime("%H:%M:%S")
    })
    
    # Clear input and rerun to show response
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div class="footer-text" style='text-align: center;'>
    <p>Powered by RAG (Retrieval-Augmented Generation) | Built with Streamlit</p>
    <p>Your AI-powered trading assistant for stock analysis and options trading</p>
</div>
""", unsafe_allow_html=True)
