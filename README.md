# AlgoTrader RAG Chatbot with Streamlit GUI

A powerful RAG (Retrieval-Augmented Generation) chatbot for stock trading and options analysis, now with an intuitive Streamlit web interface.

## Features

- 🤖 **AI-Powered Trading Assistant**: Get intelligent trading recommendations based on your stock data
- 📊 **Real-time Data Processing**: Automatically processes new CSV files and updates the knowledge base
- 💬 **Interactive Chat Interface**: Ask follow-up questions and get detailed responses
- 📈 **Stock Analysis**: Intraday trades, swing trading opportunities, and options strategies
- 🔍 **Smart Search**: Uses FAISS vector search to find the most relevant stock data for your queries

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file in the root directory with your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Data Structure

Ensure your CSV files are in the `stock_csvs/` folder. The system will automatically:
- Process new CSV files
- Generate embeddings
- Build a searchable index
- Track processed files

## Usage

### Running the Streamlit App

```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

### Using the Chatbot

1. **Initialize System**: Click "🔄 Initialize System" to load the RAG system
2. **Process Data**: Click "📁 Process New CSVs" to handle any new data files
3. **Ask Questions**: Type your trading-related questions in the chat interface
4. **Follow-up**: Ask follow-up questions based on previous responses

### Sample Questions

- "Suggest top 5 intraday trades for next trading day"
- "What are the best options trading strategies?"
- "Show me stocks with high volatility for day trading"
- "Recommend swing trading opportunities"
- "Analyze market sentiment for NIFTY options"

## Features

### Sidebar Controls
- **System Status**: View current system metrics and data counts
- **Initialize System**: Load the RAG system and check for new data
- **Process CSVs**: Handle new CSV files automatically
- **Sample Questions**: Quick access to common queries

### Chat Interface
- **Real-time Responses**: Get instant trading insights
- **Chat History**: View your conversation history
- **Follow-up Questions**: Build on previous responses
- **Clear History**: Reset conversation when needed

### Data Processing
- **Automatic Updates**: Processes new CSV files as they're added
- **Smart Deduplication**: Prevents duplicate data processing
- **Vector Search**: Fast, accurate retrieval of relevant information
- **Embedding Cache**: Efficient storage and retrieval of text embeddings

## File Structure

```
rag/
├── streamlit_app.py          # Main Streamlit application
├── trade_rag.py             # Core RAG functionality
├── requirements.txt          # Python dependencies
├── README.md                # This file
├── .env                     # Environment variables (create this)
├── stock_csvs/              # Your CSV data files
├── stock_embeddings.pkl     # Cached embeddings
├── faiss_stock_hnsw.index  # FAISS search index
└── processed_files.txt      # Track processed files
```

## Technical Details

- **Embedding Model**: SentenceTransformers "all-MiniLM-L6-v2"
- **Vector Database**: FAISS HNSW index for fast similarity search
- **LLM**: OpenAI GPT-4 for generating trading recommendations
- **Framework**: Streamlit for the web interface
- **Data Processing**: Pandas for CSV handling and data manipulation

## Troubleshooting

### Common Issues

1. **OpenAI API Error**: Ensure your API key is correctly set in `.env`
2. **Memory Issues**: Large datasets may require more RAM; consider reducing batch sizes
3. **CSV Processing**: Ensure CSV files are properly formatted and accessible

### Performance Tips

- Process large CSV files in smaller batches
- Use the "Process New CSVs" button only when needed
- Clear chat history periodically for better performance

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the chatbot.

## License

This project is for educational and trading analysis purposes. Use at your own risk and always do your own research before making trading decisions.



