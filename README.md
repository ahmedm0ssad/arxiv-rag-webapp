# 🔍 arXiv RAG Web Application

> An intelligent research assistant that combines vector search with generative AI to answer questions about computer science research papers.

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🚀 Overview

This web application implements a **Retrieval-Augmented Generation (RAG)** pipeline that allows users to ask natural language questions about computer science research papers from arXiv. It combines semantic search with AI-powered response generation to provide accurate, contextual answers backed by relevant sources.

### 🎯 Key Features

- **🔍 Semantic Search**: Uses sentence transformers and FAISS for fast, accurate document retrieval
- **🤖 AI-Powered Responses**: Integration with Google's Gemini AI for natural language generation
- **📊 Source Attribution**: Shows which papers contributed to each answer with similarity scores
- **🎨 Modern UI**: Clean, responsive Bootstrap interface
- **⚡ Fast Performance**: Optimized vector search with normalized embeddings

## 🛠️ Technologies Used

- **Backend**: Flask, Python
- **AI/ML**: Sentence Transformers, FAISS, Google Gemini API
- **Frontend**: HTML, CSS, Bootstrap, JavaScript
- **Data Processing**: Pandas, NumPy, scikit-learn

## 📋 Prerequisites

- Python 3.7+
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/YOURUSERNAME/arxiv-rag-webapp.git
cd arxiv-rag-webapp
```

### 2. Set Up Environment
```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure API Key
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your Google Gemini API key
GOOGLE_API_KEY=your_actual_api_key_here
```

### 4. Prepare Data Files
Place your data files in the parent directory:
- `arxiv_rag_data.pkl` - Preprocessed paper chunks
- `arxiv_faiss_index` - FAISS vector index

### 5. Run the Application
```bash
python app.py
```

Navigate to `http://localhost:5000` and start querying!

## 💡 How It Works

1. **Query Processing**: User questions are normalized and embedded using sentence transformers
2. **Vector Search**: FAISS performs fast cosine similarity search across paper embeddings
3. **Context Assembly**: Top-k relevant chunks are selected and formatted
4. **AI Generation**: Google Gemini generates contextual responses using retrieved information
5. **Source Attribution**: Results include source papers and similarity scores

## 🔧 Project Structure

```
rag_webapp/
├── app.py                 # Main Flask application
├── templates/
│   └── index.html        # Frontend interface
├── static/               # CSS/JS assets
├── requirements.txt      # Python dependencies
├── .env                  # Environment template
└── README.md             # This file
```

## 🎯 Use Cases

- **Research Discovery**: Find relevant papers for literature reviews
- **Technical Q&A**: Get explanations of complex CS concepts with paper citations
- **Academic Writing**: Find supporting evidence and related work
- **Learning**: Explore topics with AI-guided explanations
- Display of source papers used to generate the response

## Technical Details

- Flask: Web framework
- FAISS: Efficient similarity search library
- Sentence Transformers: Text embedding model
- Google Generative AI: Large language model for text generation

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🔮 Future Enhancements

- [ ] Support for multiple document formats (PDF, Word, etc.)
- [ ] Advanced filtering options (date range, paper categories)
- [ ] User authentication and query history
- [ ] Citation export functionality
- [ ] Multi-language support
- [ ] Real-time arXiv paper ingestion

## 🚨 Troubleshooting

### Common Issues

**API Key Error**: Make sure your `.env` file contains a valid Google Gemini API key.

**Missing Data Files**: Ensure `arxiv_rag_data.pkl` and `arxiv_faiss_index` are in the parent directory.

**Slow Performance**: If running on CPU, consider using a smaller embedding model or reducing the number of retrieved chunks.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Sentence Transformers](https://www.sbert.net/) for semantic embeddings
- [FAISS](https://faiss.ai/) for efficient similarity search
- [Google Gemini](https://ai.google.dev/) for response generation
- [arXiv](https://arxiv.org/) for providing open access to research papers

## 📧 Contact

**Ahmed Mossad** - [ahmed.abdelfattah.mossad@gmail.com](mailto:ahmed.abdelfattah.mossad@gmail.com)

Project Link: [https://github.com/ahmedm0ssad/arxiv-rag-webapp](https://github.com/ahmedm0ssad/arxiv-rag-webapp)

![App UI Screenshot] (Screenshot (4).png)

![App UI Screenshot] (Screenshot (5).png)

---

⭐ **Star this repository if you found it helpful!**
