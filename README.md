# 🩺 Medi-AI Chatbot

**Medi-AI Chatbot** is an AI-powered application designed to help users find answers from medical documents. Built using **LangChain** and **Streamlit**, this chatbot leverages advanced language models to provide accurate and context-aware responses.

---

## 🚀 Features

- **Interactive Chat**: Ask medical questions and get answers based on the uploaded documents.  
- **Preprocessed PDF Support**: Upload PDFs beforehand to create a vector store for efficient retrieval.  
- **Customizable Models**: Choose from different language models (e.g., Mistral-7B).  
- **Source Document References**: Provides references to the source documents for transparency.  
- **User-Friendly Interface**: Clean and modern UI with dark mode styling.

## 🛠️ Technologies Used

- **Python**: Core programming language.
- **Streamlit**: For building the interactive web application.
- **LangChain**: For managing the language model and retrieval-based QA.
- **HuggingFace**: For embeddings and language model integration.
- **FAISS**: For efficient vector-based document retrieval.
- **dotenv**: For managing environment variables.

---

## 📂 Project Structure

```
Medi-AI-Chatbot/ 
├── medai.py                # Main Streamlit application code
├── preprocess.py           # Script to preprocess PDFs and create the vector store
├── vectorstore/            # Directory for storing FAISS vector database
├── data/                   # Directory for storing PDFs
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Medi-AI-Chatbot.git
cd Medi-AI-Chatbot
```
### 2. Set Up a Virtual Environment

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

- Create a .env file in the project root and add the following:
  ```
  HF_TOKEN=your_huggingface_api_token
  ```
