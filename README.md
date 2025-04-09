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

## ▶️ Workflow

### Step 1: Preprocess PDFs

Before running the Streamlit app, you need to preprocess the PDFs to create the vector store.

1. Place your PDFs in the `data/` directory.
2. Run the preprocessing script:

```bash
python preprocess.py
```
This script will:

- Load the PDFs.

- Split the content into chunks.

- Generate embeddings using HuggingFace models.

- Save the embeddings in the FAISS vector store.

### Step 2: Run the Streamlit App
Once the vector store is created, you can run the Streamlit app:

```bash
streamlit run medai.py
```

### Step 3: Interact with the Chatbot
- Open the application in your browser (usually at http://localhost:8501).
- Ask your medical questions in the chat input box.
- The chatbot will retrieve answers from the preprocessed vector store.

## 🖼️ Screenshots

### Main Interface

![Screenshot 2025-04-08 222845](https://github.com/user-attachments/assets/73c28c3d-8a3d-4b01-8214-3066d0bd0740)


## 🤝 Contributing

Contributions are welcome! If you'd like to contribute, please fork the repository and submit a pull request.

---

## 🌟 Acknowledgments

- [LangChain](https://python.langchain.com/)
- [Streamlit](https://streamlit.io/)
- [HuggingFace](https://huggingface.co/)
- [FAISS](https://github.com/facebookresearch/faiss)

--- 

## 🙋‍♂️ Support

If you encounter any issues or have questions, feel free to open an issue in the repository or contact me at [athulakhil28@gmail.com](mailto:athulakhil28@gmail.com).<br>
My LinkedIn profile : [LinkedIn](https://www.linkedin.com/in/h-athulkrishnan/)



