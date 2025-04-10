import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    """Load the FAISS vector store."""
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    try:
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        st.error(f"Failed to load vector store: {e}")
        return None

def set_custom_prompt(custom_prompt_template):
    """Set a custom prompt template."""
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

def load_llm(huggingface_repo_id, HF_TOKEN):
    """Load the HuggingFace LLM."""
    try:
        llm = HuggingFaceEndpoint(
            repo_id=huggingface_repo_id,
            temperature=0.5,
            task="text-generation",
            model_kwargs={"token": HF_TOKEN, "max_length": "512"}
        )
        return llm
    except Exception as e:
        st.error(f"Failed to load LLM: {e}")
        return None

def format_response(result, source_documents):
    """Format the response and source documents for better readability."""
    # Format the answer
    formatted_result = f"### Answer:\n{result.strip()}\n"

    # Format the source documents
    formatted_sources = []
    for doc in source_documents:
        source = doc.metadata.get('source', 'Unknown')
        page = doc.metadata.get('page_label', 'Unknown')
        content = doc.page_content.strip()
        # Truncate content if it's too long
        if len(content) > 300:
            content = content[:300] + "..."
        formatted_sources.append(
            f"- **Source**: `{source}` (Page: {page})\n  **Content**: {content}"
        )

    # Combine the formatted result and sources
    formatted_sources_text = "\n\n".join(formatted_sources)
    return f"{formatted_result}\n### Source Documents:\n{formatted_sources_text}"

def main():
    # Custom CSS for styling
    st.markdown("""
    <style>
        body {
            background-color: #1e1e2f;
            color: #ffffff;
        }
        .stTextInput textarea {
            background-color: #2d2d3d;
            color: #ffffff;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 8px;
            transition: background-color 0.3s ease;
        }
                
        .stButton button:hover {
            background-color: #45a049;
        }
        /* Markdown text styling */
        .stMarkdown {
            color: #ffffff;
            font-size: 16px;
            line-height: 1.6;
        }
        .stSidebar {
            background-color: #2d2d3d;
        }
                
        /* Title styling */
        .stTitle {
            color: #4CAF50;
            font-size: 32px;
            font-weight: bold;
            text-align: center;
        }

        /* Caption styling */
        .stCaption {
            color: #ffffff;
            font-size: 14px;
            text-align: center;
            margin-bottom: 20px;
        }
                

        /* Chat history styling */
        .stChatMessage {
            background-color: #2d2d3d;
            color: #ffffff;
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 10px;
        }
                
        /* Sidebar styling */
        .stSidebar {
            background-color: #2d2d3d;
            color: #ffffff;
            padding: 20px;
        }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.markdown("### Model Settings")
        st.selectbox("Choose Model", ["Mistral-7B"], index=0)
        st.markdown("### About")
        st.markdown("""
        - **Medi-AI Chatbot** helps you find answers from medical documents.
        - Built with [LangChain](https://python.langchain.com/) and [Streamlit](https://streamlit.io/).
        """)

    # Main title
    st.markdown("<div class='stTitle'>🩺 Medi-AI Chatbot</div>", unsafe_allow_html=True)
    st.markdown("<div class='stCaption'>Your AI assistant for medical document queries.</div>", unsafe_allow_html=True)

    # Initialize session state for chat messages
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    # # Display chat history
    # for message in st.session_state.messages:
    #     role_class = 'user' if message['role'] == 'user' else 'assistant'
    #     st.markdown(f"<div class='stChatMessage {role_class}'>{message['content']}</div>", unsafe_allow_html=True)

    # User input
    prompt = st.chat_input("Ask your medical question here...")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer the user's question.
        If you don't know the answer, just say that you don't know. Don't try to make up an answer.
        Don't provide anything out of the given context.

        Context: {context}
        Question: {question}

        Start the answer directly. No small talk, please.
        """

        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN = os.environ.get("HF_TOKEN")

        try:
            # Load vector store
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store.")
                return

            # Initialize the QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            # Query the QA chain
            with st.spinner("Processing your query..."):
                response = qa_chain.invoke({'query': prompt})

                result = response["result"]
                source_documents = response["source_documents"]

                # Format the result and source documents
                formatted_output = format_response(result, source_documents)

                # Display the response
                st.chat_message('assistant').markdown(formatted_output)
                st.session_state.messages.append({'role': 'assistant', 'content': formatted_output})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()



# Future Improvements:
# Uploading documents to the vector store
# Adding more models and embeddings