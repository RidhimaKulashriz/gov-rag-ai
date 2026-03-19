import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
import tempfile
import google.generativeai as genai
import time
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

# Check for Google API Key (Gemini)
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    st.error("""
    Google Gemini API Key Not Found
    
    Please add your Google Gemini API key:
    
    1. Get your API key from Google AI Studio (https://aistudio.google.com/app/apikey)
    2. Create a `.env` file in the project root with:
       GOOGLE_API_KEY=your_api_key_here
    3. Restart the Streamlit app
    
    See `.env.example` for the template.
    """)
    st.stop()

# Configure Generative AI
genai.configure(api_key=google_api_key)

# Function to get available model - list all models and find working one
@st.cache_resource
def get_available_model():
    """Dynamically detect which models are actually available"""
    try:
        # List all available models
        all_models = genai.list_models()
        available_models = []
        
        for model in all_models:
            # Check if model supports generateContent
            if hasattr(model, 'supported_generation_methods'):
                if 'generateContent' in model.supported_generation_methods:
                    model_name = model.name.replace('models/', '')
                    available_models.append(model_name)
        
        if available_models:
            # Return the first available model
            return available_models[0]
        else:
            # Fallback if list is empty
            st.warning("Could not detect available models, using fallback")
            return "gemini-pro"
            
    except Exception as e:
        st.warning(f"Could not list models: {e}, using fallback 'gemini-pro'")
        return "gemini-pro"

# Rate limiting - track last request time
if 'last_request_time' not in st.session_state:
    st.session_state.last_request_time = None
if 'rate_limit_reset_time' not in st.session_state:
    st.session_state.rate_limit_reset_time = None

def enforce_rate_limit():
    """Enforce rate limiting to prevent quota exhaustion"""
    # If we hit a rate limit, wait until reset time
    if st.session_state.rate_limit_reset_time:
        now = datetime.now()
        if now < st.session_state.rate_limit_reset_time:
            wait_seconds = (st.session_state.rate_limit_reset_time - now).total_seconds()
            st.error(f"""
            Rate Limit Hit on Gemini API
            
            Your free tier quota has been exceeded. Please wait **{int(wait_seconds)} seconds** before trying again.
            
            To increase your quota:
            1. Upgrade to a paid plan at https://ai.google.dev/pricing
            2. Add a billing method to your Google Cloud project
            3. Or wait for your quota to reset
            """)
            return False
        else:
            # Reset time has passed, clear the restriction
            st.session_state.rate_limit_reset_time = None
    
    return True

def call_llm_with_retry(llm, prompt, max_retries=1):
    """Call LLM with minimal retry - optimized for free tier to fail fast"""
    try:
        response = llm.invoke(prompt)
        return response
    except Exception as e:
        error_str = str(e)
        
        # Check for rate limit error (429)
        if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
            # Extract retry delay if available
            retry_delay = 60
            
            if "retry in" in error_str.lower():
                import re
                match = re.search(r'retry in (\d+(?:\.\d+)?)', error_str.lower())
                if match:
                    retry_delay = max(int(float(match.group(1))) + 5, 60)
            
            # Set rate limit reset time
            st.session_state.rate_limit_reset_time = datetime.now() + timedelta(seconds=retry_delay)
            
            raise Exception(f"⚠️ Free tier quota exhausted. Please wait {retry_delay} seconds before trying again, or upgrade your plan at https://ai.google.dev/pricing")
        else:
            # Not a rate limit error, raise immediately
            raise e

# Page configuration
st.set_page_config(
    page_title="Government RAG AI Assistant",
    page_icon="",
    layout="wide"
)

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Custom CSS - Black Theme
st.markdown("""
<style>
    :root {
        --primary-color: #ffffff;
        --background-color: #1a1a1a;
        --secondary-bg: #2d2d2d;
        --text-color: #ffffff;
        --border-color: #404040;
        --accent-color: #00d4ff;
    }
    
    .stApp {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    
    .main {
        background-color: #1a1a1a;
    }
    
    .sidebar .sidebar-content {
        background-color: #2d2d2d;
    }
    
    .stSidebar {
        background-color: #2d2d2d;
    }
    
    .stHeader {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff;
    }
    
    .main-header {
        color: #ffffff;
        text-align: center;
        padding: 2rem;
        background-color: #2d2d2d;
        border: 2px solid #404040;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .stButton > button {
        background-color: #00d4ff;
        color: #000000;
        font-weight: bold;
        border: none;
    }
    
    .stButton > button:hover {
        background-color: #00b8cc;
    }
    
    .stTextInput > div > div > input {
        background-color: #2d2d2d;
        color: #ffffff;
        border: 1px solid #404040;
    }
    
    .stChatMessage {
        background-color: #2d2d2d;
        border: 1px solid #404040;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .stChatMessage.user {
        background-color: #2d5a7d;
        border-color: #00d4ff;
    }
    
    .stChatMessage.assistant {
        background-color: #3a3a3a;
        border-color: #00d4ff;
    }
    
    .stSpinner {
        color: #00d4ff;
    }
    
    .stSuccess {
        background-color: #1a4d2e;
        border: 1px solid #2d8659;
        color: #ffffff;
    }
    
    .stError {
        background-color: #4d1a1a;
        border: 1px solid #8b2d2d;
        color: #ffffff;
    }
    
    .stWarning {
        background-color: #4d3a1a;
        border: 1px solid #8b6b2d;
        color: #ffffff;
    }
    
    .stSlider {
        color: #ffffff;
    }
    
    .stSlider > div > div > div {
        background-color: #2d2d2d;
    }
    
    label {
        color: #ffffff;
    }
    
    .stFileUploader {
        background-color: #2d2d2d;
        border: 2px dashed #404040;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">Government RAG AI Assistant</h1>', 
            unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Document Upload")
    uploaded_files = st.file_uploader(
        "Upload government documents (PDF)",
        type=['pdf'],
        accept_multiple_files=True
    )
    
    st.header("Settings")
    chunk_size = st.slider("Chunk Size", 500, 2000, 1000)
    chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200)
    
    if st.button("Process Documents", type="primary"):
        if uploaded_files:
            with st.spinner("Processing documents..."):
                try:
                    # Initialize embeddings using Gemini
                    # Use sentence-transformers as fallback for better compatibility
                    from langchain_community.embeddings import HuggingFaceEmbeddings
                    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                    
                    all_texts = []
                    
                    for uploaded_file in uploaded_files:
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name
                        
                        # Load PDF
                        loader = PyPDFLoader(tmp_path)
                        documents = loader.load()
                        
                        # Split text
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            separators=["\n\n", "\n", " ", ""]
                        )
                        texts = text_splitter.split_documents(documents)
                        all_texts.extend(texts)
                        
                        # Clean up temp file
                        os.unlink(tmp_path)
                    
                    # Create vector store
                    st.session_state.vectorstore = Chroma.from_documents(
                        documents=all_texts,
                        embedding=embeddings,
                        persist_directory="./chroma_db"
                    )
                    
                    st.success(f"Processed {len(uploaded_files)} documents successfully!")
                    
                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")
        else:
            st.warning("Please upload documents first.")

# Main chat interface
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Chat with Documents")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        if st.session_state.vectorstore:
            with st.chat_message("assistant"):
                # Check rate limit first
                if not enforce_rate_limit():
                    st.session_state.messages.append(
                        {"role": "assistant", "content": "Rate limit exceeded. Please wait before trying again."}
                    )
                else:
                    with st.spinner("Thinking..."):
                        try:
                            # Get available model dynamically
                            model_name = get_available_model()
                            
                            # Initialize LLM with Gemini
                            llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
                            
                            # Get retriever - use k=2 to minimize token usage on free tier
                            retriever = st.session_state.vectorstore.as_retriever(
                                search_kwargs={"k": 2}
                            )
                            
                            # Retrieve relevant documents
                            docs = retriever.invoke(prompt)
                            context = "\n\n".join([doc.page_content for doc in docs])
                            
                            # Create optimized prompt - shorter template to save tokens
                            prompt_template = PromptTemplate(
                                input_variables=["context", "question"],
                                template="""Based on the context below, answer the question briefly.

Context:
{context}

Question: {question}

Answer:"""
                            )
                            
                            # Format prompt and get response with retry logic
                            formatted_prompt = prompt_template.format(
                                context=context,
                                question=prompt
                            )
                            
                            response = call_llm_with_retry(llm, formatted_prompt)
                            response_text = response.content if hasattr(response, 'content') else str(response)
                            st.markdown(response_text)
                            
                            # Add to session state
                            st.session_state.messages.append(
                                {"role": "assistant", "content": response_text}
                            )
                        except Exception as e:
                            error_msg = str(e)
                            st.error(f"Error generating response: {error_msg}")
                            
                            # Provide helpful guidance for common errors
                            if "quota" in error_msg.lower() or "429" in error_msg:
                                st.info("""
                                **Free Tier Quota Exceeded**
                                
                                Your free tier quota has been reached. Options:
                                1. Wait for your quota to reset (usually daily)
                                2. Upgrade to a paid plan: https://ai.google.dev/pricing
                                3. Use fewer/shorter documents
                                4. Ask more specific questions to reduce token usage
                                """)
                            elif "not found" in error_msg.lower():
                                st.info("""
                                **Model Not Available**
                                
                                The model is not available in your region or API tier. 
                                Try upgrading your API key or check availability.
                                """)
        else:
            with st.chat_message("assistant"):
                st.warning("Please upload and process documents first.")

with col2:
    st.header("Document Info")
    if st.session_state.vectorstore:
        st.info("Vector store is ready")
        if uploaded_files:
            st.write(f"Documents loaded: {len(uploaded_files)}")
    else:
        st.warning("No documents processed yet")
    
    # Clear chat button
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>"
    "Built with LangChain + Streamlit | Government RAG AI Assistant</p>", 
    unsafe_allow_html=True
)