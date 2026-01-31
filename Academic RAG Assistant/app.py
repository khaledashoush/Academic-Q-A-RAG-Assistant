"""
Academic RAG Assistant - Professional Final Version
"""

import os
import sys
import tempfile
import shutil
import time
import json
from pathlib import Path
import traceback

import streamlit as st
import pandas as pd
from datetime import datetime
import hashlib

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from utils import DocumentProcessor, VectorStoreManager, LLMManager, PromptManager

# Set page configuration with professional theme
st.set_page_config(
    page_title=config.APP_NAME,
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': 'https://github.com/your-repo/issues',
        'About': f"# {config.APP_NAME}\nAcademic Research Assistant v{config.APP_VERSION}"
    }
)

# Professional CSS with Dark Gray Background and Professional Chat Design
st.markdown("""
<style>
    /* Base Styles with Dark Gray Background */
    .stApp {
        background: linear-gradient(135deg, #4b5563 0%, #6b7280 100%) !important;
        background-attachment: fixed;
    }
    
    /* Main content area */
    .main-content {
        background: #374151 !important;
        border-radius: 20px;
        padding: 20px;
        margin: 20px 0;
    }
    
    /* Header with Professional Gradient */
    .main-header {
        background: linear-gradient(135deg, #1e40af 0%, #1e3a8a 100%);
        padding: 2.5rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 12px 24px rgba(30, 64, 175, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.1) 50%, transparent 70%);
        animation: shine 3s infinite linear;
    }
    
    @keyframes shine {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    /* Professional Chat Messages */
    .chat-messages-container {
        background: #1f2937 !important;
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
        min-height: 500px;
        max-height: 600px;
        overflow-y: auto;
        border: 1px solid #4b5563;
    }
    
    /* User Message - Right Side */
    .user-message-wrapper {
        display: flex;
        justify-content: flex-end;
        margin: 1.5rem 0;
        padding-right: 10px;
    }
    
    .user-message {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        padding: 1.2rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        max-width: 70%;
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.3);
        position: relative;
        animation: slideInRight 0.4s ease-out;
    }
    
    .user-message::after {
        content: '';
        position: absolute;
        bottom: 0;
        right: -10px;
        width: 0;
        height: 0;
        border-left: 10px solid #3b82f6;
        border-top: 10px solid transparent;
        border-bottom: 10px solid transparent;
    }
    
    /* Assistant Message - Left Side */
    .assistant-message-wrapper {
        display: flex;
        justify-content: flex-start;
        margin: 1.5rem 0;
        padding-left: 10px;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 1.2rem 1.5rem;
        border-radius: 20px 20px 20px 5px;
        max-width: 70%;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.3);
        position: relative;
        animation: slideInLeft 0.4s ease-out;
    }
    
    .assistant-message::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: -10px;
        width: 0;
        height: 0;
        border-right: 10px solid #10b981;
        border-top: 10px solid transparent;
        border-bottom: 10px solid transparent;
    }
    
    @keyframes slideInRight {
        from { transform: translateX(50px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideInLeft {
        from { transform: translateX(-50px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    /* Message Content Styling */
    .message-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.8rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .message-role {
        font-weight: 700;
        font-size: 1.1em;
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }
    
    .message-time {
        font-size: 0.8em;
        opacity: 0.9;
        background: rgba(255, 255, 255, 0.15);
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
    }
    
    .message-content {
        line-height: 1.7;
        font-size: 1.05em;
        margin-top: 0.5rem;
        color: white;
    }
    
    .sources-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(255, 255, 255, 0.25);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.85em;
        margin-top: 1rem;
        color: white;
        font-weight: 600;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Clear Titles */
    h1, h2, h3, h4 {
        color: #f9fafb !important;
        font-weight: 700 !important;
    }
    
    .section-title {
        color: #f9fafb !important;
        font-weight: 700 !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
        padding: 10px 20px;
        background: linear-gradient(135deg, #4b5563 0%, #6b7280 100%);
        border-radius: 12px;
        border-left: 5px solid #3b82f6;
        border: 1px solid #4b5563;
    }
    
    /* Sidebar Cards with Dark Gray Background */
    .sidebar-card {
        background: linear-gradient(135deg, #4b5563 0%, #6b7280 100%) !important;
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        border: 1px solid #6b7280;
        color: #f9fafb;
    }
    
    /* Evaluation Section */
    .evaluation-section {
        background: linear-gradient(135deg, #4b5563 0%, #6b7280 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin-top: 1rem;
        border: 2px solid #6b7280;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        color: #f9fafb;
    }
    
    /* Evaluation Section Header */
    .eval-header {
        color: #f9fafb !important;
        font-weight: 700 !important;
        margin-bottom: 1rem !important;
    }
    
    .eval-buttons {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    
    .eval-btn {
        padding: 1rem;
        border: none;
        border-radius: 12px;
        cursor: pointer;
        font-weight: 700;
        font-size: 1em;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
        min-height: 100px;
    }
    
    .eval-btn:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
    }
    
    .eval-good {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
    }
    
    .eval-average {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
    }
    
    .eval-poor {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
    }
    
    /* Quick Questions Buttons */
    .quick-question-btn {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 1.2rem;
        cursor: pointer;
        transition: all 0.3s ease;
        text-align: center;
        box-shadow: 0 6px 12px rgba(99, 102, 241, 0.2);
        width: 100%;
    }
    
    .quick-question-btn:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 20px rgba(99, 102, 241, 0.3);
    }
    
    /* Statistics Card */
    .statistics-card {
        background: linear-gradient(135deg, #4b5563 0%, #6b7280 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        border: 1px solid #6b7280;
        color: #f9fafb;
    }
    
    /* Input Styling */
    .stTextInput > div > div > input {
        border-radius: 12px !important;
        border: 2px solid #3b82f6 !important;
        padding: 1rem 1.5rem !important;
        font-size: 1.1em !important;
        transition: all 0.3s ease !important;
        background: #1f2937 !important;
        color: white !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #9ca3af !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #1d4ed8 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2) !important;
    }
    
    /* Button Styling */
    .stButton > button {
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 1em !important;
        padding: 0.75rem 1.5rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        border: none !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2) !important;
    }
    
    /* Scrollbar Styling */
    .chat-messages-container::-webkit-scrollbar {
        width: 8px;
    }
    
    .chat-messages-container::-webkit-scrollbar-track {
        background: #1f2937;
        border-radius: 4px;
    }
    
    .chat-messages-container::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        border-radius: 4px;
    }
    
    /* Loading Animation */
    .thinking-indicator {
        text-align: center;
        padding: 2rem;
        color: #9ca3af;
    }
    
    .loading-dots {
        display: inline-flex;
        gap: 0.5rem;
        margin-bottom: 1rem;
    }
    
    .loading-dots span {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        animation: bounce 1.4s infinite ease-in-out both;
    }
    
    .loading-dots span:nth-child(1) { animation-delay: -0.32s; }
    .loading-dots span:nth-child(2) { animation-delay: -0.16s; }
    
    @keyframes bounce {
        0%, 80%, 100% { transform: scale(0); }
        40% { transform: scale(1); }
    }
    
    /* Hide Button for Evaluation */
    .hide-eval-btn {
        background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-size: 0.9em;
        cursor: pointer;
        transition: all 0.3s ease;
        margin-top: 1rem;
        width: 100%;
    }
    
    .hide-eval-btn:hover {
        background: linear-gradient(135deg, #4b5563 0%, #374151 100%);
    }
    
    /* Streamlit Component Overrides */
    .stInfo, .stSuccess, .stWarning, .stError {
        background-color: #4b5563 !important;
        color: #f9fafb !important;
        border: 1px solid #6b7280 !important;
    }
    
    /* Metric Cards */
    .stMetric {
        background: #1f2937 !important;
        border: 1px solid #4b5563 !important;
        border-radius: 10px !important;
        padding: 10px !important;
    }
    
    /* Expandable Sections */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #4b5563 0%, #6b7280 100%) !important;
        color: #f9fafb !important;
        border: 1px solid #6b7280 !important;
    }
    
    /* Welcome Screen */
    .welcome-container {
        background: linear-gradient(135deg, #4b5563 0%, #6b7280 100%);
        text-align: center;
        padding: 3rem;
        border-radius: 20px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.2);
        border: 1px solid #6b7280;
        color: #f9fafb;
    }
    
    /* Custom container for evaluation rating */
    .eval-rating-container {
        background: linear-gradient(135deg, #4b5563 0%, #6b7280 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-top: 1rem;
        border: 1px solid #6b7280;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #6b7280, transparent);
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state with improved evaluation handling
def initialize_session_state():
    """Initialize session state with professional features"""
    defaults = {
        'documents_processed': False,
        'vector_store': None,
        'llm_manager': None,
        'chat_history': [],
        'uploaded_files': [],
        'total_chunks': 0,
        'document_metadata': {},
        'debug_messages': [],
        'last_error': None,
        'processing_step': 'idle',
        'ollama_status': 'unknown',
        'evaluations': {},
        'show_evaluation': {},
        'evaluation_stats': {'good': 0, 'average': 0, 'poor': 0},
        'processing_files': False,
        'current_query': '',
        'quick_questions_asked': set(),
        'evaluation_response': None  # Track last evaluation response
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Test Ollama connection
def test_ollama_connection():
    """Test if Ollama is running"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            st.session_state.ollama_status = 'running'
            return True
    except Exception as e:
        st.session_state.ollama_status = 'not_running'
        if 'debug_messages' in st.session_state:
            st.session_state.debug_messages.append(f"Ollama connection failed: {str(e)}")
    return False

# Professional header
def render_header():
    """Render professional header with status indicators"""
    status_emoji = "✅" if st.session_state.ollama_status == 'running' else "❌"
    status_color = "#10b981" if st.session_state.ollama_status == 'running' else "#ef4444"
    
    st.markdown(f"""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 2.8em;">📚 {config.APP_NAME}</h1>
        <p style="font-size: 1.2em; margin: 0.5rem 0 1.5rem 0; opacity: 0.9;">
            Academic Research Assistant powered by Llama 3 & RAG
        </p>
        <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 1rem;">
            <div style="text-align: center;">
                <div style="font-size: 0.9em; opacity: 0.8;">Version</div>
                <div style="font-weight: bold;">v{config.APP_VERSION}</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 0.9em; opacity: 0.8;">Model</div>
                <div style="font-weight: bold;">{config.LLM_MODEL}</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 0.9em; opacity: 0.8;">Embeddings</div>
                <div style="font-weight: bold;">{config.EMBEDDING_MODEL}</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 0.9em; opacity: 0.8;">Ollama</div>
                <div style="font-weight: bold; color: {status_color};">{status_emoji} {st.session_state.ollama_status}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Professional sidebar with dark theme
def render_sidebar():
    """Render professional sidebar with cards"""
    with st.sidebar:
        # Document Upload Card
        st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
        st.markdown("### 📄 Document Upload")
        
        uploaded_files = st.file_uploader(
            "Upload academic documents",
            type=[".pdf", ".txt", ".docx", ".md"],
            accept_multiple_files=True,
            key="file_uploader",
            help="Upload PDF, TXT, DOCX, or MD files"
        )
        
        if uploaded_files:
            st.session_state.uploaded_files = uploaded_files
            
            with st.expander(f"📁 Uploaded Files ({len(uploaded_files)})", expanded=False):
                for file in uploaded_files:
                    file_size = len(file.getvalue()) / 1024 / 1024
                    st.info(f"**{file.name}** ({file_size:.2f} MB)")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Processing Settings Card
        st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
        st.markdown("### ⚙️ Processing Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            chunk_size = st.slider(
                "Chunk Size",
                min_value=500,
                max_value=2000,
                value=1000,
                step=100,
                help="Size of text chunks for processing"
            )
        with col2:
            chunk_overlap = st.slider(
                "Overlap",
                min_value=0,
                max_value=500,
                value=200,
                step=50,
                help="Overlap between chunks"
            )
        
        retrieval_k = st.slider(
            "Retrieval Count",
            min_value=1,
            max_value=10,
            value=4,
            step=1,
            help="Number of document chunks to retrieve"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Actions Card
        st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
        st.markdown("### 🚀 Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            process_btn = st.button(
                "Process & Initialize",
                type="primary",
                use_container_width=True,
                key="process_btn",
                disabled=not uploaded_files or st.session_state.get('processing_files', False),
                help="Process uploaded documents and initialize the system"
            )
        
        with col2:
            clear_btn = st.button(
                "Clear All",
                use_container_width=True,
                key="clear_btn",
                help="Clear all data and reset the system"
            )
        
        if st.session_state.get('processing_files', False):
            st.warning("⏳ Processing in progress...")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # System Status Card
        st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
        st.markdown("### 📊 System Status")
        
        if st.session_state.documents_processed:
            st.success(f"✅ **System Ready**")
            st.caption(f"📚 Chunks: {st.session_state.total_chunks}")
            st.caption(f"📄 Files: {len(st.session_state.uploaded_files)}")
            
            # Evaluation stats
            if any(st.session_state.evaluation_stats.values()):
                st.markdown("---")
                st.markdown("#### 📈 Response Ratings")
                for rating, count in st.session_state.evaluation_stats.items():
                    if count > 0:
                        emoji = {"good": "👍", "average": "😐", "poor": "👎"}[rating]
                        st.caption(f"{emoji} {rating.title()}: {count}")
        else:
            st.warning("⚠️ **Awaiting Documents**")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Debug Mode
        debug_mode = st.checkbox("Enable Debug Mode", value=False, key="debug_mode")
        
        if debug_mode and st.session_state.debug_messages:
            st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
            st.markdown("### 🔍 Debug Log")
            with st.expander("View Debug Messages", expanded=False):
                for msg in st.session_state.debug_messages[-10:]:
                    st.caption(f"• {msg}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        return uploaded_files, process_btn, clear_btn, debug_mode

# Process documents
def process_documents(uploaded_files, debug_mode=False):
    """Process uploaded documents"""
    
    def add_debug(msg):
        if debug_mode:
            timestamp = datetime.now().strftime("%H:%M:%S")
            st.session_state.debug_messages.append(f"[{timestamp}] {msg}")
    
    try:
        st.session_state.processing_files = True
        add_debug("Starting document processing...")
        
        temp_dir = tempfile.mkdtemp()
        add_debug(f"Created temp directory: {temp_dir}")
        
        all_chunks = []
        doc_processor = DocumentProcessor()
        vector_manager = VectorStoreManager()
        
        progress_bar = st.progress(0)
        status_container = st.empty()
        
        for idx, uploaded_file in enumerate(uploaded_files):
            add_debug(f"Processing file {idx+1}/{len(uploaded_files)}: {uploaded_file.name}")
            
            with status_container.container():
                st.info(f"**Processing:** {uploaded_file.name}")
            
            file_ext = Path(uploaded_file.name).suffix.lower()
            file_type = file_ext[1:] if file_ext.startswith('.') else file_ext
            temp_path = os.path.join(temp_dir, f"temp_{uploaded_file.name}")
            
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
            
            try:
                docs = doc_processor.load_document(temp_path, file_type)
                chunks = doc_processor.chunk_documents(
                    docs,
                    chunk_size=st.session_state.get('chunk_size', 1000),
                    chunk_overlap=st.session_state.get('chunk_overlap', 200)
                )
                all_chunks.extend(chunks)
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
            progress_bar.progress((idx + 1) / len(uploaded_files))
        
        add_debug(f"Total chunks created: {len(all_chunks)}")
        
        with status_container.container():
            st.info("**Creating vector store...**")
        
        vector_store = vector_manager.create_vector_store(all_chunks)
        
        with status_container.container():
            st.info("**Initializing AI model...**")
        
        llm_manager = LLMManager()
        llm = llm_manager.initialize_llm(model=config.LLM_MODEL)
        
        st.session_state.vector_store = vector_manager
        st.session_state.llm_manager = llm_manager
        st.session_state.total_chunks = len(all_chunks)
        st.session_state.documents_processed = True
        st.session_state.document_metadata = doc_processor.extract_metadata(all_chunks)
        st.session_state.processing_files = False
        
        add_debug("Processing complete!")
        shutil.rmtree(temp_dir)
        
        return True
        
    except Exception as e:
        error_msg = f"Error processing documents: {str(e)}"
        add_debug(error_msg)
        
        st.session_state.last_error = str(e)
        st.session_state.processing_files = False
        
        if debug_mode:
            st.error(f"❌ **Processing Error:** {str(e)}")
            with st.expander("Error Details"):
                st.code(traceback.format_exc(), language="python")
        
        return False

# Handle evaluation with immediate hide functionality
def handle_evaluation(message_index, rating):
    """Handle evaluation of a response"""
    st.session_state.evaluations[message_index] = rating
    
    if rating == 'good':
        st.session_state.evaluation_stats['good'] += 1
    elif rating == 'average':
        st.session_state.evaluation_stats['average'] += 1
    elif rating == 'poor':
        st.session_state.evaluation_stats['poor'] += 1
    
    # Immediately hide the evaluation section
    st.session_state.show_evaluation[message_index] = False
    
    # Store that we just evaluated this message
    st.session_state.evaluation_response = message_index
    
    # Show toast notification
    st.toast(f"✅ Response rated as: {rating.title()}", icon="👍")

# Professional chat interface with improved evaluation handling
def render_chat_interface(debug_mode=False):
    """Render professional chat interface"""
    
    # Chat container
    st.markdown('<div class="chat-messages-container">', unsafe_allow_html=True)
    
    if not st.session_state.chat_history:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; color: #9ca3af;">
            <div style="font-size: 4em; margin-bottom: 1rem;">💬</div>
            <h3>Start a Conversation</h3>
            <p>Ask questions about your documents or use quick questions below.</p>
        </div>
        """, unsafe_allow_html=True)
    
    for i, message in enumerate(st.session_state.chat_history):
        if message['role'] == 'user':
            # User message on the right
            user_html = f"""
            <div class="user-message-wrapper">
                <div class="user-message">
                    <div class="message-header">
                        <span class="message-role">👤 You</span>
                        <span class="message-time">{message.get('timestamp', '')}</span>
                    </div>
                    <div class="message-content">{message['content']}</div>
                </div>
            </div>
            """
            st.markdown(user_html, unsafe_allow_html=True)
        else:
            # Assistant message on the left
            sources_html = ""
            if message.get('sources') and int(message['sources']) > 0:
                sources_html = f'<div class="sources-badge">📚 Sources: {message["sources"]}</div>'
            
            assistant_html = f"""
            <div class="assistant-message-wrapper">
                <div class="assistant-message">
                    <div class="message-header">
                        <span class="message-role">🤖 Assistant</span>
                        <span class="message-time">{message.get('timestamp', '')}</span>
                    </div>
                    <div class="message-content">{message['content']}</div>
                    {sources_html}
                </div>
            </div>
            """
            st.markdown(assistant_html, unsafe_allow_html=True)
            
            # Check if we should show evaluation section
            show_eval = st.session_state.show_evaluation.get(i, False)
            eval_status = st.session_state.evaluations.get(i)
            
            # If we just evaluated this message, don't show the evaluation section
            if st.session_state.get('evaluation_response') == i:
                st.session_state.evaluation_response = None
                show_eval = False
            
            # Show evaluation section only if it's explicitly set to show
            if show_eval and not eval_status:
                # Create evaluation buttons using Streamlit for better control
                with st.container():
                    st.markdown('<div class="evaluation-section">', unsafe_allow_html=True)
                    st.markdown("### 📊 Rate this response")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("👍 Excellent", key=f"eval_good_{i}", use_container_width=True):
                            handle_evaluation(i, 'good')
                            st.rerun()
                    
                    with col2:
                        if st.button("😐 Good", key=f"eval_average_{i}", use_container_width=True):
                            handle_evaluation(i, 'average')
                            st.rerun()
                    
                    with col3:
                        if st.button("👎 Needs Work", key=f"eval_poor_{i}", use_container_width=True):
                            handle_evaluation(i, 'poor')
                            st.rerun()
                    
                    # Hide button - hides without rating
                    if st.button("Hide", key=f"hide_eval_{i}", use_container_width=True):
                        st.session_state.show_evaluation[i] = False
                        st.rerun()
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                # Show evaluation rating if already rated
                if eval_status:
                    status_emoji = {"good": "👍", "average": "😐", "poor": "👎"}[eval_status]
                    st.markdown(f"""
                    <div class="eval-rating-container">
                        <strong>{status_emoji} Rated: {eval_status.title()}</strong>
                        <br>
                        <small>Click "Update Rating" to change your rating</small>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Show rating/update button
            col1, col2 = st.columns([1, 2])
            with col1:
                if eval_status:
                    if st.button("Update Rating", key=f"update_{i}", use_container_width=True):
                        st.session_state.show_evaluation[i] = True
                        st.rerun()
                else:
                    if st.button("📊 Rate Response", key=f"rate_{i}", use_container_width=True):
                        st.session_state.show_evaluation[i] = True
                        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat input
    st.markdown('<div class="section-title">💭 Ask a Question</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([5, 1])
    with col1:
        question = st.text_input(
            "Type your question here...",
            key="question_input",
            label_visibility="collapsed",
            placeholder="e.g., What are the main research findings in these documents?",
            disabled=not st.session_state.documents_processed
        )
    
    with col2:
        send_btn = st.button(
            "Send",
            use_container_width=True,
            key="send_btn",
            type="primary",
            disabled=not st.session_state.documents_processed
        )
    
    if send_btn and question:
        handle_question(question, debug_mode)
    
    # Quick Questions
    st.markdown('<div class="section-title">⚡ Quick Questions</div>', unsafe_allow_html=True)
    st.markdown("Click any question below to ask instantly:")
    
    quick_questions = [
        ("📋", "Summarize Main Topics", "Summarize the main topics in these documents"),
        ("🔍", "Key Concepts", "What are the key concepts mentioned in the documents?"),
        ("📚", "Important References", "List important references cited in the documents"),
        ("🎯", "Research Methodology", "What research methodologies are discussed?"),
        ("📊", "Main Findings", "What are the main findings or conclusions?"),
        ("💡", "Research Gaps", "What research gaps are identified in the documents?"),
    ]
    
    cols = st.columns(3)
    for idx, (icon, title, question_text) in enumerate(quick_questions):
        with cols[idx % 3]:
            if st.button(
                f"{icon}\n**{title}**",
                key=f"quick_{idx}",
                use_container_width=True,
                help=question_text,
                disabled=not st.session_state.documents_processed
            ):
                question_hash = hashlib.md5(question_text.encode()).hexdigest()
                if question_hash not in st.session_state.quick_questions_asked:
                    st.session_state.quick_questions_asked.add(question_hash)
                
                handle_question(question_text, debug_mode)

# Handle question
def handle_question(question, debug_mode=False):
    """Handle user question"""
    
    if not st.session_state.documents_processed:
        st.warning("⚠️ Please process documents first!")
        return
    
    if not question or question.strip() == "":
        return
    
    try:
        st.session_state.chat_history.append({
            'role': 'user',
            'content': question,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        })
        
        # Thinking indicator
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown("""
        <div class="thinking-indicator">
            <div class="loading-dots">
                <span></span>
                <span></span>
                <span></span>
            </div>
            <p>Thinking...</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Get prompt template
        prompt_manager = PromptManager()
        prompt_template = prompt_manager.get_prompt_template("qa")
        
        # Retrieve relevant context
        vector_manager = st.session_state.vector_store
        retrieval_k = st.session_state.get('retrieval_k', 4)
        
        retriever = vector_manager.get_retriever(k=retrieval_k)
        
        # Get relevant documents
        try:
            relevant_docs = retriever.invoke(question)
        except AttributeError:
            try:
                relevant_docs = retriever._get_relevant_documents(question)
            except AttributeError:
                relevant_docs = vector_manager.vector_store.similarity_search(question, k=retrieval_k)
        
        # Generate response
        llm = st.session_state.llm_manager.llm
        
        if not relevant_docs:
            response = "I couldn't find relevant information in the provided documents to answer this question."
        else:
            context_parts = []
            for doc in relevant_docs[:3]:
                source = doc.metadata.get('source', 'Unknown')
                page = doc.metadata.get('page', 'N/A')
                context_parts.append(f"[From: {source} (Page {page})]\n{doc.page_content[:500]}...")
            
            context = "\n\n".join(context_parts)
            
            from langchain_core.prompts import PromptTemplate as LangPromptTemplate
            from langchain_classic.chains import LLMChain
            
            prompt = LangPromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            chain = LLMChain(llm=llm, prompt=prompt)
            
            response = chain.run(context=context, question=question)
            
            sources = list(set([
                doc.metadata.get('source', 'Unknown')
                for doc in relevant_docs[:3]
            ]))
            
            if sources:
                response += f"\n\n**Sources:** {', '.join(sources)}"
        
        # Clear thinking indicator
        thinking_placeholder.empty()
        
        # Add assistant response
        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': response,
            'sources': len(relevant_docs),
            'timestamp': datetime.now().strftime("%H:%M:%S")
        })
        
        st.rerun()
        
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        
        if 'thinking_placeholder' in locals():
            thinking_placeholder.empty()
        
        st.error(f"❌ **Error:** {str(e)[:200]}")
        
        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': f"Sorry, I encountered an error while processing your question: {str(e)[:100]}...",
            'error': True,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        })
        
        if debug_mode:
            with st.expander("Error Details"):
                st.code(traceback.format_exc(), language="python")

# Main application
def main():
    """Main application entry point"""
    
    # Initialize
    initialize_session_state()
    
    # Test Ollama connection
    test_ollama_connection()
    
    # Render header
    render_header()
    
    # Render sidebar
    uploaded_files, process_btn, clear_btn, debug_mode = render_sidebar()
    
    # Handle clear button
    if clear_btn:
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        initialize_session_state()
        st.rerun()
    
    # Handle process button
    if process_btn and uploaded_files:
        if len(uploaded_files) > 0:
            st.session_state.last_error = None
            success = process_documents(uploaded_files, debug_mode)
            
            if success:
                st.balloons()
                st.success("✅ Documents processed successfully!")
                st.rerun()
    
    # Main content
    if st.session_state.documents_processed:
        render_chat_interface(debug_mode)
        
        # Statistics section with clear title
        st.markdown('<div class="section-title">📈 Chat Statistics</div>', unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="statistics-card">', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                total_msgs = len(st.session_state.chat_history)
                st.metric("Total Messages", total_msgs)
            
            with col2:
                assistant_msgs = len([m for m in st.session_state.chat_history if m['role'] == 'assistant'])
                st.metric("Assistant Responses", assistant_msgs)
            
            with col3:
                rated_responses = len(st.session_state.evaluations)
                st.metric("Rated Responses", rated_responses)
            
            if rated_responses > 0:
                st.markdown("---")
                st.markdown("### 📊 Evaluation Summary")
                
                eval_data = pd.DataFrame({
                    'Rating': ['Good', 'Average', 'Poor'],
                    'Count': [
                        st.session_state.evaluation_stats['good'],
                        st.session_state.evaluation_stats['average'],
                        st.session_state.evaluation_stats['poor']
                    ]
                })
                
                st.bar_chart(eval_data.set_index('Rating'))
            
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Clean welcome screen with dark theme
        st.markdown("""
        <div class="welcome-container">
            <div style="font-size: 4em; margin-bottom: 1rem;">📚</div>
            <h2>Welcome to Academic RAG Assistant</h2>
            <p style="font-size: 1.1em; margin: 1.5rem 0;">
                Your intelligent research companion powered by AI
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Getting Started
        st.markdown("### 🚀 Getting Started")
        st.info("""
        1. **📄 Upload** academic documents using the sidebar
        2. **⚙️ Configure** processing settings as needed
        3. **🚀 Process** documents to initialize the system
        4. **💬 Ask questions** and explore your documents
        """)
        
        # Features
        st.markdown("### ✨ Key Features")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🤖 AI-Powered")
            st.write("Advanced AI models for accurate answers")
            
        with col2:
            st.markdown("#### 🔍 Smart Search")
            st.write("Intelligent document retrieval")
            
        with col3:
            st.markdown("#### 📊 Analytics")
            st.write("Comprehensive chat statistics")

if __name__ == "__main__":
    main()