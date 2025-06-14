import streamlit as st
import tempfile
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
import time

# Load environment variables
load_dotenv()

# Initialize OpenAI client
@st.cache_resource
def initialize_openai():
    return OpenAI()

@st.cache_resource
def initialize_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-large")



def process_pdf(uploaded_file, collection_name):
    """Process uploaded PDF and store in vector database"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Load PDF
        loader = PyPDFLoader(file_path=tmp_file_path)
        docs = loader.load()
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=400
        )
        split_docs = text_splitter.split_documents(documents=docs)
        
        # Create embeddings and store in vector database
        embedding_model = initialize_embeddings()
        
        # ✅ Fixed: Use environment variables consistently
        vector_store = QdrantVectorStore.from_documents(
            documents=split_docs,
            url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            api_key=os.getenv("QDRANT_API_KEY"),
            collection_name=collection_name,
            embedding=embedding_model
        )
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return True, len(split_docs)
    
    except Exception as e:
        return False, str(e)

def get_vector_store(collection_name):
    """Get existing vector store"""
    try:
        embedding_model = initialize_embeddings()
        vector_db = QdrantVectorStore.from_existing_collection(
            url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            api_key=os.getenv("QDRANT_API_KEY"),
            collection_name=collection_name,
            embedding=embedding_model
        )
        return vector_db
    except Exception as e:
        st.error(f"Error connecting to vector database: {e}")
        return None

def chat_with_pdf(query, vector_db, client):
    """Generate response based on PDF context"""
    try:
        # Search for relevant documents
        search_results = vector_db.similarity_search(query=query)
        
        context = "\n\n\n".join([
            f"Page Content: {result.page_content}\nPage Number: {result.metadata['page_label']}\nFile Location: {result.metadata['source']}" 
            for result in search_results
        ])
        
        # Create system prompt
        system_prompt = f"""
        You are a helpful AI Assistant who answers user queries based on the available context
        retrieved from a PDF file along with page contents and page number.

        You should only answer the user based on the following context and guide the user
        to the right page number for more information.

        Context:
        {context}
        """
        
        # Generate response
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Fixed model name
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
        )
        
        return response.choices[0].message.content, search_results
    
    except Exception as e:
        return f"Error generating response: {e}", []

def format_file_size(size_bytes):
    """Convert bytes to human readable format"""
    if size_bytes == 0:
        return "0B"
    size_names = ["B", "KB", "MB", "GB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"

# Streamlit App
def main():
    st.set_page_config(
        page_title="PDF Chat Assistant",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .sidebar-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            text-align: center;
            color: white;
            font-weight: bold;
        }
        .status-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 10px;
            border-left: 4px solid #00cc88;
            margin: 1rem 0;
        }
        .upload-section {
            background-color: #fafbfc;
            padding: 1.5rem;
            border-radius: 15px;
            border: 2px dashed #cccccc;
            margin: 1rem 0;
            text-align: center;
        }
        .feature-item {
            display: flex;
            align-items: center;
            margin: 0.5rem 0;
            padding: 0.5rem;
            background-color: #f8f9fa;
            border-radius: 8px;
        }
        .stats-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("📚 Smart Chat using RAG")
    st.markdown("Transform your PDFs into interactive conversations with AI!")
    
    # Initialize session state
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False
    if 'collection_name' not in st.session_state:
        st.session_state.collection_name = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'vector_db' not in st.session_state:
        st.session_state.vector_db = None
    if 'processed_file_info' not in st.session_state:
        st.session_state.processed_file_info = None
    
    # Initialize clients
    try:
        client = initialize_openai()
    except Exception as e:
        st.error(f"Failed to initialize OpenAI client: {e}")
        st.stop()
    
    # Enhanced Sidebar
    with st.sidebar:
        # Step indicator
        st.markdown("### 📋 Quick Guide")
        steps = [
            ("1️⃣", "Upload PDF", st.session_state.get('uploaded_file') is not None),
            ("2️⃣", "Process Document", st.session_state.pdf_processed),
            ("3️⃣", "Start Chatting", len(st.session_state.chat_history) > 0)
        ]
        
        for emoji, step, completed in steps:
            status_icon = "✅" if completed else "⏳"
            st.markdown(f"{emoji} {step} {status_icon}")
        
        st.divider()
        
        # File Upload Section
        st.markdown("### 📁 Upload Document")
        
        uploaded_file = st.file_uploader(
            "Choose your PDF file",
            type="pdf",
            help="Supported format: PDF (max 200MB)",
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            st.session_state.uploaded_file = uploaded_file
            
            # File Information Card
            file_size = len(uploaded_file.getvalue())
            
            st.info(f"📄 **{uploaded_file.name}**\n📏 Size: {format_file_size(file_size)}")
            
            # Generate collection name from filename
            collection_name = f"pdf_{uploaded_file.name.replace('.pdf', '').replace(' ', '_').lower()}"
            
            # Process Button - Modified to show different text based on processing status
            if st.session_state.pdf_processed and st.session_state.collection_name == collection_name:
                # Show "Document Processed" when already processed
                st.button(
                    "✅ Document Processed", 
                    type="secondary",
                    use_container_width=True,
                    disabled=True
                )
            else:
                # Show "Process Document" when not processed
                process_btn = st.button(
                    "🚀 Process Document", 
                    type="primary",
                    use_container_width=True
                )
                
                if process_btn:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    with st.spinner("Processing your document..."):
                        status_text.text("📖 Reading PDF...")
                        progress_bar.progress(25)
                        time.sleep(0.5)
                        
                        status_text.text("✂️ Splitting into chunks...")
                        progress_bar.progress(50)
                        time.sleep(0.5)
                        
                        status_text.text("🧠 Creating embeddings...")
                        progress_bar.progress(75)
                        
                        success, result = process_pdf(uploaded_file, collection_name)
                        progress_bar.progress(100)
                        
                        if success:
                            st.session_state.pdf_processed = True
                            st.session_state.collection_name = collection_name
                            st.session_state.vector_db = get_vector_store(collection_name)
                            st.session_state.processed_file_info = {
                                'name': uploaded_file.name,
                                'chunks': result,
                                'size': format_file_size(file_size)
                            }
                            
                            status_text.text("✅ Processing complete!")
                            st.success(f"Successfully processed {result} chunks!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(f"❌ Processing failed: {result}")
                            status_text.text("❌ Processing failed")
                            progress_bar.empty()
        
        # Document Status Section
        if st.session_state.pdf_processed and st.session_state.processed_file_info:
            st.divider()
            
            # Show processed document info
            info = st.session_state.processed_file_info
            st.markdown("### 📊 Document Info")
            st.success(f"""
            **File:** {info['name']}
            **Size:** {info['size']}
            **Chunks:** {info['chunks']}
            """)
            
            # Action Buttons
            st.markdown("### ⚙️ Actions")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Clear Chat", use_container_width=True):
                    st.session_state.chat_history = []
                    st.success("Chat history cleared!")
                    time.sleep(0.5)
                    st.rerun()
            
            with col2:
                if st.button("🔄 New Doc", use_container_width=True):
                    # Reset session state
                    st.session_state.pdf_processed = False
                    st.session_state.collection_name = None
                    st.session_state.chat_history = []
                    st.session_state.vector_db = None
                    st.session_state.processed_file_info = None
                    if 'uploaded_file' in st.session_state:
                        del st.session_state.uploaded_file
                    st.rerun()
            
            # Chat Statistics
            if len(st.session_state.chat_history) > 0:
                st.markdown("### 💬 Chat Stats")
                st.metric("Questions Asked", len(st.session_state.chat_history))
        
        # Help Section
        st.divider()
        with st.expander("❓ Help & Tips"):
            st.markdown("""
            **How to get better results:**
            - Ask specific questions about the document
            - Reference particular sections or topics
            - Use keywords from the PDF content
            
            **Supported features:**
            - 🔍 Semantic search
            - 📄 Page references
            - 💬 Conversation memory
            - 📖 Source citations
            """)
    
    # Main chat interface
    if st.session_state.pdf_processed and st.session_state.vector_db:
        st.header("💬 Chat with your PDF")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for i, (question, answer, sources) in enumerate(st.session_state.chat_history):
                with st.chat_message("user"):
                    st.write(question)
                
                with st.chat_message("assistant"):
                    st.write(answer)
                    
                    # Show sources
                    if sources:
                        with st.expander("📖 Sources"):
                            for j, source in enumerate(sources[:1]):  # Show top 1 sources
                                st.write(f"**Source {j+1}:**")
                                st.write(f"Page: {source.metadata.get('page_label', 'N/A')}")
                                st.write(f"Content: {source.page_content[:200]}...")
                                st.divider()
        
        # Chat input
        with st.form("chat_form", clear_on_submit=True):
            col1, col2 = st.columns([4, 1])
            
            with col1:
                user_question = st.text_input(
                    "Ask a question about your PDF:",
                    placeholder="What is this document about?",
                    label_visibility="collapsed"
                )
            
            with col2:
                submit_button = st.form_submit_button("Send", type="primary")
        
        # Process user question
        if submit_button and user_question:
            with st.spinner("Answering..."):
                response, sources = chat_with_pdf(user_question, st.session_state.vector_db, client)
                
                # Add to chat history
                st.session_state.chat_history.append((user_question, response, sources))
                
                # Rerun to update chat display
                st.rerun()
    
    else:
        # Enhanced Welcome message
        st.info("👈 Upload and process a PDF document in the sidebar to start chatting!")
        
        # Feature highlights
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🔍 Smart Search
            Advanced semantic search through your entire document to find relevant information quickly.
            """)
        
        with col2:
            st.markdown("""
            ### 💬 Natural Chat
            Ask questions in plain English and get contextual answers with page references.
            """)
        
        with col3:
            st.markdown("""
            ### 📖 Source Citations
            Every answer includes source citations so you can verify and explore further.
            """)

if __name__ == "__main__":
    main()