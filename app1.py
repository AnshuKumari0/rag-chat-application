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
        
        vector_store = QdrantVectorStore.from_documents(
            documents=split_docs,
            url="http://localhost:6333",
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
            url="http://localhost:6333",
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
                                                    #   k=3
        
        # # Create context
        # context = "\n\n\n".join([
        #     f"Page Content: {result.page_content}\nPage Number: {result.metadata.get('page_label', 'N/A')}\nSource: {result.metadata.get('source')}" 
        #     for result in search_results
        # ])
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
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
        )
        
        return response.choices[0].message.content, search_results
    
    except Exception as e:
        return f"Error generating response: {e}", []

# Streamlit App
def main():
    st.set_page_config(
        page_title="This is RAG chat application",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 This is RAG chat application")
    st.markdown("Upload a PDF document and chat with it using AI!")
    
    # Initialize session state
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False
    if 'collection_name' not in st.session_state:
        st.session_state.collection_name = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'vector_db' not in st.session_state:
        st.session_state.vector_db = None
    
    # Initialize clients
    client = initialize_openai()
    
    # Sidebar for PDF upload
    with st.sidebar:
        st.header("📄 PDF Upload")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a PDF document to chat with"
        )
        
        if uploaded_file is not None:
            st.success(f"File uploaded: {uploaded_file.name}")
            
            # Generate collection name from filename
            collection_name = f"pdf_{uploaded_file.name.replace('.pdf', '').replace(' ', '_').lower()}"
            
            if st.button("🔄 Process PDF", type="primary"):
                with st.spinner("Processing pdf..."):
                    success, result = process_pdf(uploaded_file, collection_name)
                    
                    if success:
                        st.success(f"✅ PDF processed successfully! Created {result} chunks.")
                        st.session_state.pdf_processed = True
                        st.session_state.collection_name = collection_name
                        st.session_state.vector_db = get_vector_store(collection_name)
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"❌ Error processing PDF: {result}")
        
        # Show processing status
        if st.session_state.pdf_processed:
            st.success("✅ PDF Ready for Chat!")
            st.info(f"Collection: {st.session_state.collection_name}")
            
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()
    
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
                            for j, source in enumerate(sources[:2]):  # Show top 2 sources
                                st.write(f"**Source {j+1}:**")
                                st.write(f"Page: {source.metadata.get('page', 'N/A')}")
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
            with st.spinner("Thinking..."):
                response, sources = chat_with_pdf(user_question, st.session_state.vector_db, client)
                
                # Add to chat history
                st.session_state.chat_history.append((user_question, response, sources))
                
                # Rerun to update chat display
                st.rerun()
    
    else:
        # Welcome message
        st.info("👆 Please upload a PDF file in the sidebar to get started!")
        
        # Instructions
        # with st.container():
        #     st.markdown("""
        #     ### How to use:
        #     1. **Upload PDF**: Use the sidebar to upload your PDF document
        #     2. **Process**: Click "Process PDF" to prepare it for chat
        #     3. **Chat**: Ask questions about your document
            
        #     ### Features:
        #     - 🔍 Semantic search through your PDF
        #     - 📄 Page number references
        #     - 💬 Interactive chat interface
        #     - 📖 Source citations
        #     """)

if __name__ == "__main__":
    main()