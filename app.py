import streamlit as st
from pathlib import Path
import tempfile
import os
import time
from typing import List, Dict
from pinecone import Pinecone
from src.table_aware_chunker import TableRecursiveChunker
from src.processor import TableProcessor
from src.llm import LLMChat
from src.embedding import EmbeddingModel
from chonkie import RecursiveRules
from src.vectordb import ChunkType, process_documents, ingest_data, PineconeRetriever

# Custom CSS for better UI  
st.set_page_config(
    page_title="📚 Table RAG Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .user-message {
        background-color: #f0f2f6;
    }
    .assistant-message {
        background-color: #e8f0fe;
    }
    .st-emotion-cache-1v0mbdj.e115fcil1 {
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# Enhanced RAG Template using LangChain's ChatPromptTemplate
RAG_TEMPLATE = [
    {
        "role": "system",
        "content": """You are a knowledgeable assistant specialized in analyzing documents and tables. 
                        Your responses should be:
                        - Accurate and based on the provided context
                        - Concise (three sentences maximum)
                        - Professional yet conversational
                        - Include specific references to tables when relevant
                        
                    If you cannot find an answer in the context, acknowledge this clearly."""
    },
    {
        "role": "human",
        "content": "Context: {context}\n\nQuestion: {question}"
    }
]

def simulate_streaming_response(text: str, delay: float = 0.02) -> str:
    """Simulate streaming response by yielding chunks of text with delay."""
    words = text.split()
    result = ""
    
    for i, word in enumerate(words):
        result += word + " "
        time.sleep(delay)
        # Add punctuation pause
        if any(p in word for p in ['.', '!', '?', ',']):
            time.sleep(delay * 2)
        yield result

def clear_pinecone_index(pc, index_name="vector-index"):
    """Clear the Pinecone index and reset app state."""
    try:
        if pc.has_index(index_name):
            pc.delete_index(index_name)
            st.session_state.documents_processed = False
            st.session_state.retriever = None
            st.session_state.messages = []
            st.session_state.llm = None
            st.session_state.uploaded_files = []
        st.success("🧹 Database cleared successfully!")
    except Exception as e:
        st.error(f"❌ Error clearing database: {str(e)}")

def format_context(results: List[Dict]) -> str:
    """Format retrieved results into context string."""
    context_parts = []
    
    for result in results:
        if result.get("chunk_type") == ChunkType.TABLE.value:
            table_text = f"Table: {result['markdown_table']}"
            if result.get("table_description"):
                table_text += f"\nDescription: {result['table_description']}"
            context_parts.append(table_text)
        else:
            context_parts.append(result.get("page_content", ""))
    
    return "\n\n".join(context_parts)

def format_chat_message(message: Dict[str, str], results: List[Dict] = None) -> str:
    """Format chat message with retrieved tables in a visually appealing way."""
    content = message["content"]
    
    if results:
        for result in results:
            if result.get("chunk_type") == ChunkType.TABLE.value:
                content += "\n\n---\n\n📊 **Relevant Table:**\n" + result['markdown_table']
    
    return content

def initialize_components(pinecone_api_key: str):
    """Initialize all required components with LangChain integration."""
    try:
        # Initialize Pinecone
        pc = Pinecone(api_key=pinecone_api_key)
        
        # Initialize LangChain LLM with custom parameters
        llm = LLMChat(
            model_name="mistral:7b",
            temperature=0.3  # Lower temperature for more focused responses
        )
        st.session_state.llm = llm
        
        # Initialize LangChain Embeddings
        embedder = EmbeddingModel("nomic-embed-text")
        
        # Initialize Chunker
        chunker = TableRecursiveChunker(
            tokenizer="gpt2",
            chunk_size=512,
            rules=RecursiveRules(),
            min_characters_per_chunk=12
        )
        
        # Initialize Processor
        processor = TableProcessor(
            llm_model=llm,
            embedding_model=embedder,
            batch_size=8
        )
        
        return pc, llm, embedder, chunker, processor
    
    except Exception as e:
        st.error(f"❌ Error initializing components: {str(e)}")
        return None, None, None, None, None

def process_all_documents(uploaded_files, chunker, processor, pc, embedder):
    """Process uploaded documents with enhanced progress tracking."""
    if not uploaded_files:
        st.warning("📤 Please upload at least one document.")
        return False

    try:
        temp_dir = tempfile.mkdtemp()
        file_paths = []

        with st.status("📑 Processing Documents", expanded=True) as status:
            # Save uploaded files
            st.write("📁 Saving uploaded files...")
            for uploaded_file in uploaded_files:
                st.write(f"Saving {uploaded_file.name}...")
                file_path = Path(temp_dir) / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                file_paths.append(str(file_path))
            
            # Process documents
            st.write("🔄 Processing documents...")
            processed_chunks = process_documents(
                file_paths=file_paths,
                chunker=chunker,
                processor=processor,
                output_path='./output.md'
            )
            
            # Ingest data
            st.write("📥 Ingesting data to vector database...")
            ingest_data(
                processed_chunks=processed_chunks,
                embedding_model=embedder,
                pinecone_client=pc
            )
            
            # Setup retriever
            st.write("🎯 Setting up retriever...")
            st.session_state.retriever = PineconeRetriever(
                pinecone_client=pc,
                index_name="vector-index",
                namespace="rag",
                embedding_model=embedder,
                llm_model=st.session_state.llm
            )
            
            st.session_state.documents_processed = True
            status.update(label="✅ Processing complete!", state="complete", expanded=False)

        return True

    except Exception as e:
        st.error(f"❌ Error processing documents: {str(e)}")
        return False
    
    finally:
        # Cleanup
        for file_path in file_paths:
            try:
                os.remove(file_path)
            except Exception:
                pass
        try:
            os.rmdir(temp_dir)
        except Exception:
            pass

def main():
    st.title("📚 Table RAG Assistant")
    st.markdown("---")
    pc = None
    # Sidebar Configuration with improved styling
    with st.sidebar:
        st.title("⚙️ Configuration")
        pinecone_api_key = st.text_input("🔑 Enter Pinecone API Key:", type="password")
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🧹 Clear DB", use_container_width=True):
                clear_pinecone_index(pc)
        
        with col2:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.llm.clear_history()
                st.rerun()
        
        # Display uploaded files
        if st.session_state.uploaded_files:
            st.markdown("---")
            st.subheader("📁 Uploaded Files")
            for file in st.session_state.uploaded_files:
                st.write(f"- {file.name}")
    
    pc = None
    if not pinecone_api_key:
        st.sidebar.warning("⚠️ Please enter Pinecone API key to continue.")
        st.stop()
    
    # Initialize components if not already done
    if st.session_state.retriever is None:
        pc, llm, embedder, chunker, processor = initialize_components(pinecone_api_key)
        clear_pinecone_index(pc)
        if None in (pc, llm, embedder, chunker, processor):
            st.stop()
    
    # Document Upload Section with improved UI
    if not st.session_state.documents_processed:
        st.header("📄 Document Upload")
        st.markdown("Upload your documents to get started. Supported formats: PDF, DOCX, TXT, CSV, XLSX")
        
        uploaded_files = st.file_uploader(
            "Drop your files here",
            accept_multiple_files=True,
            type=["pdf", "docx", "txt", "csv", "xlsx"]
        )
        
        if uploaded_files:
            st.session_state.uploaded_files = uploaded_files
        
        if st.button("🚀 Process Documents", use_container_width=True):
            if process_all_documents(uploaded_files, chunker, processor, pc, embedder):
                st.success("✨ Documents processed successfully!")
    
    # Enhanced Chat Interface with Simulated Streaming
    if st.session_state.documents_processed:
        st.header("💬 Chat Interface")
        st.markdown("Ask questions about your documents and tables")
        
        # Display chat history with improved styling
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(format_chat_message(message, message.get("results")))
        
        # Chat input with simulated streaming
        if prompt := st.chat_input("Ask a question..."):
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Generate response with simulated streaming
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                
                with st.spinner("🤔 Thinking..."):
                    # Retrieve relevant content
                    results = st.session_state.retriever.invoke(
                        question=prompt,
                        top_k=3
                    )
                    
                    # Format context and get response from LLM
                    context = format_context(results)
                    chat = st.session_state.llm
                    
                    input_vars = {
                        "question": prompt,
                        "context": context
                    }

                    # Get full response first
                    full_response = chat.chat_with_template(RAG_TEMPLATE, input_vars)
                    
                    # Simulate streaming of the response
                    for partial_response in simulate_streaming_response(full_response):
                        response_placeholder.markdown(partial_response + "▌")
                    
                    # Display final response with tables
                    formatted_response = format_chat_message(
                        {"role": "assistant", "content": full_response},
                        results
                    )
                    response_placeholder.markdown(formatted_response)
                    
                    # Save to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "results": results
                    })

if __name__ == "__main__":
    main()