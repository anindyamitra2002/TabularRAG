from typing import List, Dict, Any, Optional
import pandas as pd
import time
from tqdm import tqdm
import logging
from pinecone import Pinecone, ServerlessSpec
from dataclasses import dataclass
from enum import Enum
from src.table_aware_chunker import TableRecursiveChunker
from src.processor import TableProcessor
from src.llm import LLMChat
from src.embedding import EmbeddingModel
from chonkie import RecursiveRules
from src.loader import MultiFormatDocumentLoader
from dotenv import load_dotenv
import os

load_dotenv()
# API Keys
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('table_aware_rag')

class ChunkType(Enum):
    TEXT = "text_chunk"
    TABLE = "table_chunk"

@dataclass
class ProcessedChunk:
    text: str  # This will be the embedable text (table description for tables)
    chunk_type: ChunkType
    token_count: int
    markdown_table: Optional[str] = None  # Store original markdown table format
    start_index: Optional[int] = None
    end_index: Optional[int] = None

def process_documents(
    file_paths: List[str],
    chunker: TableRecursiveChunker,
    processor: TableProcessor,
    output_path: str = './output.md'
) -> List[ProcessedChunk]:
    """
    Process documents into text and table chunks
    """
    # Load documents
    loader = MultiFormatDocumentLoader(
        file_paths=file_paths,
        enable_ocr=False,
        enable_tables=True
    )
    
    # Save to markdown and read content
    with open(output_path, 'w') as f:
        for doc in loader.lazy_load():
            f.write(doc.page_content)
    
    with open(output_path, 'r') as file:
        text = file.read()
    
    # Get text and table chunks
    text_chunks, table_chunks = chunker.chunk(text)
    
    # Process chunks
    processed_chunks = []
    
    # Process text chunks
    for chunk in text_chunks:
        processed_chunks.append(
            ProcessedChunk(
                text=chunk.text,
                chunk_type=ChunkType.TEXT,
                token_count=chunk.token_count,
                start_index=chunk.start_index,
                end_index=chunk.end_index
            )
        )
    
    # Process table chunks
    table_results = processor(table_chunks)
    for table in table_results:
        # Convert table chunk to string representation if needed
        table_str = str(table["text"].text)
        
        processed_chunks.append(
            ProcessedChunk(
                text=table["table_description"],  # Use description for embedding
                chunk_type=ChunkType.TABLE,
                token_count=len(table["table_description"].split()),
                markdown_table=table_str  # Store string version of table
            )
        )
    
    return processed_chunks

class PineconeRetriever:
    def __init__(
        self,
        pinecone_client: Pinecone,
        index_name: str,
        namespace: str,
        embedding_model: Any,
        llm_model: Any
    ):
        """
        Initialize retriever with configurable models
        """
        self.pinecone = pinecone_client
        self.index = self.pinecone.Index(index_name)
        self.namespace = namespace
        self.embedding_model = embedding_model
        self.llm_model = llm_model
    
    def _prepare_query(self, question: str) -> List[float]:
        """Generate embedding for query"""
        return self.embedding_model.embed(question)
    
    def invoke(
        self,
        question: str,
        top_k: int = 5,
        chunk_type_filter: Optional[ChunkType] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve similar documents with optional filtering by chunk type
        """
        query_embedding = self._prepare_query(question)
        
        # Prepare filter if chunk type specified
        filter_dict = None
        if chunk_type_filter:
            filter_dict = {"chunk_type": chunk_type_filter.value}
        
        results = self.index.query(
            namespace=self.namespace,
            vector=query_embedding,
            top_k=top_k,
            include_values=False,
            include_metadata=True,
            filter=filter_dict
        )
        
        retrieved_docs = []
        for match in results.matches:
            doc = {
                "score": match.score,
                "chunk_type": match.metadata["chunk_type"]
            }
            
            # Handle different chunk types
            if match.metadata["chunk_type"] == ChunkType.TABLE.value:
                doc["table_description"] = match.metadata["text"]  # The embedded description
                doc["markdown_table"] = match.metadata["markdown_table"]  # Original table format
            else:
                doc["page_content"] = match.metadata["text"]
                
            retrieved_docs.append(doc)
        
        return retrieved_docs

def ingest_data(
    processed_chunks: List[ProcessedChunk],
    embedding_model: Any,
    pinecone_client: Pinecone,
    index_name: str = "vector-index",
    namespace: str = "rag",
    batch_size: int = 100
):
    """
    Ingest processed chunks into Pinecone
    """
    # Create or get index
    if not pinecone_client.has_index(index_name):
        pinecone_client.create_index(
            name=index_name,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        
        while not pinecone_client.describe_index(index_name).status['ready']:
            time.sleep(1)
    
    index = pinecone_client.Index(index_name)
    
    # Process in batches
    for i in tqdm(range(0, len(processed_chunks), batch_size)):
        batch = processed_chunks[i:i+batch_size]
        
        # Generate embeddings for the text content
        texts = [chunk.text for chunk in batch]
        embeddings = embedding_model.embed_batch(texts)
        
        # Prepare records
        records = []
        for idx, chunk in enumerate(batch):
            metadata = {
                "text": chunk.text,  # This is the description for tables
                "chunk_type": chunk.chunk_type.value,
                "token_count": chunk.token_count
            }
            
            # Add markdown table to metadata if it's a table chunk
            if chunk.markdown_table is not None:
                # Ensure the table is in string format
                metadata["markdown_table"] = str(chunk.markdown_table)
            
            records.append({
                "id": f"chunk_{i + idx}",
                "values": embeddings[idx],
                "metadata": metadata
            })
        
        # Upsert to Pinecone
        try:
            index.upsert(vectors=records, namespace=namespace)
        except Exception as e:
            logger.error(f"Error during upsert: {str(e)}")
            logger.error(f"Problematic record metadata: {records[0]['metadata']}")
            raise
            
        time.sleep(0.5)  # Rate limiting


def main():
    # Initialize components
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    chunker = TableRecursiveChunker(
        tokenizer="gpt2",
        chunk_size=512,
        rules=RecursiveRules(),
        min_characters_per_chunk=12
    )
    
    llm = LLMChat("qwen2.5:0.5b")
    embedder = EmbeddingModel("nomic-embed-text")
    
    processor = TableProcessor(
        llm_model=llm,
        embedding_model=embedder,
        batch_size=8
    )
    
    try:
        # Process documents
        processed_chunks = process_documents(
            file_paths=['/teamspace/studios/this_studio/TabularRAG/data/FeesPaymentReceipt_7thsem.pdf'],
            chunker=chunker,
            processor=processor
        )
        
        # Ingest data
        ingest_data(
            processed_chunks=processed_chunks,
            embedding_model=embedder,
            pinecone_client=pc
        )
        
        # Test retrieval
        retriever = PineconeRetriever(
            pinecone_client=pc,
            index_name="vector-index",
            namespace="rag",
            embedding_model=embedder,
            llm_model=llm
        )
        
        # # Test text-only retrieval
        # text_results = retriever.invoke(
        #     question="What is paid fees amount?",
        #     top_k=3,
        #     chunk_type_filter=ChunkType.TEXT
        # )
        # print("Text results:")
        # for result in text_results:
        #     print(result)
        # Test table-only retrieval
        # table_results = retriever.invoke(
        #     question="What is paid fees amount?",
        #     top_k=3,
        #     chunk_type_filter=ChunkType.TABLE
        # )
        # print("Table results:")
        # for result in table_results:
        #     print(result)
        
        results = retriever.invoke(
            question="What is paid fees amount?",
            top_k=3
        )
        
        for i, result in enumerate(results, 1):
            print(f"\nResult {i}:")
            if result["chunk_type"] == ChunkType.TABLE.value:
                print(f"Table Description: {result['table_description']}")
                print("Table Format:")
                print(result['markdown_table'])
            else:
                print(f"Content: {result['page_content']}")
            print(f"Score: {result['score']}")
            
    except Exception as e:
        logger.error(f"Error in pipeline: {str(e)}")

if __name__ == "__main__":
    main()