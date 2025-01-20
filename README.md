# Table-Aware Retrieval-Augmented Generation (RAG) System

## **Overview**
The Table-Aware Retrieval-Augmented Generation (RAG) System is an advanced AI-driven solution designed for handling documents containing both plain text and structured tabular data in Markdown format. By leveraging state-of-the-art language models (LLMs) and embedding models, this system extracts meaningful insights from documents and efficiently retrieves relevant information.

This solution is particularly powerful in contexts where tabular data plays a critical role, such as data analysis, financial reporting, academic research, or business intelligence. Its core capabilities include accurate chunking of content, generation of descriptive summaries for tables, and high-performance vector-based retrieval using Pinecone.

---

## **Why Use This System? (Features)**
### **1. Table-Aware Chunking**
   - The system includes a custom `TableRecursiveChunker` built on top of Chonkie's logic to preserve tables during document chunking, ensuring the integrity of tabular data and enabling accurate representation.

### **2. Automated Table Summarization**
   - Each table is processed using a large language model to generate concise, meaningful descriptions that highlight trends, patterns, and key insights.

### **3. High-Performance Retrieval**
   - Leveraging Pinecone for vector-based storage and retrieval, the system ensures fast, scalable, and precise querying.

### **4. Modular Design**
   - The system uses a highly modular architecture with separate components for chunking, processing, embedding, and querying. This design allows for easy extension or customization.

### **5. Multi-Format Document Support**
   - With the integrated `MultiFormatDocumentLoader`, the system can process documents in various formats, including plain text, PDF, and Markdown, ensuring flexibility for different use cases.

### **6. Batch Processing**
   - Supports efficient batch processing of documents, ensuring scalability for large datasets.

---

## **System Workflow/Architecture**
### **Workflow**
1. **Document Loading**:
   - Documents are loaded using `MultiFormatDocumentLoader`, which supports OCR and table extraction.

2. **Content Chunking**:
   - Text and tables are chunked separately using `TableRecursiveChunker`. Tables are preserved as independent units to maintain their structure.

3. **Table Processing**:
   - Tables are passed to the `TableProcessor`, where:
     - Descriptions are generated using a large language model.
     - Embeddings are created for the descriptions.

4. **Embedding and Storage**:
   - The processed chunks (text and table descriptions) are embedded using an embedding model and stored in Pinecone for efficient retrieval.

5. **Query and Retrieval**:
   - Queries are processed by generating embeddings, which are matched with stored vectors in Pinecone. Results can be filtered by type (text or table).

6. **Result Interpretation**:
   - Retrieved chunks (text and tables) are presented along with metadata, such as descriptions for tables.

### **System Architecture**
<image>

---

## **Installation**
### **Prerequisites**
- Python >= 3.9
- Virtual environment (optional but recommended)
- Pinecone API Key (Sign up for Pinecone to get an API key)
- Docker (if running in a containerized environment)

### **Setup Instructions**
1. **Clone the Repository**
   ```bash
   git clone https://github.com/anindyamitra2002/TabularRAG.git
   cd TabularRAG
   ```

2. **Set Up a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**
   ```bash
   python app.py
   ```

### **Running with Docker**
This system supports both CPU and GPU configurations with Docker.

#### **CPU Configuration**
1. Build and run the container:
   ```bash
   docker compose -f docker-compose.yml up --build
   ```

2. Access the application at `http://localhost:8501`.

#### **GPU Configuration**
1. Build and run the container:
   ```bash
   docker compose -f docker-compose-gpu.yml up --build
   ```

2. Access the application at `http://localhost:8501`.

---

## **Directory Structure**
```
└── TabularRAG/
    ├── README.md
    ├── Dockerfile.cpu
    ├── Dockerfile.gpu
    ├── docker-compose.yml
    ├── docker-compose-gpu.yml
    ├── app.py
    ├── compose.yaml
    ├── execute.sh
    ├── requirements.txt
    ├── test.py
    ├── data/
    │   ├── helper.md
    │   └── output.md
    └── src/
        ├── embedding.py
        ├── llm.py
        ├── loader.py
        ├── processor.py
        ├── table_aware_chunker.py
        └── vectordb.py
```

---

## **Usage**
### **Processing Documents**
1. Place the input files in a directory (e.g., `data/`).
2. Run the processing script:
   ```bash
   python app.py --input_dir ./data --output_path ./output.md
   ```
   The processed chunks (text and table descriptions) will be saved in Pinecone.

### **Querying the System**
Use the `PineconeRetriever` class to query the stored data:
```python
from src.vectordb import PineconeRetriever

retriever = PineconeRetriever(
    pinecone_client=pinecone_client,
    index_name="vector-index",
    namespace="rag",
    embedding_model=embedding_model,
    llm_model=llm_model
)

query = "What are the key insights from the sales table?"
results = retriever.invoke(query)
print(results)
```

## **Script Briefing**
### **1. `src/loader.py`**
The `loader.py` script handles document loading for multiple formats, including plain text, Markdown, and PDF. It supports OCR for PDFs and ensures robust error handling to extract both plain text and tables from documents seamlessly.

### **2. `src/table_aware_chunker.py`**
The `table_aware_chunker.py` script ensures that tables are preserved as independent units during the chunking process. It extends a base chunker to recognize and handle tables separately, using regex to extract tables from Markdown documents and balancing chunk sizes for optimal processing.

### **3. `src/processor.py`**
The `processor.py` script defines the `TableProcessor` class, which handles the summarization and embedding of tables using a large language model. It supports batch processing to efficiently handle large datasets, ensuring that table descriptions are concise and meaningful.


### **4. `src/vectordb.py`**
The `vectordb.py` script manages interactions with the Pinecone vector database, providing methods for embedding storage, retrieval, and querying. It supports data ingestion, querying, and namespace management, enabling efficient storage and retrieval of text and table embeddings.


### **5. `src/embedding.py`**
The `embedding.py` script defines the configuration and initialization of the embedding model used for vector representation. It loads a pre-trained embedding model and generates high-dimensional vectors for text and table descriptions, suitable for storage in Pinecone.

### **6. `src/llm.py`**
The `llm.py` script defines interaction with the large language model (LLM) for summarization and query answering. It connects to an external API for text generation tasks, handles token limits and retries, and provides methods to generate descriptive summaries for tables and insightful answers to user queries.
---

## **Conclusion**
The Table-Aware RAG System is a versatile and efficient solution for handling documents with both textual and tabular data. By preserving the structure of tables, generating descriptive insights, and enabling high-performance retrieval, the system bridges the gap between structured and unstructured data analysis. Its modular architecture makes it easy to extend for various use cases, such as research, data reporting, or question answering.

