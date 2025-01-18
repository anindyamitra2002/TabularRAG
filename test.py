from src.table_aware_chunker import TableRecursiveChunker
from src.processor import TableProcessor
from src.llm import LLMChat
from src.embedding import EmbeddingModel
from chonkie import RecursiveRules

# Initialize the table-aware chunker
chunker = TableRecursiveChunker(
    tokenizer="gpt2",
    chunk_size=512,
    rules=RecursiveRules(),  # Uses default rules
    min_characters_per_chunk=12
)

with open('/teamspace/studios/this_studio/TabularRAG/data/output.md', 'r') as file:
    text = file.read()

# Get both text chunks and table chunks
text_chunks, table_chunks = chunker.chunk(text)

# Process text chunks
for i, chunk in enumerate(text_chunks):
    print("Text Chunk No: ", i)
    print(f"Text chunk (Level {chunk.level}):")
    print(f"Token count: {chunk.token_count}")
    print("---")


llm = LLMChat("qwen2.5:0.5b")
embedder = EmbeddingModel("nomic-embed-text")

# Create table processor with initialized models
processor = TableProcessor(
    llm_model=llm,
    embedding_model=embedder,
    batch_size=8
)

results = processor(table_chunks)
# Process the tables
for i, table in enumerate(results):
    print(f"Table chunk No: {i}")
    print(table["text"])
    print(f"Token count: {table['table_description']}")
    print("---")
    

